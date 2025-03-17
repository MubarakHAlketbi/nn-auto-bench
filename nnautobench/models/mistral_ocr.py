from __future__ import annotations

import json
import logging
import os
import time

from dotenv import load_dotenv
from mistralai import Mistral, ImageURLChunk

from .base_model import BaseModel
from nnautobench.utils.image_utils import encode_image_base64
from nnautobench.utils.prompt_utils import create_field_extraction_prompt_ocr, get_sample_output

load_dotenv()
logger = logging.getLogger(__name__)

class MistralOCRModel(BaseModel):
    def __init__(self, model_name, api_base, **kwargs):
        super().__init__(model_name, api_base, **kwargs)  # Pass all kwargs to BaseModel
        self.client = self._create_client()

    def _create_client(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.warning("MISTRAL_API_KEY environment variable not set. MistralOCRModel may not work.")
        return Mistral(api_key=api_key)

    def get_ocr_text(self, image_path):
        """Extract OCR text from an image using mistral-ocr-latest, aligned with API documentation."""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return ""
        base64_image = encode_image_base64(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"
        try:
            ocr_response = self.client.ocr.process(
                document=ImageURLChunk(image_url=data_url),
                model="mistral-ocr-latest",
                pages=[0],  # Explicitly specify first page for single-page images
            )
            if ocr_response.pages and len(ocr_response.pages) > 0:
                ocr_text = ocr_response.pages[0].markdown
                if not ocr_text:
                    logger.warning(f"No text extracted from {image_path}")
                else:
                    logger.debug(f"OCR text extracted from {image_path}: {ocr_text[:100]}...")
                return ocr_text or ""
            else:
                logger.warning(f"No pages in OCR response for {image_path}")
                return ""
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""

    def create_prompt(self, fields, descriptions=None, image_paths=None, ctx=[], input_text=None, layout="vision_default"):
        """Create a prompt for field extraction with improved clarity and context validation."""
        # Handle OCR text extraction
        if image_paths and image_paths[0]:
            ocr_text = self.get_ocr_text(image_paths[0])
            if not ocr_text:
                logger.warning(f"No text available for {image_paths[0]}. Using empty prompt.")
                question = "No text extracted from the image. Return empty values for all fields."
                return [{"role": "user", "content": question}]
        else:
            ocr_text = ""
            logger.warning("No image path provided; using empty OCR text.")

        actual_few_shot = len(ctx)
        # Construct the main question with clearer instructions
        question = (
            f"Consider the following document text extracted from an image:\n"
            f"--------------------------------\n"
            f"{ocr_text}\n"
            f"--------------------------------\n"
            f"Extract the following fields in JSON format: {', '.join(fields)}\n"
            f"If no relevant information is found, return an empty string ('') for that field.\n"
            f"The output should be a flattened JSON object as shown in the format below.\n"
            f"OUTPUT JSON FORMAT\n"
            f"{json.dumps({field: {'value': '..'} for field in fields}, indent=2)}"
        )

        if actual_few_shot == 0:
            messages = [{"role": "user", "content": question}]
        else:
            messages = []
            for i in range(actual_few_shot):
                sample_ocr_text = ctx[i].get("text", "")
                sample_output = get_sample_output(fields, ctx[i]["accepted"])
                if not sample_ocr_text or not sample_output:
                    logger.warning(f"Invalid context example at index {i}: empty text or output")
                    continue
                sample_answer = json.dumps(sample_output, ensure_ascii=False)
                sample_prompt = (
                    f"Consider the following document text extracted from an image:\n"
                    f"--------------------------------\n"
                    f"{sample_ocr_text}\n"
                    f"--------------------------------\n"
                    f"Extract the following fields in JSON format: {', '.join(fields)}\n"
                    f"If no relevant information is found, return an empty string ('') for that field.\n"
                    f"OUTPUT JSON FORMAT\n"
                    f"{json.dumps({field: {'value': '..'} for field in fields}, indent=2)}"
                )
                messages.append({"role": "user", "content": sample_prompt})
                messages.append({"role": "assistant", "content": sample_answer})
            messages.append({"role": "user", "content": question})
        
        logger.debug(f"Generated prompt messages: {messages}")
        return messages

    def predict(self, messages, conf_score_method):
        """Predict field values with robust error handling and separated confidence scoring."""
        max_retries = 5
        base_wait_time = 5  # Increased to avoid rate limiting
        
        # Primary prediction
        for attempt in range(max_retries):
            try:
                response = self.client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                    temperature=0,
                    max_tokens=3000,
                )
                content = response.choices[0].message.content
                usage = response.usage if hasattr(response, 'usage') else {}
                logger.debug(f"Prediction response: {content}")
                break
            except Exception as e:
                if self.is_retryable(e) and attempt < max_retries - 1:
                    wait_time = base_wait_time * (2 ** attempt)
                    logger.warning(f"Retryable error {getattr(e, 'status_code', 'unknown')}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Prediction failed after {max_retries} retries: {e}")
                    raise
        
        # Confidence score calculation (separate step)
        conf_score = {}
        if conf_score_method == "prob":
            conf_score_prompt = (
                "Provide confidence scores for each field in the previous response "
                "in JSON format: {'field_name': confidence (0-1)}"
            )
            conf_messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": conf_score_prompt}
            ]
            for attempt in range(max_retries):
                try:
                    conf_response = self.client.chat.complete(
                        model="mistral-large-latest",
                        messages=conf_messages,
                        temperature=0,
                        max_tokens=3000,
                    )
                    raw_content = conf_response.choices[0].message.content
                    logger.debug(f"Raw confidence score response: {raw_content}")
                    conf_score = json.loads(raw_content)
                    logger.debug(f"Parsed confidence scores: {conf_score}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Confidence score parsing failed. Raw response: {raw_content}")
                    conf_score = {}
                    break
                except Exception as e:
                    if self.is_retryable(e) and attempt < max_retries - 1:
                        wait_time = base_wait_time * (2 ** attempt)
                        logger.warning(f"Retryable error in conf score {getattr(e, 'status_code', 'unknown')}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Confidence score calculation failed after {max_retries} retries: {e}")
                        conf_score = {}
                        break

        return content, usage, conf_score

    def is_retryable(self, exception):
        """Check if an exception is retryable for Mistral API."""
        if not hasattr(exception, 'status_code'):
            return False
        return exception.status_code in {429, 500, 502, 503, 504}  # Rate limit, server errors, timeouts