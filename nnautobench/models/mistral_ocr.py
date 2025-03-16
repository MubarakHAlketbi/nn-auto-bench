from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from mistralai import Mistral, ImageURLChunk

from .base_model import BaseModel
from nnautobench.utils.image_utils import encode_image_base64
from nnautobench.utils.prompt_utils import create_field_extraction_prompt_ocr, get_sample_output

load_dotenv()
logger = logging.getLogger(__name__)

class MistralOCRModel(BaseModel):
    def __init__(self, model_name, api_base):
        # model_name and api_base are passed from MODEL_CONFIGS but not used directly
        self.model_name = model_name  # For compatibility with BaseModel
        self.client = self._create_client()

    def _create_client(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.warning("MISTRAL_API_KEY environment variable not set. MistralOCRModel may not work.")
        return Mistral(api_key=api_key)

    def get_ocr_text(self, image_path):
        """Extract OCR text from an image using mistral-ocr-latest."""
        base64_image = encode_image_base64(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"
        ocr_response = self.client.ocr.process(
            document=ImageURLChunk(image_url=data_url),
            model="mistral-ocr-latest",
        )
        # Assuming ocr_response.text contains the extracted text (adjust based on actual API response)
        return getattr(ocr_response, 'text', '')

    def create_prompt(self, fields, descriptions=None, image_paths=None, ctx=[], input_text=None, layout="vision_default"):
        """Generate messages using OCR text extracted from the image."""
        # Extract OCR text from the provided image path
        if image_paths and image_paths[0]:
            ocr_text = self.get_ocr_text(image_paths[0])
        else:
            ocr_text = ""

        # Create prompt using OCR text, similar to GPT4oModel (text-based)
        actual_few_shot = len(ctx)
        question = create_field_extraction_prompt_ocr(
            fields,
            descriptions,
            disable_output_format=False,
            ocr_text=ocr_text,
        )
        if actual_few_shot == 0:
            messages = [{"role": "user", "content": question}]
        else:
            messages = []
            for i in range(actual_few_shot):
                sample_output = get_sample_output(fields, ctx[i]["accepted"])
                answer = json.dumps(sample_output, ensure_ascii=False)
                sample_ocr_text = ctx[i].get("text", "")
                sample_prompt = create_field_extraction_prompt_ocr(
                    fields,
                    descriptions,
                    disable_output_format=False,
                    ocr_text=sample_ocr_text,
                )
                messages.append({"role": "user", "content": sample_prompt})
                messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": question})
        return messages

    def predict(self, messages, conf_score_method):
        """Use mistral-large-latest to extract fields from the OCR text."""
        try:
            response = self.client.chat.completions.create(
                model="mistral-large-latest",
                messages=messages,
                temperature=0,
                max_tokens=3000,
            )
            content = response.choices[0].message.content
            usage = response.usage.dict() if hasattr(response, 'usage') else {}
            conf_score = {}  # Confidence scoring not implemented here
        except Exception as e:
            logger.error(f"Error in MistralOCRModel predict: {e}")
            raise
        return content, usage, conf_score