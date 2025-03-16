import logging
import os
import base64
import io
from PIL import Image
import json
import threading
from mistralai import Mistral
from nnautobench.models.base_model import BaseModel
from mistralai.models import ImageURLChunk

logger = logging.getLogger(__name__)

class MistralOCRModel(BaseModel):
    def __init__(self, model_name, api_base):
        super().__init__(model_name, api_base)
        # Initialize Mistral client with API key validation
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set.")
        self.mistral_client = Mistral(api_key=api_key)
        # Thread-local storage for thread safety in parallel processing
        self.local = threading.local()

    def create_prompt(self, fields, descriptions=None, image_paths=None, ctx=[], input_text=None, layout="vision_default"):
        """Store image path and fields for use in predict."""
        self.local.current_image_path = image_paths[0] if image_paths else None
        self.local.current_fields = fields
        # Return a placeholder message; actual prompt is built in predict
        return [{"role": "user", "content": "Extract fields"}]

    def predict(self, messages, conf_score_method="prob"):
        """Make a prediction using the Mistral OCR and chat APIs."""
        # Set up image path and fields (assuming these are set elsewhere in the code)
        image_path = self.local.current_image_path
        if not image_path:
            raise ValueError("No image path provided")

        # Encode image to base64
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.png':
            with Image.open(image_path) as image:
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                output = io.BytesIO()
                image.save(output, format='JPEG')
                image_data = output.getvalue()
            mime_type = 'image/jpeg'
        elif ext in ('.jpg', '.jpeg'):
            with open(image_path, "rb") as f:
                image_data = f.read()
            mime_type = 'image/jpeg'
        else:
            raise ValueError(f"Unsupported image extension: {ext}")

        base64_data_url = f"data:{mime_type};base64,{base64.b64encode(image_data).decode()}"

        # Step 1: Process with OCR
        ocr_response = self.mistral_client.ocr.process(
            document=ImageURLChunk(image_url=base64_data_url),
            model="mistral-ocr-latest",
        )
        if not ocr_response.pages:
            raise ValueError("OCR processing returned no pages.")
        ocr_content = ocr_response.pages[0].markdown

        # Step 2: Prepare prompt (simplified; adjust based on your needs)
        prompt = f"Extract information from this OCR content:\n\n{ocr_content}\n\nReturn the result as a JSON object."

        # Step 3: Chat completion
        chat_response = self.mistral_client.chat.complete(
            model="mistral-large-latest",  # Updated to a valid model
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        # Log the response for debugging
        logger.info(f"chat_response type: {type(chat_response)}")
        logger.info(f"chat_response content: {chat_response}")

        # Handle the response
        if isinstance(chat_response, str):
            try:
                content = json.loads(chat_response)
                logger.info("Parsed chat_response as JSON string")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse chat_response: {chat_response}")
                raise ValueError(f"Invalid JSON response: {e}")
        elif hasattr(chat_response, 'choices'):
            content = chat_response.choices[0].message.content
        else:
            logger.error(f"Unexpected chat_response structure: {chat_response}")
            raise ValueError("API response lacks expected 'choices' attribute")

        # Mock usage and confidence scores (adjust as needed)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        conf_score = {}

        return content, usage, conf_score

    def post_process(self, content):
        """Reuse the base post-processing to ensure JSON compatibility."""
        return super().post_process(content)