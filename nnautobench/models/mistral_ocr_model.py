import threading
import json
import os
import base64
from pathlib import Path
from PIL import Image
import io
from mistralai import Mistral, ImageURLChunk
from nnautobench.models.base_model import BaseModel

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

    def predict(self, messages, conf_score_method):
        """Handle OCR and field extraction for a single image."""
        image_path = self.local.current_image_path
        fields = self.local.current_fields

        if not image_path:
            raise ValueError("No image path provided")

        # Determine image type from extension and prepare data
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.png':
            # Convert PNG to JPEG
            with Image.open(image_path) as image:
                if image.mode == 'RGBA':
                    image = image.convert('RGB')  # Remove alpha channel if present
                output = io.BytesIO()
                image.save(output, format='JPEG')
                image_data = output.getvalue()
            mime_type = 'image/jpeg'
        elif ext in ('.jpg', '.jpeg'):
            # Read JPEG directly
            with open(image_path, "rb") as f:
                image_data = f.read()
            mime_type = 'image/jpeg'
        else:
            raise ValueError(f"Unsupported image extension: {ext}")

        # Encode image to base64 and create data URL
        encoded = base64.b64encode(image_data).decode()
        base64_data_url = f"data:{mime_type};base64,{encoded}"

        # Step 1: Process with OCR using ImageURLChunk
        ocr_response = self.mistral_client.ocr.process(
            document=ImageURLChunk(image_url=base64_data_url),
            model="mistral-ocr-latest",
        )
        # Validate OCR response
        if not ocr_response.pages:
            raise ValueError("OCR processing returned no pages.")
        ocr_markdown = ocr_response.pages[0].markdown

        # Step 2: Extract fields using a language model
        field_instructions = "\n".join(f"{i+1}. {field}" for i, field in enumerate(fields))
        prompt = (
            f"This is the image's OCR in markdown:\n\n{ocr_markdown}\n\n"
            f"Extract the following fields from this document:\n{field_instructions}\n\n"
            "Provide the output in JSON format with no extra commentary."
        )
        chat_response = self.mistral_client.chat.complete(
            model="ministral-8b-latest",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = chat_response.choices[0].message.content

        # Parse the response
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            parsed_content = {}

        # Mock usage and confidence scores (Mistral API may not provide these natively)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        conf_score = {}  # Placeholder; extend if confidence scores are needed

        return content, usage, conf_score

    def post_process(self, content):
        """Reuse the base post-processing to ensure JSON compatibility."""
        return super().post_process(content)