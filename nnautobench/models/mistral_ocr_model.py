import threading
import json
import os
from pathlib import Path
from mistralai import Mistral
from nnautobench.models.base_model import BaseModel

class MistralOCRModel(BaseModel):
    def __init__(self, model_name, api_base):
        super().__init__(model_name, api_base)
        # Initialize Mistral client
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        # Thread-local storage for thread safety
        self.local = threading.local()

    def create_prompt(self, fields, descriptions=None, image_paths=None, ctx=[], input_text=None, layout="vision_default"):
        """Store image path and fields for use in predict."""
        self.local.current_image_path = image_paths[0] if image_paths else None
        self.local.current_fields = fields
        # Return a placeholder message; actual prompt is built in predict
        return [{"role": "user", "content": "Extract fields"}]

    def predict(self, messages, conf_score_method):
        """Handle OCR and field extraction."""
        image_path = self.local.current_image_path
        fields = self.local.current_fields

        if not image_path:
            raise ValueError("No image path provided")

        # Step 1: Upload image and perform OCR
        image_file = Path(image_path)
        uploaded_file = self.mistral_client.files.upload(
            file={
                "file_name": image_file.stem,
                "content": image_file.read_bytes(),
            },
            purpose="ocr",
        )
        signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        ocr_response = self.mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
        )
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
        conf_score = {}  # Adjust if confidence scores are needed

        return content, usage, conf_score

    def post_process(self, content):
        """Reuse the base post-processing to ensure JSON compatibility."""
        return super().post_process(content)