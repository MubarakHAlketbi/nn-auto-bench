from __future__ import annotations

import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod

import dotenv
from openai import OpenAI

from nnautobench.utils.rate_limiter import RateLimiter
from nnautobench.utils.common_utils import clean_gpt_response
from nnautobench.utils.conf_score_prompts import get_conf_score_prob_prompt, get_conf_score_yes_no_prompt
from nnautobench.utils.prompt_utils import create_field_extraction_prompt

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, model_name, api_base, rate_limit_max_requests=None, rate_limit_window_seconds=None, **kwargs):
        """Initialize the model with optional rate limit parameters."""
        self.model_name = model_name
        self.api_base = api_base
        self.client = self._create_client()
        if rate_limit_max_requests and rate_limit_window_seconds:
            self.rate_limiter = RateLimiter(rate_limit_max_requests, rate_limit_window_seconds)
        else:
            self.rate_limiter = None

    def _create_client(self):
        """Create an API client with the appropriate API key."""
        api_key = os.getenv("BASE_API_KEY")  # Generic API key fallback
        if not api_key:
            logger.warning("BASE_API_KEY environment variable not set. API calls may fail.")
            api_key = "EMPTY"  # Fallback if no key is set
        return OpenAI(api_key=api_key, base_url=self.api_base)

    def is_retryable(self, exception):
        """Determine if an exception is retryable. Subclasses should override this."""
        return False  # Default: no retries

    def call_api(self, func, *args, **kwargs):
        """Generic method to make API calls with rate limiting and retry logic."""
        if self.rate_limiter:
            self.rate_limiter.wait_for_token()
        max_retries = 5
        delay = 1  # Initial delay in seconds
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self.is_retryable(e):
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries ({max_retries}) reached for {self.model_name}: {str(e)}")
                        raise
                    wait_time = delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff + jitter
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {self.model_name} after {wait_time:.2f}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    raise

    def completions_with_backoff(self, **kwargs):
        """Make a completion request with rate limiting and backoff."""
        return self.call_api(self.client.chat.completions.create, **kwargs)

    def predict(self, prompt: str) -> tuple[str, dict]:
        """Generate a prediction from a prompt."""
        response = self.completions_with_backoff(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=4096,
        )
        response_text = response.choices[0].message.content
        return clean_gpt_response(response_text), {}

    def extract_fields(self, text: str, fields: list[str], format: str = "json") -> tuple[str, dict]:
        """Extract specified fields from text."""
        prompt = create_field_extraction_prompt(text, fields, format)
        response = self.completions_with_backoff(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in extracting information."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=4096,
        )
        response_text = response.choices[0].message.content
        try:
            if format == "json":
                result = json.loads(response_text)
                return response_text, result
            return response_text, {}
        except json.JSONDecodeError:
            logger.error(f"Failed to parse {format} response: {response_text}")
            return response_text, {}

    def get_confidence_score(self, prompt: str, response: str, problem_type: str = "yes_no") -> float:
        """Get a confidence score for a response."""
        conf_prompt = (
            get_conf_score_yes_no_prompt(prompt, response)
            if problem_type == "yes_no"
            else get_conf_score_prob_prompt(prompt, response)
        )
        conf_response = self.completions_with_backoff(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a confidence scoring assistant."},
                {"role": "user", "content": conf_prompt},
            ],
            temperature=0,
            max_tokens=256,
        )
        conf_text = conf_response.choices[0].message.content.strip()
        try:
            return float(conf_text)
        except ValueError:
            logger.error(f"Could not parse confidence score from: {conf_text}")
            return 0.0

    @abstractmethod
    def evaluate(self, prompt: str, expected_response: str) -> tuple[bool, float]:
        """Evaluate the model's response against an expected response."""
        pass