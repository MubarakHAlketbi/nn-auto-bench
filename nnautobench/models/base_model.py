from __future__ import annotations

import json
import logging
import os
import random
import time
from abc import ABC

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
        self.model_name = model_name
        self.api_base = api_base
        self.client = self._create_client()
        if rate_limit_max_requests and rate_limit_window_seconds:
            self.rate_limiter = RateLimiter(rate_limit_max_requests, rate_limit_window_seconds)
        else:
            self.rate_limiter = None

    def _create_client(self):
        api_key = os.getenv("BASE_API_KEY")
        if not api_key:
            logger.warning("BASE_API_KEY environment variable not set. API calls may fail.")
            api_key = "EMPTY"
        return OpenAI(api_key=api_key, base_url=self.api_base)

    def is_retryable(self, exception):
        return False

    def call_api(self, func, *args, **kwargs):
        if self.rate_limiter:
            self.rate_limiter.wait_for_token()
        max_retries = 5
        delay = 1
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self.is_retryable(e):
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries ({max_retries}) reached for {self.model_name}: {str(e)}")
                        raise
                    wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {self.model_name} after {wait_time:.2f}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    raise

    def completions_with_backoff(self, **kwargs):
        return self.call_api(self.client.chat.completions.create, **kwargs)

    def predict(self, messages, conf_score_method):
        try:
            response = self.completions_with_backoff(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=3000,
            )
            content = response.choices[0].message.content
            usage = response.usage.dict() if hasattr(response, 'usage') else {}
            conf_score = {}
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise
        return content, usage, conf_score

    def post_process(self, content):
        return clean_gpt_response(content)

    def evaluate(self, prompt: str, expected_response: str) -> tuple[bool, float]:
        """Default evaluation method; override if needed."""
        logger.warning(f"evaluate not implemented for {self.model_name}; returning default response")
        return False, 0.0