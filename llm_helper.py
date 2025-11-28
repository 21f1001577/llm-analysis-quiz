import logging
import json
import re
from typing import List, Dict, Any
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

class LLMHelper:
    """LLM wrapper with MODE A/B support and strict typing."""

    def __init__(self, config):

        if isinstance(config, type):
            config = config()

        self.config = config

        self.api_key = getattr(config, "OPENAI_API_KEY", None)
        self.base_url = getattr(config, "OPENAI_BASE_URL", None)

        self.model_default = getattr(config, "OPENAI_MODEL", "gpt-4o-mini")
        self.model_light = getattr(config, "OPENAI_MODEL_LIGHT", self.model_default)
        self.model_strict = getattr(config, "OPENAI_MODEL_STRICT", self.model_default)

        logger.info(
            f"[LLMHelper] Models → default={self.model_default}, "
            f"A={self.model_light}, B={self.model_strict}"
        )

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def _choose_model(self, mode: str) -> str:
        if mode == "A":
            return self.model_light
        if mode == "B":
            return self.model_strict
        return self.model_default

    def get_completion(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        mode: str = "A"
    ) -> str | None:

        if not self.client:
            logger.error("OpenAI client not initialized")
            return None

        model = self._choose_model(mode)
        logger.info(f"[LLM] Mode={mode} | Model={model}")

        system_msg = system_message or "You are a precise extraction assistant. Output JSON only when asked."
        user_msg = prompt

        # Typed messages for Pylance
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        # ============================================================
        # 🔍 DEBUG: FULL PROMPT LOGGING
        # ============================================================
        try:
            logger.warning("========== LLM REQUEST ==========")
            logger.warning(f"[Model] {model}")

            logger.warning("---- SYSTEM MESSAGE ----")
            logger.warning(system_msg)

            logger.warning("---- USER MESSAGE ----")
            logger.warning(user_msg)

            logger.warning("---- RAW PAYLOAD ----")
            logger.warning(json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }, indent=2))
            logger.warning("=================================")
        except Exception:
            logger.warning("Failed to log LLM request payload.")

        # ============================================================
        # API CALL
        # ============================================================
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # ============================================================
            # 🔍 DEBUG: FULL RAW RESPONSE LOGGING
            # ============================================================
            try:
                logger.warning("========== LLM RESPONSE ==========")
                logger.warning(json.dumps(response.model_dump(), indent=2))
                logger.warning("==================================")
            except Exception:
                logger.warning("Failed to serialize response fully:")
                logger.warning(str(response))

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"[LLM] Completion error: {e}")
            return None


    # ---------------------------------------------------------------
    # JSON Extraction Helper
    # ---------------------------------------------------------------
    def extract_json_from_text(self, text: str) -> Dict[str, Any] | None:
        """Extract the first valid JSON object."""
        try:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
        return None
