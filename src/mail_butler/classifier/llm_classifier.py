from __future__ import annotations

import json
import logging

import httpx

from mail_butler.classifier.base import BaseClassifier
from mail_butler.models import Classification, ClassificationMethod, EmailCategory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an email classifier. Classify the email into exactly one category:
- newsletter: recurring informational content (blogs, digests, updates)
- promotion: commercial/marketing offers, sales, deals
- personal: direct personal communication
- work: professional/work-related communication
- transactional: order confirmations, receipts, shipping, password resets
- spam: unsolicited junk mail
- social: social media notifications (follows, likes, comments)
- notification: automated system notifications (alerts, reminders)

Respond with ONLY a JSON object, no other text:
{"category": "<category>", "confidence": <0.0-1.0>}"""

USER_PROMPT_TEMPLATE = """From: {sender}
Subject: {subject}
Preview: {preview}"""

VALID_CATEGORIES = {c.value for c in EmailCategory if c != EmailCategory.UNKNOWN}


class LLMClassifier(BaseClassifier):
    """Ollama-based classifier for nuanced email classification."""

    def __init__(self, model: str = "qwen3.5:latest", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    @property
    def name(self) -> str:
        return "llm"

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(m.get("name", "").startswith(self.model.split(":")[0]) for m in models)
        except httpx.ConnectError:
            return False

    def classify(
        self,
        email_id: str,
        subject: str | None,
        sender: str,
        content_preview: str,
        headers: dict[str, str],
    ) -> Classification:
        prompt = USER_PROMPT_TEMPLATE.format(
            sender=sender,
            subject=subject or "(no subject)",
            preview=content_preview[:500] if content_preview else "(empty)",
        )

        try:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.1},
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            response_text = resp.json().get("message", {}).get("content", "")
            return self._parse_response(response_text, email_id)
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            return self._fallback(email_id)
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out for email %s", email_id)
            return self._fallback(email_id)
        except Exception:
            logger.warning("LLM classification failed for email %s", email_id, exc_info=True)
            return self._fallback(email_id)

    def classify_batch(self, emails: list[dict]) -> list[Classification]:
        """Classify emails sequentially (LLM is the bottleneck)."""
        return [
            self.classify(
                email_id=e["id"],
                subject=e.get("subject"),
                sender=e["sender_address"],
                content_preview=e.get("content_preview", ""),
                headers=e.get("headers", {}),
            )
            for e in emails
        ]

    def _parse_response(self, response_text: str, email_id: str) -> Classification:
        """Parse JSON response from the LLM."""
        # Try to extract JSON from the response (LLM might add extra text)
        text = response_text.strip()

        # Find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("No JSON found in LLM response: %s", text[:200])
            return self._fallback(email_id)

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in LLM response: %s", text[:200])
            return self._fallback(email_id)

        category_str = data.get("category", "").lower().strip()
        if category_str not in VALID_CATEGORIES:
            logger.warning("Invalid category from LLM: %s", category_str)
            return self._fallback(email_id)

        confidence = float(data.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))

        return Classification(
            email_id=email_id,
            category=EmailCategory(category_str),
            confidence=confidence,
            method=ClassificationMethod.LLM,
            model_version=self.model,
        )

    def _fallback(self, email_id: str) -> Classification:
        return Classification(
            email_id=email_id,
            category=EmailCategory.UNKNOWN,
            confidence=0.0,
            method=ClassificationMethod.LLM,
            model_version=self.model,
        )
