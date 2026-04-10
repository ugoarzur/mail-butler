from __future__ import annotations

import json
import logging

from mail_butler.classifier.llm_classifier import LLMClassifier
from mail_butler.classifier.rules import RuleBasedClassifier
from mail_butler.classifier.sklearn_classifier import SklearnClassifier
from mail_butler.models import Classification, EmailCategory

logger = logging.getLogger(__name__)


def auto_classify(
    email: dict,
    rules: RuleBasedClassifier,
    sklearn: SklearnClassifier | None,
    llm: LLMClassifier | None,
    threshold: float = 0.7,
) -> Classification:
    """Chain classifiers: rules -> sklearn -> llm, escalating on low confidence."""
    headers = email.get("headers", {})
    if isinstance(headers, str):
        headers = json.loads(headers)

    # 1. Try rules first (instant)
    result = rules.classify(
        email_id=email["id"],
        subject=email.get("subject"),
        sender=email["sender_address"],
        content_preview=email.get("content_preview", ""),
        headers=headers,
    )
    if result.category != EmailCategory.UNKNOWN and result.confidence >= threshold:
        return result

    # 2. Try sklearn if available and trained
    if sklearn is not None:
        try:
            result = sklearn.classify(
                email_id=email["id"],
                subject=email.get("subject"),
                sender=email["sender_address"],
                content_preview=email.get("content_preview", ""),
                headers=headers,
            )
            if result.confidence >= threshold:
                return result
        except RuntimeError:
            pass  # Model not loaded/trained

    # 3. Fall back to LLM
    if llm is not None:
        result = llm.classify(
            email_id=email["id"],
            subject=email.get("subject"),
            sender=email["sender_address"],
            content_preview=email.get("content_preview", ""),
            headers=headers,
        )
        if result.category != EmailCategory.UNKNOWN:
            return result

    # Nothing worked - return rules result (even if UNKNOWN)
    return rules.classify(
        email_id=email["id"],
        subject=email.get("subject"),
        sender=email["sender_address"],
        content_preview=email.get("content_preview", ""),
        headers=headers,
    )


__all__ = [
    "RuleBasedClassifier",
    "SklearnClassifier",
    "LLMClassifier",
    "auto_classify",
]
