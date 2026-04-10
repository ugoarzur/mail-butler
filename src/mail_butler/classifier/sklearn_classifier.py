from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from mail_butler.classifier.base import BaseClassifier
from mail_butler.models import Classification, ClassificationMethod, EmailCategory

logger = logging.getLogger(__name__)


def _build_text_feature(subject: str | None, content_preview: str, sender: str) -> str:
    """Combine email fields into a single text feature for TF-IDF."""
    parts = []
    if subject:
        # Weight subject higher by repeating it
        parts.append(subject)
        parts.append(subject)
    if content_preview:
        parts.append(content_preview[:500])
    if sender:
        parts.append(sender.split("@")[-1] if "@" in sender else sender)
    return " ".join(parts)


class SklearnClassifier(BaseClassifier):
    """TF-IDF + Multinomial Naive Bayes classifier."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._pipeline: Pipeline | None = None
        self._categories: list[str] = []

    @property
    def name(self) -> str:
        return "sklearn"

    def _create_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        strip_accents="unicode",
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", MultinomialNB(alpha=0.1)),
            ]
        )

    def load_model(self) -> bool:
        """Load a previously trained model. Returns False if no model exists."""
        if not self.model_path.exists():
            return False
        try:
            data = joblib.load(self.model_path)
            self._pipeline = data["pipeline"]
            self._categories = data["categories"]
            logger.info("Loaded sklearn model from %s", self.model_path)
            return True
        except Exception:
            logger.warning("Failed to load sklearn model from %s", self.model_path, exc_info=True)
            return False

    def save_model(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("No model to save. Train first.")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "categories": self._categories}, self.model_path)
        logger.info("Saved sklearn model to %s", self.model_path)

    def train(self, texts: list[str], labels: list[str]) -> None:
        """Train the classifier on labeled data."""
        if len(texts) < 10:
            raise ValueError(f"Need at least 10 training samples, got {len(texts)}")

        self._categories = sorted(set(labels))
        self._pipeline = self._create_pipeline()
        self._pipeline.fit(texts, labels)
        logger.info(
            "Trained sklearn model on %d samples with %d categories",
            len(texts),
            len(self._categories),
        )

    def train_from_rules(self, classified_emails: list[dict], min_confidence: float = 0.7) -> None:
        """Bootstrap training from rule-based classification results.

        Args:
            classified_emails: List of dicts with keys: subject, sender_address,
                              content_preview, category, confidence.
            min_confidence: Only use rule-based results above this confidence.
        """
        texts = []
        labels = []
        for e in classified_emails:
            if e.get("confidence", 0) < min_confidence:
                continue
            category = e.get("category", "")
            if category == EmailCategory.UNKNOWN:
                continue
            text = _build_text_feature(
                e.get("subject"), e.get("content_preview", ""), e.get("sender_address", "")
            )
            if text.strip():
                texts.append(text)
                labels.append(category)

        if len(texts) < 10:
            raise ValueError(
                f"Not enough high-confidence rule-based classifications for training "
                f"(got {len(texts)}, need at least 10). Classify more emails with rules first."
            )

        self.train(texts, labels)

    def classify(
        self,
        email_id: str,
        subject: str | None,
        sender: str,
        content_preview: str,
        headers: dict[str, str],
    ) -> Classification:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() or train() first.")

        text = _build_text_feature(subject, content_preview, sender)
        predicted = self._pipeline.predict([text])[0]
        probas = self._pipeline.predict_proba([text])[0]
        confidence = float(np.max(probas))

        return Classification(
            email_id=email_id,
            category=EmailCategory(predicted),
            confidence=confidence,
            method=ClassificationMethod.SKLEARN,
            model_version=f"mnb-{len(self._categories)}cat",
        )

    def classify_batch(self, emails: list[dict]) -> list[Classification]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() or train() first.")

        texts = [
            _build_text_feature(
                e.get("subject"), e.get("content_preview", ""), e.get("sender_address", "")
            )
            for e in emails
        ]
        predictions = self._pipeline.predict(texts)
        probas = self._pipeline.predict_proba(texts)

        results = []
        for i, e in enumerate(emails):
            results.append(
                Classification(
                    email_id=e["id"],
                    category=EmailCategory(predictions[i]),
                    confidence=float(np.max(probas[i])),
                    method=ClassificationMethod.SKLEARN,
                    model_version=f"mnb-{len(self._categories)}cat",
                )
            )
        return results
