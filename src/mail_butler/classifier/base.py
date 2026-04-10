from __future__ import annotations

from abc import ABC, abstractmethod

from mail_butler.models import Classification


class BaseClassifier(ABC):
    @abstractmethod
    def classify(
        self,
        email_id: str,
        subject: str | None,
        sender: str,
        content_preview: str,
        headers: dict[str, str],
    ) -> Classification: ...

    @abstractmethod
    def classify_batch(self, emails: list[dict]) -> list[Classification]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
