from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class EmailCategory(StrEnum):
    NEWSLETTER = "newsletter"
    PROMOTION = "promotion"
    PERSONAL = "personal"
    WORK = "work"
    TRANSACTIONAL = "transactional"
    SPAM = "spam"
    SOCIAL = "social"
    NOTIFICATION = "notification"
    UNKNOWN = "unknown"


class ClassificationMethod(StrEnum):
    RULES = "rules"
    SKLEARN = "sklearn"
    LLM = "llm"


@dataclass
class ParsedEmail:
    message_id: str
    uid: int
    folder: str
    subject: str | None
    sender_address: str
    sender_name: str | None
    recipients: list[str]
    date_sent: datetime | None
    is_read: bool
    is_flagged: bool
    has_attachments: bool
    content_preview: str
    headers: dict[str, str]
    size_bytes: int = 0
    raw_text_body: str | None = None


@dataclass
class Classification:
    email_id: str
    category: EmailCategory
    confidence: float
    method: ClassificationMethod
    sub_category: str | None = None
    model_version: str | None = None


@dataclass
class MailboxStats:
    total_emails: int
    unread_count: int
    classified_count: int
    category_distribution: dict[EmailCategory, int] = field(default_factory=dict)
    unread_by_category: dict[EmailCategory, int] = field(default_factory=dict)
    top_senders: list[tuple[str, int]] = field(default_factory=list)
    spam_ratio: float = 0.0
    emails_per_day: dict[str, int] = field(default_factory=dict)
    emails_per_hour: dict[int, int] = field(default_factory=dict)
    oldest_email: datetime | None = None
    newest_email: datetime | None = None
