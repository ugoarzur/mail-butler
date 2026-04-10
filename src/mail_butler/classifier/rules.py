from __future__ import annotations

import re

from mail_butler.classifier.base import BaseClassifier
from mail_butler.models import Classification, ClassificationMethod, EmailCategory

# Known transactional sender domains
TRANSACTIONAL_DOMAINS = {
    "paypal.com",
    "stripe.com",
    "square.com",
    "venmo.com",
    "amazon.com",
    "apple.com",
    "google.com",
    "bank",
    "chase.com",
    "wellsfargo.com",
    "hsbc.com",
    "fedex.com",
    "ups.com",
    "dhl.com",
    "usps.com",
}

# Known ESP (Email Service Provider) X-Mailer values
ESP_MAILERS = {
    "mailchimp",
    "sendgrid",
    "mailgun",
    "sendinblue",
    "brevo",
    "constantcontact",
    "campaignmonitor",
    "hubspot",
    "klaviyo",
    "mailjet",
    "postmark",
    "sparkpost",
}

# Subject patterns
TRANSACTIONAL_PATTERNS = re.compile(
    r"(your order|order confirm|shipping confirm|delivery|receipt|invoice|"
    r"payment|votre commande|confirmation|facture|livraison|mot de passe|"
    r"password reset|verify your|confirm your|account statement|"
    r"your .{0,20} has shipped)",
    re.IGNORECASE,
)

PROMOTION_PATTERNS = re.compile(
    r"(\d+%\s*off|sale|deal|discount|promo|limited time|free shipping|"
    r"offre|soldes|reduction|gratuit|exclusive offer|derniere chance|"
    r"last chance|don't miss|ne manquez pas|flash sale|clearance)",
    re.IGNORECASE,
)

SOCIAL_PATTERNS = re.compile(
    r"(followed you|liked your|commented on|mentioned you|"
    r"tagged you|sent you a message|friend request|connection request|"
    r"new follower|invitation to connect)",
    re.IGNORECASE,
)

SOCIAL_DOMAINS = {
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "reddit.com",
    "github.com",
    "medium.com",
    "slack.com",
    "discord.com",
}


SUB_CATEGORY_PATTERNS: dict[EmailCategory, list[tuple[str, re.Pattern[str]]]] = {
    EmailCategory.TRANSACTIONAL: [
        ("receipt", re.compile(r"(receipt|facture|invoice)", re.IGNORECASE)),
        ("shipping", re.compile(r"(shipping|delivery|livraison|has shipped)", re.IGNORECASE)),
        ("password_reset", re.compile(r"(password reset|mot de passe)", re.IGNORECASE)),
        ("order_confirmation", re.compile(r"(your order|order confirm|votre commande)", re.IGNORECASE)),
        ("account_verification", re.compile(r"(verify your|confirm your)", re.IGNORECASE)),
        ("payment", re.compile(r"(payment|account statement)", re.IGNORECASE)),
    ],
    EmailCategory.SOCIAL: [
        ("follow", re.compile(r"(followed you|new follower)", re.IGNORECASE)),
        ("like", re.compile(r"(liked your)", re.IGNORECASE)),
        ("comment", re.compile(r"(commented on)", re.IGNORECASE)),
        ("mention", re.compile(r"(mentioned you|tagged you)", re.IGNORECASE)),
        ("message", re.compile(r"(sent you a message)", re.IGNORECASE)),
        ("connection_request", re.compile(r"(friend request|connection request|invitation to connect)", re.IGNORECASE)),
    ],
}


def _detect_sub_category(category: EmailCategory, subject: str) -> str | None:
    """Attempt to determine sub-category from subject line patterns."""
    for sub_cat, pattern in SUB_CATEGORY_PATTERNS.get(category, []):
        if pattern.search(subject):
            return sub_cat
    return None


class RuleBasedClassifier(BaseClassifier):
    """Fast, deterministic classifier based on email headers and metadata."""

    @property
    def name(self) -> str:
        return "rules"

    def classify(
        self,
        email_id: str,
        subject: str | None,
        sender: str,
        content_preview: str,
        headers: dict[str, str],
    ) -> Classification:
        subject = subject or ""
        sender_domain = sender.split("@")[-1].lower() if "@" in sender else ""
        sender_local = sender.split("@")[0].lower() if "@" in sender else sender.lower()

        has_list_unsubscribe = "List-Unsubscribe" in headers
        has_list_id = "List-ID" in headers or "List-Id" in headers
        precedence = headers.get("Precedence", "").lower()
        x_mailer = headers.get("X-Mailer", "").lower()

        # Check ESP mailer
        is_esp = any(esp in x_mailer for esp in ESP_MAILERS)

        # Rule 1: SPAM indicators (very low confidence needed - these are obvious)
        if precedence == "junk":
            return self._result(email_id, EmailCategory.SPAM, 0.9)

        # Rule 2: TRANSACTIONAL - noreply senders with transactional patterns
        is_noreply = sender_local in ("noreply", "no-reply", "donotreply", "ne-pas-repondre")
        if is_noreply and TRANSACTIONAL_PATTERNS.search(subject):
            sub = _detect_sub_category(EmailCategory.TRANSACTIONAL, subject)
            return self._result(email_id, EmailCategory.TRANSACTIONAL, 0.9, sub)

        # Rule 3: TRANSACTIONAL - known transactional domains
        if any(d in sender_domain for d in TRANSACTIONAL_DOMAINS) and TRANSACTIONAL_PATTERNS.search(
            subject
        ):
            sub = _detect_sub_category(EmailCategory.TRANSACTIONAL, subject)
            return self._result(email_id, EmailCategory.TRANSACTIONAL, 0.85, sub)

        # Rule 4: PROMOTION - List-Unsubscribe + promotional patterns
        if has_list_unsubscribe and PROMOTION_PATTERNS.search(subject):
            return self._result(email_id, EmailCategory.PROMOTION, 0.9)

        # Rule 5: PROMOTION - ESP mailer + promotional patterns
        if is_esp and PROMOTION_PATTERNS.search(subject):
            return self._result(email_id, EmailCategory.PROMOTION, 0.85)

        # Rule 6: SOCIAL - known social domains
        if any(d in sender_domain for d in SOCIAL_DOMAINS) and SOCIAL_PATTERNS.search(subject):
            sub = _detect_sub_category(EmailCategory.SOCIAL, subject)
            return self._result(email_id, EmailCategory.SOCIAL, 0.85, sub)

        # Rule 7: NEWSLETTER - List-ID present (non-promotional)
        if has_list_id and not PROMOTION_PATTERNS.search(subject):
            return self._result(email_id, EmailCategory.NEWSLETTER, 0.8)

        # Rule 8: NEWSLETTER - precedence bulk/list with List-Unsubscribe
        if precedence in ("bulk", "list") and has_list_unsubscribe:
            return self._result(email_id, EmailCategory.NEWSLETTER, 0.75)

        # Rule 9: PROMOTION - List-Unsubscribe without List-ID (likely commercial)
        if has_list_unsubscribe and is_esp:
            return self._result(email_id, EmailCategory.PROMOTION, 0.7)

        # Rule 10: NOTIFICATION - noreply sender without transactional patterns
        if is_noreply:
            return self._result(email_id, EmailCategory.NOTIFICATION, 0.7)

        # Rule 11: TRANSACTIONAL - subject patterns alone (lower confidence)
        if TRANSACTIONAL_PATTERNS.search(subject):
            sub = _detect_sub_category(EmailCategory.TRANSACTIONAL, subject)
            return self._result(email_id, EmailCategory.TRANSACTIONAL, 0.6, sub)

        # No rule matched
        return self._result(email_id, EmailCategory.UNKNOWN, 0.0)

    def classify_batch(self, emails: list[dict]) -> list[Classification]:
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

    def _result(
        self, email_id: str, category: EmailCategory, confidence: float,
        sub_category: str | None = None,
    ) -> Classification:
        return Classification(
            email_id=email_id,
            category=category,
            confidence=confidence,
            method=ClassificationMethod.RULES,
            sub_category=sub_category,
        )
