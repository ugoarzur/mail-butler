from __future__ import annotations

import email
import email.header
import email.utils
import html.parser
import logging
import re
from datetime import UTC, datetime
from email.message import Message

from mail_butler.models import ParsedEmail

logger = logging.getLogger(__name__)

PREVIEW_MAX_LENGTH = 500

# Headers we want to preserve for later analysis (classification, mailing list detection)
INTERESTING_HEADERS = [
    "List-Unsubscribe",
    "List-ID",
    "List-Id",
    "Precedence",
    "X-Mailer",
    "X-PM-Message-Id",
    "Feedback-ID",
    "X-SES-Outgoing",
]


class _HTMLStripper(html.parser.HTMLParser):
    """Simple HTML tag stripper."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html_content: str) -> str:
    """Strip HTML tags and return plain text."""
    stripper = _HTMLStripper()
    try:
        stripper.feed(html_content)
        return stripper.get_text()
    except Exception:
        # Fallback: crude regex strip
        return re.sub(r"<[^>]+>", " ", html_content)


def decode_header_value(raw: str | None) -> str:
    """Decode an RFC 2047 encoded header value."""
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    decoded_parts = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded_parts.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(part)
    return " ".join(decoded_parts)


def _extract_body(msg: Message) -> tuple[str | None, str | None]:
    """Extract text/plain and text/html body from a MIME message.

    Returns (text_body, html_body).
    """
    text_body: str | None = None
    html_body: str | None = None

    if not msg.is_multipart():
        content_type = msg.get_content_type()
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload is None:
            return None, None
        try:
            decoded = payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            decoded = payload.decode("latin-1", errors="replace")

        if content_type == "text/plain":
            text_body = decoded
        elif content_type == "text/html":
            html_body = decoded
        return text_body, html_body

    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue
        charset = part.get_content_charset() or "utf-8"
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        try:
            decoded = payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            decoded = payload.decode("latin-1", errors="replace")

        if content_type == "text/plain" and text_body is None:
            text_body = decoded
        elif content_type == "text/html" and html_body is None:
            html_body = decoded

    return text_body, html_body


def _has_attachments(msg: Message) -> bool:
    """Check if message has attachments (without downloading them)."""
    if not msg.is_multipart():
        return False
    for part in msg.walk():
        disposition = part.get("Content-Disposition", "")
        if "attachment" in disposition.lower():
            return True
        content_type = part.get_content_type()
        if content_type not in (
            "text/plain",
            "text/html",
            "multipart/alternative",
            "multipart/mixed",
            "multipart/related",
        ):
            filename = part.get_filename()
            if filename:
                return True
    return False


def _extract_interesting_headers(msg: Message) -> dict[str, str]:
    """Extract headers useful for classification and mailing list detection."""
    headers: dict[str, str] = {}
    for header_name in INTERESTING_HEADERS:
        value = msg.get(header_name)
        if value:
            headers[header_name] = decode_header_value(value)
    return headers


def parse_email(uid: int, raw_data: dict, folder: str) -> ParsedEmail:
    """Parse raw IMAP fetch data into a ParsedEmail.

    Args:
        uid: IMAP UID of the message.
        raw_data: Dict from IMAPClient.fetch() for this UID.
        folder: IMAP folder name.
    """
    # Extract raw bytes - try multiple possible keys
    raw_bytes: bytes | None = None
    for key in (b"BODY[]", b"RFC822", "BODY[]", "RFC822"):
        if key in raw_data:
            raw_bytes = raw_data[key]
            break
    if raw_bytes is None:
        raise ValueError(f"No message body found for UID {uid}")

    msg = email.message_from_bytes(raw_bytes)

    # Sender
    from_raw = msg.get("From", "")
    sender_name_raw, sender_address = email.utils.parseaddr(from_raw)
    sender_name = decode_header_value(sender_name_raw) or None
    sender_address = sender_address.lower()

    # Subject
    subject = decode_header_value(msg.get("Subject"))

    # Recipients
    to_raw = msg.get("To", "")
    recipients = [addr for _, addr in email.utils.getaddresses([to_raw]) if addr]

    # Date
    date_sent: datetime | None = None
    date_raw = msg.get("Date")
    if date_raw:
        parsed_date = email.utils.parsedate_to_datetime(date_raw)
        date_sent = parsed_date.astimezone(UTC)
    if date_sent is None:
        # Fallback to INTERNALDATE from IMAP
        for key in (b"INTERNALDATE", "INTERNALDATE"):
            if key in raw_data and raw_data[key]:
                internal = raw_data[key]
                if isinstance(internal, datetime):
                    date_sent = internal.astimezone(UTC)
                break

    # Message-ID
    message_id = msg.get("Message-ID", f"<uid-{uid}@{folder}>")

    # Flags
    flags: tuple = ()
    for key in (b"FLAGS", "FLAGS"):
        if key in raw_data:
            flags = raw_data[key]
            break
    flag_strings = {f.decode() if isinstance(f, bytes) else str(f) for f in flags}
    is_read = "\\Seen" in flag_strings
    is_flagged = "\\Flagged" in flag_strings

    # Size
    size_bytes = 0
    for key in (b"RFC822.SIZE", "RFC822.SIZE"):
        if key in raw_data:
            size_bytes = raw_data[key]
            break

    # Body
    text_body, html_body = _extract_body(msg)
    if text_body:
        raw_text = text_body
    elif html_body:
        raw_text = strip_html(html_body)
    else:
        raw_text = ""

    content_preview = raw_text[:PREVIEW_MAX_LENGTH].strip() if raw_text else ""

    # Attachments
    has_attachments = _has_attachments(msg)

    # Interesting headers
    headers = _extract_interesting_headers(msg)

    return ParsedEmail(
        message_id=message_id,
        uid=uid,
        folder=folder,
        subject=subject or None,
        sender_address=sender_address,
        sender_name=sender_name,
        recipients=recipients,
        date_sent=date_sent,
        is_read=is_read,
        is_flagged=is_flagged,
        has_attachments=has_attachments,
        content_preview=content_preview,
        headers=headers,
        size_bytes=size_bytes,
        raw_text_body=raw_text or None,
    )
