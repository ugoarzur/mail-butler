from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from mail_butler.models import Classification, ParsedEmail

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    id TEXT PRIMARY KEY,
    message_id TEXT,
    folder TEXT NOT NULL,
    uid INTEGER NOT NULL,
    subject TEXT,
    sender_address TEXT NOT NULL,
    sender_name TEXT,
    recipients_json TEXT,
    date_sent TEXT,
    is_read INTEGER DEFAULT 0,
    is_flagged INTEGER DEFAULT 0,
    has_attachments INTEGER DEFAULT 0,
    content_preview TEXT,
    headers_json TEXT,
    size_bytes INTEGER DEFAULT 0,
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails(sender_address);
CREATE INDEX IF NOT EXISTS idx_emails_date ON emails(date_sent);
CREATE INDEX IF NOT EXISTS idx_emails_folder ON emails(folder);
CREATE INDEX IF NOT EXISTS idx_emails_message_id ON emails(message_id);

CREATE TABLE IF NOT EXISTS classifications (
    email_id TEXT PRIMARY KEY REFERENCES emails(id),
    category TEXT NOT NULL,
    confidence REAL NOT NULL,
    method TEXT NOT NULL,
    sub_category TEXT,
    model_version TEXT,
    classified_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_class_category ON classifications(category);

CREATE TABLE IF NOT EXISTS sync_state (
    folder TEXT PRIMARY KEY,
    last_uid INTEGER DEFAULT 0,
    last_sync TEXT,
    total_messages INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


def _make_email_id(uid: int, folder: str) -> str:
    return f"{folder}:{uid}"


class Database:
    """SQLite storage for a single mail account."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        # Set schema version if empty
        cur = self._conn.execute("SELECT version FROM schema_version")
        if cur.fetchone() is None:
            self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        self._conn.commit()
        logger.info("Database initialized at %s", self.db_path)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Email operations --

    def upsert_email(self, email: ParsedEmail) -> None:
        email_id = _make_email_id(email.uid, email.folder)
        self.conn.execute(
            """INSERT OR REPLACE INTO emails
            (id, message_id, folder, uid, subject, sender_address, sender_name,
             recipients_json, date_sent, is_read, is_flagged, has_attachments,
             content_preview, headers_json, size_bytes, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                email_id,
                email.message_id,
                email.folder,
                email.uid,
                email.subject,
                email.sender_address,
                email.sender_name,
                json.dumps(email.recipients),
                email.date_sent.isoformat() if email.date_sent else None,
                int(email.is_read),
                int(email.is_flagged),
                int(email.has_attachments),
                email.content_preview,
                json.dumps(email.headers),
                email.size_bytes,
                datetime.now(UTC).isoformat(),
            ),
        )

    def upsert_emails_batch(self, emails: list[ParsedEmail]) -> None:
        for em in emails:
            self.upsert_email(em)
        self.conn.commit()

    def get_unclassified_emails(self, limit: int = 100) -> list[dict]:
        """Get emails that haven't been classified yet.

        Returns dicts with email fields + raw_text not available (would need re-fetch).
        """
        cur = self.conn.execute(
            """SELECT e.id, e.message_id, e.folder, e.uid, e.subject, e.sender_address,
                      e.sender_name, e.recipients_json, e.date_sent, e.is_read, e.is_flagged,
                      e.has_attachments, e.content_preview, e.headers_json, e.size_bytes
               FROM emails e
               LEFT JOIN classifications c ON e.id = c.email_id
               WHERE c.email_id IS NULL
               ORDER BY e.date_sent DESC
               LIMIT ?""",
            (limit,),
        )
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_email_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM emails")
        return cur.fetchone()[0]

    # -- Classification operations --

    def save_classification(self, classification: Classification) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO classifications
            (email_id, category, confidence, method, sub_category, model_version, classified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                classification.email_id,
                classification.category.value,
                classification.confidence,
                classification.method.value,
                classification.sub_category,
                classification.model_version,
                datetime.now(UTC).isoformat(),
            ),
        )

    def save_classifications_batch(self, classifications: list[Classification]) -> None:
        for c in classifications:
            self.save_classification(c)
        self.conn.commit()

    def get_classified_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM classifications")
        return cur.fetchone()[0]

    # -- Sync state --

    def get_last_uid(self, folder: str) -> int:
        cur = self.conn.execute("SELECT last_uid FROM sync_state WHERE folder = ?", (folder,))
        row = cur.fetchone()
        return row[0] if row else 0

    def update_sync_state(self, folder: str, last_uid: int, total_messages: int) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO sync_state (folder, last_uid, last_sync, total_messages)
            VALUES (?, ?, ?, ?)""",
            (folder, last_uid, datetime.now(UTC).isoformat(), total_messages),
        )
        self.conn.commit()

    # -- Stats queries --

    def get_all_emails(self, period_days: int | None = None) -> list[dict]:
        """Get all emails, optionally filtered by period."""
        query = """
            SELECT e.id, e.folder, e.subject, e.sender_address, e.sender_name,
                   e.date_sent, e.is_read, e.is_flagged, e.has_attachments, e.size_bytes,
                   e.headers_json,
                   c.category, c.confidence, c.method
            FROM emails e
            LEFT JOIN classifications c ON e.id = c.email_id
        """
        params: list = []
        if period_days:
            query += " WHERE e.date_sent >= datetime('now', ?)"
            params.append(f"-{period_days} days")
        query += " ORDER BY e.date_sent DESC"
        cur = self.conn.execute(query, params)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_account_summary(self) -> dict:
        """Quick summary for the accounts listing."""
        total = self.get_email_count()
        classified = self.get_classified_count()
        unread_cur = self.conn.execute("SELECT COUNT(*) FROM emails WHERE is_read = 0")
        unread = unread_cur.fetchone()[0]
        sync_cur = self.conn.execute("SELECT folder, last_sync, total_messages FROM sync_state")
        sync_info = [
            dict(zip(["folder", "last_sync", "total_messages"], row)) for row in sync_cur.fetchall()
        ]
        return {
            "total_emails": total,
            "classified": classified,
            "unread": unread,
            "sync_state": sync_info,
        }
