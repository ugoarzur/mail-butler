from __future__ import annotations

import logging
from types import TracebackType

from imapclient import IMAPClient

from mail_butler.config import AccountConfig

logger = logging.getLogger(__name__)

FETCH_BATCH_SIZE = 100


class MailButlerIMAPClient:
    """Wrapper around IMAPClient for mailbox access."""

    def __init__(self, account: AccountConfig) -> None:
        self.account = account
        self._client: IMAPClient | None = None

    def connect(self) -> None:
        if self._client is not None:
            return
        logger.info(
            "Connecting to %s:%d (account: %s)",
            self.account.host,
            self.account.port,
            self.account.name,
        )
        self._client = IMAPClient(
            host=self.account.host,
            port=self.account.port,
            ssl=self.account.ssl,
        )
        self._client.login(self.account.username, self.account.password)
        logger.info("Connected successfully")

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.logout()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> MailButlerIMAPClient:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.disconnect()

    @property
    def client(self) -> IMAPClient:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first or use as context manager.")
        return self._client

    def list_folders(self) -> list[str]:
        """List all IMAP folders."""
        raw_folders = self.client.list_folders()
        return [name for _flags, _delimiter, name in raw_folders]

    def fetch_uids(self, folder: str, since_uid: int = 0, limit: int | None = None) -> list[int]:
        """Get UIDs in a folder, optionally starting after since_uid."""
        self.client.select_folder(folder, readonly=True)
        if since_uid > 0:
            criteria = [b"UID", f"{since_uid + 1}:*".encode()]
        else:
            criteria = [b"ALL"]
        uids = self.client.search(criteria)
        # Filter out UIDs <= since_uid (IMAP search can include the boundary)
        uids = [u for u in uids if u > since_uid]
        if limit:
            uids = uids[:limit]
        return uids

    def fetch_messages(self, folder: str, uids: list[int]) -> dict[int, dict]:
        """Fetch full message data for given UIDs in batches.

        Uses BODY.PEEK[] to avoid marking messages as read.
        Returns {uid: {b'BODY[]': bytes, b'FLAGS': tuple, b'RFC822.SIZE': int, ...}}
        """
        self.client.select_folder(folder, readonly=True)
        result: dict[int, dict] = {}
        for i in range(0, len(uids), FETCH_BATCH_SIZE):
            batch = uids[i : i + FETCH_BATCH_SIZE]
            logger.debug(
                "Fetching batch %d-%d of %d UIDs",
                i + 1,
                min(i + FETCH_BATCH_SIZE, len(uids)),
                len(uids),
            )
            fetched = self.client.fetch(
                batch, ["BODY.PEEK[]", "FLAGS", "RFC822.SIZE", "INTERNALDATE"]
            )
            result.update(fetched)
        return result
