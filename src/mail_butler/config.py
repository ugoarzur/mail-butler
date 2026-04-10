from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".config" / "mail-butler" / "config.toml",
    Path("config.toml"),
]

DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "mail-butler"


class AccountConfig(BaseModel):
    name: str
    host: str
    port: int = 993
    ssl: bool = True
    auth_method: str = "app_password"
    username: str = ""
    password: str = ""
    folders: list[str] = ["INBOX"]


class ClassifierConfig(BaseModel):
    method: str = "auto"
    ollama_model: str = "qwen3.5:latest"
    ollama_base_url: str = "http://localhost:11434"
    confidence_threshold: float = 0.7
    batch_size: int = 50


class Settings(BaseModel):
    data_dir: Path = DEFAULT_DATA_DIR
    log_level: str = "INFO"
    classifier: ClassifierConfig = ClassifierConfig()
    accounts: list[AccountConfig] = []


def get_account_db_path(settings: Settings, account_name: str) -> Path:
    return settings.data_dir / f"{account_name}.db"


def get_account_model_path(settings: Settings, account_name: str) -> Path:
    return settings.data_dir / f"{account_name}.joblib"


def resolve_account(name: str | None, settings: Settings) -> AccountConfig:
    """Resolve which account to use.

    If name is given, find it. If only one account exists, use it.
    Otherwise raise an error asking the user to specify --account.
    """
    if not settings.accounts:
        raise SystemExit("No accounts configured. Run 'mail-butler config init' first.")

    if name:
        for account in settings.accounts:
            if account.name == name:
                return account
        available = ", ".join(a.name for a in settings.accounts)
        raise SystemExit(f"Account '{name}' not found. Available: {available}")

    if len(settings.accounts) == 1:
        return settings.accounts[0]

    available = ", ".join(a.name for a in settings.accounts)
    raise SystemExit(
        f"Multiple accounts configured. Specify --account NAME. Available: {available}"
    )


def _apply_env_credentials(account: AccountConfig) -> AccountConfig:
    """Load username/password from environment variables.

    For account "personal", looks for:
      MAIL_BUTLER_PERSONAL_USERNAME
      MAIL_BUTLER_PERSONAL_PASSWORD

    Hyphens in account names are replaced with underscores
    (e.g. "work-google" -> MAIL_BUTLER_WORK_GOOGLE_*).
    """
    import os

    env_name = account.name.upper().replace("-", "_")
    prefix = f"MAIL_BUTLER_{env_name}"
    if not account.username:
        account.username = os.environ.get(f"{prefix}_USERNAME", "")
    if not account.password:
        account.password = os.environ.get(f"{prefix}_PASSWORD", "")
    return account


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from TOML config file, then overlay env vars for credentials."""
    path = config_path
    if path is None:
        for candidate in DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                path = candidate
                break

    if path is None or not path.exists():
        return Settings()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    general = raw.get("general", {})
    classifier_raw = raw.get("classifier", {})
    accounts_raw = raw.get("accounts", [])

    settings = Settings(
        data_dir=Path(general.get("data_dir", str(DEFAULT_DATA_DIR))).expanduser(),
        log_level=general.get("log_level", "INFO"),
        classifier=ClassifierConfig(**classifier_raw),
        accounts=[AccountConfig(**a) for a in accounts_raw],
    )

    # Overlay env var credentials
    for account in settings.accounts:
        _apply_env_credentials(account)

    return settings
