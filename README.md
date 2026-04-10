# Mail Butler

Local email management tool that connects to your mailboxes via IMAP, classifies emails using ML/LLM, and provides actionable recommendations to clean up your inbox.

## Goals

- **Zero unread emails** -- classify and prioritize everything
- **Reduce noise** -- identify newsletters, promotions and spam that pollute your inbox
- **Stats & insights** -- sender frequency, category distribution, spam ratio, temporal patterns
- **Unsubscribe recommendations** -- detect mailing lists and suggest how to unsubscribe

## Features

### MVP -- Classification + Statistics

- Connect to Gmail, Outlook, or any IMAP provider
- Fetch and parse emails incrementally (only new emails on each run)
- Classify emails into categories: newsletter, promotion, personal, work, transactional, spam, social, notification
- 3-tier classification: rule-based heuristics -> scikit-learn (TF-IDF + Naive Bayes) -> Ollama LLM fallback
- Rich terminal dashboard with stats, top senders, category distribution

### V1 -- Folder Recommendations

- Analyze current folder structure and detect misclassified emails
- Recommend optimal folder organization (category-based, hybrid, or minimal)
- Preview changes before applying, non-destructive by default

### V2 -- Cleanup & Unsubscribe

- Detect mailing lists via `List-Unsubscribe` headers (RFC 2369)
- Identify top "polluters" (senders with most unread/ignored emails)
- Generate cleanup report with unsubscribe links
- Suggest old emails to archive or delete

## Architecture

- **1 SQLite database per account** -- complete isolation between mailboxes
- **1 sklearn model per account** -- each mailbox learns its own patterns
- Local-only processing, no data leaves your machine

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.12+ |
| Package manager | uv |
| IMAP | imapclient |
| Email parsing | email (stdlib) + mail-parser |
| CLI | typer + rich |
| Config | pydantic-settings + TOML |
| ML classification | scikit-learn |
| LLM classification | Ollama (Mistral / Qwen) |
| Storage | sqlite3 (stdlib) |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) (optional, for LLM classification)

### Installation

```bash
git clone <repo-url>
cd mail-butler
uv sync
```

### Configuration

```bash
# Create config interactively
uv run mail-butler config init

# Or copy the example and edit manually
cp config.example.toml ~/.config/mail-butler/config.toml
```

Set credentials via environment variables (never in the config file):

```bash
export MAIL_BUTLER_PERSONAL_USERNAME="you@gmail.com"
export MAIL_BUTLER_PERSONAL_PASSWORD="your-app-password"
```

For Gmail, you need an [App Password](https://support.google.com/accounts/answer/185833) (requires 2FA enabled).

### Test connection

```bash
uv run mail-butler config test --account personal
```

## Usage

```bash
# List configured accounts
uv run mail-butler accounts

# Fetch new emails
uv run mail-butler fetch --account personal --limit 100

# Classify unclassified emails
uv run mail-butler classify --account personal --method auto

# View statistics
uv run mail-butler stats --account personal --period last-30d

# Compare all accounts
uv run mail-butler stats --all-accounts
```

### V1 -- Folder management

```bash
uv run mail-butler folders analyze --account personal
uv run mail-butler folders recommend --account personal --preview
```

### V2 -- Cleanup

```bash
uv run mail-butler cleanup scan --account personal
uv run mail-butler cleanup report --account personal
uv run mail-butler cleanup unsubscribe --account personal
```

## Data Storage

All data is stored locally in `~/.local/share/mail-butler/`:

```
~/.local/share/mail-butler/
  personal.db        # SQLite database for "personal" account
  personal.joblib    # sklearn model for "personal" account
  work.db            # SQLite database for "work" account
  work.joblib        # sklearn model for "work" account
```

Each account is fully isolated. Deleting an account's data = deleting its files.

## Development

### Linting and formatting

```bash
uv run ruff check src/ tests/        # Lint
uv run ruff format src/ tests/        # Auto-format
```

### Dead code detection

```bash
uvx vulture src/mail_butler/          # Full report (includes false positives at 60%)
uvx vulture src/mail_butler/ --min-confidence 80  # High-confidence only
```

Vulture detects unused functions, methods, and variables across the codebase. Ignore false positives from typer decorators, Pydantic fields, and Python protocol overrides (`__exit__`, `HTMLParser` methods).

### Tests

```bash
uv run pytest tests/ -v --cov=src/mail_butler
```
