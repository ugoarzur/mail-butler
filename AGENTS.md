# Project

This project is a mail butler program that analyses all the mails in a mailbox, categorizes mails and optimizes the classification folders storage. The main goal is to have less garbage, unread mails, split important mails from less important ones.

See de [Readme](./README.md) project for more general project informations

## Tech decisions

- Python 3.12+, using uv as the package manager.
- CLI built with Typer + Rich (no web server for now, but could evolve into a browser plugin).
- Local LLM classification via Ollama (qwen3.5:latest) -- everything stays local, no data leaves the machine.
- Pydantic for config validation and data models.
- SQLite (stdlib sqlite3) for storage -- one database file per mail account for full isolation. No ORM, no Alembic -- schema managed via simple CREATE TABLE IF NOT EXISTS statements.
- scikit-learn (TF-IDF + Multinomial Naive Bayes) for fast ML classification, one model per account.
- 3-tier classification chain: rule-based heuristics -> sklearn -> Ollama LLM fallback.
- I am interested in machine learning -- if you find better and cleaner ways to do classification, explain why and how it can improve the process.

## Code quality

- Ruff for linting (rules: E, F, I, UP) and formatting (line-length 100).
- Vulture for dead code detection (`uvx vulture src/mail_butler/ --min-confidence 80`).
- Credentials via environment variables only, never in config files.
- Parameterized SQL queries everywhere, no f-strings in SQL.
