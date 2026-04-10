from __future__ import annotations

import json
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mail_butler.config import (
    get_account_db_path,
    get_account_model_path,
    load_settings,
    resolve_account,
)
from mail_butler.storage import Database

app = typer.Typer(name="mail-butler", help="Local email management tool")
config_app = typer.Typer(help="Configuration management")
app.add_typer(config_app, name="config")

console = Console()


# -- Config commands --


@config_app.command("init")
def config_init() -> None:
    """Create a configuration file interactively."""
    from mail_butler.config import DEFAULT_CONFIG_PATHS

    config_path = DEFAULT_CONFIG_PATHS[0]
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        if not typer.confirm("Overwrite?", default=False):
            raise typer.Abort()

    name = typer.prompt("Account name (e.g., personal, work)")
    host = typer.prompt("IMAP host", default="imap.gmail.com")
    port = typer.prompt("IMAP port", default=993, type=int)
    ssl = typer.confirm("Use SSL?", default=True)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_content = f"""[general]
log_level = "INFO"

[classifier]
method = "auto"
ollama_model = "qwen3.5:latest"
confidence_threshold = 0.7
batch_size = 50

[[accounts]]
name = "{name}"
host = "{host}"
port = {port}
ssl = {"true" if ssl else "false"}
auth_method = "app_password"
folders = ["INBOX"]
"""
    config_path.write_text(config_content)
    console.print(f"\n[green]Config written to {config_path}[/green]")
    console.print("\nSet credentials via environment variables:")
    prefix = f"MAIL_BUTLER_{name.upper()}"
    console.print(f'  export {prefix}_USERNAME="your-email@example.com"')
    console.print(f'  export {prefix}_PASSWORD="your-app-password"')


@config_app.command("test")
def config_test(
    account: str | None = typer.Option(None, "--account", "-a", help="Account name"),
) -> None:
    """Test IMAP connection for an account."""
    settings = load_settings()
    acc = resolve_account(account, settings)

    if not acc.username or not acc.password:
        prefix = f"MAIL_BUTLER_{acc.name.upper()}"
        console.print("[red]Missing credentials.[/red] Set env vars:")
        console.print(f'  export {prefix}_USERNAME="..."')
        console.print(f'  export {prefix}_PASSWORD="..."')
        raise typer.Exit(1)

    from mail_butler.imap_client import MailButlerIMAPClient

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task(f"Connecting to {acc.host}...", total=None)
        try:
            with MailButlerIMAPClient(acc) as client:
                folders = client.list_folders()
        except Exception as e:
            console.print(f"\n[red]Connection failed:[/red] {e}")
            raise typer.Exit(1)

    console.print(f"\n[green]Connected to {acc.host} as {acc.username}[/green]")
    console.print(f"\nFolders ({len(folders)}):")
    for f in sorted(folders):
        console.print(f"  {f}")


# -- Accounts command --


@app.command("accounts")
def accounts_list() -> None:
    """List configured accounts and their status."""
    settings = load_settings()
    if not settings.accounts:
        console.print("[yellow]No accounts configured. Run 'mail-butler config init'.[/yellow]")
        return

    table = Table(title="Configured Accounts")
    table.add_column("Name", style="bold")
    table.add_column("Host")
    table.add_column("Emails", justify="right")
    table.add_column("Classified", justify="right")
    table.add_column("Unread", justify="right")
    table.add_column("Last Sync")

    for acc in settings.accounts:
        db_path = get_account_db_path(settings, acc.name)
        if db_path.exists():
            db = Database(db_path)
            db.initialize()
            summary = db.get_account_summary()
            db.close()
            last_sync = ""
            if summary["sync_state"]:
                last_sync = summary["sync_state"][0].get("last_sync", "")[:16]
            table.add_row(
                acc.name,
                acc.host,
                str(summary["total_emails"]),
                str(summary["classified"]),
                str(summary["unread"]),
                last_sync or "never",
            )
        else:
            table.add_row(acc.name, acc.host, "-", "-", "-", "never")

    console.print(table)


# -- Fetch command --


@app.command("fetch")
def fetch(
    account: str | None = typer.Option(None, "--account", "-a", help="Account name"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max emails to fetch"),
    full_sync: bool = typer.Option(False, "--full-sync", help="Re-fetch everything"),
) -> None:
    """Fetch new emails from IMAP and store them locally."""
    settings = load_settings()
    acc = resolve_account(account, settings)
    db_path = get_account_db_path(settings, acc.name)
    db = Database(db_path)
    db.initialize()

    from mail_butler.imap_client import MailButlerIMAPClient
    from mail_butler.parser import parse_email

    total_fetched = 0

    with MailButlerIMAPClient(acc) as client:
        folders = acc.folders or ["INBOX"]
        for folder in folders:
            since_uid = 0 if full_sync else db.get_last_uid(folder)
            console.print(f"\n[bold]{acc.name}[/bold] / {folder} (since UID {since_uid})")

            uids = client.fetch_uids(folder, since_uid=since_uid, limit=limit)
            if not uids:
                console.print("  No new messages")
                continue

            console.print(f"  Found {len(uids)} new messages")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Fetching {len(uids)} emails...", total=len(uids))
                raw_messages = client.fetch_messages(folder, uids)
                progress.update(task, completed=len(uids))

            parsed = []
            errors = 0
            for uid, raw_data in raw_messages.items():
                try:
                    email = parse_email(uid, raw_data, folder)
                    parsed.append(email)
                except Exception as e:
                    errors += 1
                    logging.debug("Failed to parse UID %d: %s", uid, e)

            if parsed:
                db.upsert_emails_batch(parsed)
                max_uid = max(e.uid for e in parsed)
                db.update_sync_state(folder, max_uid, len(parsed))
                total_fetched += len(parsed)

            console.print(
                f"  Stored {len(parsed)} emails" + (f" ({errors} parse errors)" if errors else "")
            )

    db.close()
    console.print(
        f"\n[green]Total: {total_fetched} emails fetched for account '{acc.name}'[/green]"
    )


# -- Classify command --


@app.command("classify")
def classify(
    account: str | None = typer.Option(None, "--account", "-a", help="Account name"),
    method: str = typer.Option(
        "auto", "--method", "-m", help="Classification method: rules, sklearn, llm, auto"
    ),
    reclassify: bool = typer.Option(
        False, "--reclassify", help="Re-classify already classified emails"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be classified"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Emails per batch"),
) -> None:
    """Classify emails using rules, ML, or LLM."""
    settings = load_settings()
    acc = resolve_account(account, settings)
    db_path = get_account_db_path(settings, acc.name)
    db = Database(db_path)
    db.initialize()

    if reclassify:
        emails = db.get_all_emails()
    else:
        emails = db.get_unclassified_emails(limit=10000)

    if not emails:
        console.print("[yellow]No emails to classify.[/yellow]")
        db.close()
        return

    # Parse headers_json back to dict where needed
    for e in emails:
        if isinstance(e.get("headers_json"), str):
            try:
                e["headers"] = json.loads(e["headers_json"])
            except (json.JSONDecodeError, TypeError):
                e["headers"] = {}
        elif "headers" not in e:
            e["headers"] = {}

    console.print(f"[bold]{len(emails)} emails to classify[/bold] (method: {method})")

    if dry_run:
        console.print("[yellow]Dry run - no changes will be saved.[/yellow]")

    from mail_butler.classifier import (
        LLMClassifier,
        RuleBasedClassifier,
        SklearnClassifier,
        auto_classify,
    )

    rules_clf = RuleBasedClassifier()
    sklearn_clf = None
    llm_clf = None

    if method in ("sklearn", "auto"):
        model_path = get_account_model_path(settings, acc.name)
        sklearn_clf = SklearnClassifier(model_path)
        if not sklearn_clf.load_model():
            if method == "sklearn":
                console.print(
                    "[yellow]No sklearn model found. Training from rule-based results...[/yellow]"
                )
                # First classify everything with rules to bootstrap
                rule_results = rules_clf.classify_batch(emails)
                train_data = []
                for e, r in zip(emails, rule_results):
                    train_data.append(
                        {**e, "category": r.category.value, "confidence": r.confidence}
                    )
                try:
                    sklearn_clf.train_from_rules(train_data)
                    sklearn_clf.save_model()
                    console.print("[green]sklearn model trained and saved.[/green]")
                except ValueError as err:
                    console.print(f"[red]{err}[/red]")
                    db.close()
                    raise typer.Exit(1)
            else:
                sklearn_clf = None  # auto mode: skip sklearn if no model

    if method in ("llm", "auto"):
        clf_config = settings.classifier
        llm_clf = LLMClassifier(model=clf_config.ollama_model, base_url=clf_config.ollama_base_url)
        if not llm_clf.is_available():
            if method == "llm":
                console.print("[red]Ollama is not available. Start Ollama first.[/red]")
                db.close()
                raise typer.Exit(1)
            else:
                console.print("[yellow]Ollama not available, skipping LLM fallback.[/yellow]")
                llm_clf = None

    classifications = []
    threshold = settings.classifier.confidence_threshold

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Classifying {len(emails)} emails...", total=len(emails))

        for i in range(0, len(emails), batch_size):
            batch = emails[i : i + batch_size]

            if method == "rules":
                batch_results = rules_clf.classify_batch(batch)
            elif method == "sklearn" and sklearn_clf:
                batch_results = sklearn_clf.classify_batch(batch)
            elif method == "llm" and llm_clf:
                batch_results = llm_clf.classify_batch(batch)
            else:  # auto
                batch_results = [
                    auto_classify(e, rules_clf, sklearn_clf, llm_clf, threshold) for e in batch
                ]

            classifications.extend(batch_results)
            progress.update(task, advance=len(batch))

    if not dry_run:
        db.save_classifications_batch(classifications)

        # Auto-train sklearn if we have enough rule-based results
        if method == "auto" and sklearn_clf is None:
            model_path = get_account_model_path(settings, acc.name)
            sklearn_clf = SklearnClassifier(model_path)
            train_data = []
            for e, c in zip(emails, classifications):
                if c.method.value == "rules" and c.confidence >= threshold:
                    train_data.append(
                        {**e, "category": c.category.value, "confidence": c.confidence}
                    )
            if len(train_data) >= 10:
                try:
                    sklearn_clf.train_from_rules(train_data)
                    sklearn_clf.save_model()
                    n = len(train_data)
                    console.print(
                        f"[green]sklearn model auto-trained on {n} high-confidence results.[/green]"
                    )
                except ValueError:
                    pass

    # Summary
    from collections import Counter

    cat_counts = Counter(c.category.value for c in classifications)
    method_counts = Counter(c.method.value for c in classifications)

    table = Table(title="Classification Results")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")

    for cat, count in cat_counts.most_common():
        pct = f"{count / len(classifications) * 100:.1f}%"
        table.add_row(cat, str(count), pct)

    console.print(table)
    console.print(f"\nMethods used: {dict(method_counts)}")
    if dry_run:
        console.print("[yellow]Dry run - nothing saved.[/yellow]")
    else:
        console.print(f"[green]{len(classifications)} classifications saved.[/green]")

    db.close()


# -- Stats command --


@app.command("stats")
def stats(
    account: str | None = typer.Option(None, "--account", "-a", help="Account name"),
    period: str = typer.Option(
        "last-30d", "--period", "-p", help="Period: last-7d, last-30d, last-90d, all"
    ),
    all_accounts: bool = typer.Option(False, "--all-accounts", help="Show stats for all accounts"),
) -> None:
    """Show email statistics."""
    settings = load_settings()

    period_days: int | None = None
    if period != "all":
        try:
            period_days = int(period.replace("last-", "").replace("d", ""))
        except ValueError:
            console.print(
                f"[red]Invalid period: {period}. Use last-7d, last-30d, last-90d, or all.[/red]"
            )
            raise typer.Exit(1)

    from mail_butler.stats import StatsAggregator

    accounts_to_show = settings.accounts if all_accounts else [resolve_account(account, settings)]

    for acc in accounts_to_show:
        db_path = get_account_db_path(settings, acc.name)
        if not db_path.exists():
            console.print(
                f"[yellow]No data for account '{acc.name}'. Run 'mail-butler fetch' first.[/yellow]"
            )
            continue

        db = Database(db_path)
        db.initialize()
        aggregator = StatsAggregator(db)
        s = aggregator.compute_stats(period_days=period_days)

        period_label = f"last {period_days} days" if period_days else "all time"

        # Summary panel
        summary = (
            f"Total: {s.total_emails}  |  "
            f"Unread: {s.unread_count}  |  "
            f"Classified: {s.classified_count}  |  "
            f"Spam+Promo ratio: {s.spam_ratio:.1%}"
        )
        console.print(Panel(summary, title=f"[bold]{acc.name}[/bold] - {period_label}"))

        # Category distribution
        if s.category_distribution:
            cat_table = Table(title="Category Distribution")
            cat_table.add_column("Category", style="bold")
            cat_table.add_column("Count", justify="right")
            cat_table.add_column("%", justify="right")
            cat_table.add_column("Unread", justify="right", style="red")
            cat_table.add_column("", min_width=20)

            total = sum(s.category_distribution.values())
            for cat in sorted(
                s.category_distribution, key=lambda c: s.category_distribution[c], reverse=True
            ):
                count = s.category_distribution[cat]
                pct = count / total * 100 if total > 0 else 0
                unread = s.unread_by_category.get(cat, 0)
                bar = "\u2588" * int(pct / 2)
                cat_table.add_row(cat.value, str(count), f"{pct:.1f}%", str(unread), bar)
            console.print(cat_table)

        # Sub-category breakdown
        if s.sub_category_distribution:
            sub_table = Table(title="Sub-category Breakdown")
            sub_table.add_column("Category / Sub-category", style="bold")
            sub_table.add_column("Count", justify="right")
            sub_table.add_column("%", justify="right")

            total_classified = s.classified_count or 1
            for key, count in sorted(
                s.sub_category_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                pct = count / total_classified * 100
                sub_table.add_row(key, str(count), f"{pct:.1f}%")
            console.print(sub_table)

        # Top senders
        if s.top_senders:
            sender_table = Table(title="Top 15 Senders")
            sender_table.add_column("Sender", style="bold")
            sender_table.add_column("Count", justify="right")

            for sender, count in s.top_senders[:15]:
                sender_table.add_row(sender, str(count))
            console.print(sender_table)

        # Peak hours
        if s.emails_per_hour:
            hours_sorted = sorted(s.emails_per_hour.items(), key=lambda x: x[1], reverse=True)[:5]
            console.print(f"\nPeak hours: {', '.join(f'{h}:00 ({c})' for h, c in hours_sorted)}")

        if s.oldest_email and s.newest_email:
            console.print(f"Date range: {s.oldest_email:%Y-%m-%d} to {s.newest_email:%Y-%m-%d}")

        db.close()
        console.print()


@app.command("version")
def version() -> None:
    """Show mail-butler version."""
    console.print("[bold]mail-butler[/bold] v0.1.0")


if __name__ == "__main__":
    app()
