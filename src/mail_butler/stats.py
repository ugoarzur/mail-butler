from __future__ import annotations

from collections import Counter
from datetime import datetime

from mail_butler.models import EmailCategory, MailboxStats
from mail_butler.storage import Database


class StatsAggregator:
    """Compute statistics from classified emails in the database."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def compute_stats(self, period_days: int | None = 30) -> MailboxStats:
        all_emails = self.db.get_all_emails(period_days=period_days)
        classified = [e for e in all_emails if e.get("category")]

        total = len(all_emails)
        unread = sum(1 for e in all_emails if not e.get("is_read"))
        classified_count = len(classified)

        # Category distribution
        cat_counter: Counter[str] = Counter()
        unread_by_cat: Counter[str] = Counter()
        sub_cat_counter: Counter[str] = Counter()
        for e in classified:
            cat = e["category"]
            cat_counter[cat] += 1
            sub_cat = e.get("sub_category")
            if sub_cat:
                sub_cat_counter[f"{cat}/{sub_cat}"] += 1
            if not e.get("is_read"):
                unread_by_cat[cat] += 1

        category_distribution = {
            EmailCategory(k): v
            for k, v in cat_counter.items()
            if k in EmailCategory.__members__.values()
        }
        unread_by_category = {
            EmailCategory(k): v
            for k, v in unread_by_cat.items()
            if k in EmailCategory.__members__.values()
        }

        # Top senders
        sender_counter: Counter[str] = Counter()
        for e in all_emails:
            sender_counter[e["sender_address"]] += 1
        top_senders = sender_counter.most_common(20)

        # Spam ratio
        spam_count = cat_counter.get(EmailCategory.SPAM, 0) + cat_counter.get(
            EmailCategory.PROMOTION, 0
        )
        spam_ratio = spam_count / classified_count if classified_count > 0 else 0.0

        # Emails per day
        emails_per_day: Counter[str] = Counter()
        emails_per_hour: Counter[int] = Counter()
        dates: list[datetime] = []

        for e in all_emails:
            date_str = e.get("date_sent")
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str)
                dates.append(dt)
                emails_per_day[dt.strftime("%Y-%m-%d")] += 1
                emails_per_hour[dt.hour] += 1
            except (ValueError, TypeError):
                continue

        oldest = min(dates) if dates else None
        newest = max(dates) if dates else None

        return MailboxStats(
            total_emails=total,
            unread_count=unread,
            classified_count=classified_count,
            category_distribution=category_distribution,
            unread_by_category=unread_by_category,
            top_senders=top_senders,
            spam_ratio=spam_ratio,
            sub_category_distribution=dict(sub_cat_counter),
            emails_per_day=dict(emails_per_day),
            emails_per_hour=dict(emails_per_hour),
            oldest_email=oldest,
            newest_email=newest,
        )
