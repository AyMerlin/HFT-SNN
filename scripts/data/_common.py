"""Argument helpers shared by the retrieval scripts."""

from __future__ import annotations

import argparse
from datetime import date, timedelta


def iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO date: {text!r}") from None


def day_range(start: date, end: date) -> list[date]:
    if end < start:
        raise SystemExit(f"--end {end} is before --start {start}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def args_dict(args: argparse.Namespace) -> dict:
    return {k: (v.isoformat() if isinstance(v, date) else str(v) if not isinstance(v, (int, float, bool)) else v)
            for k, v in vars(args).items()}
