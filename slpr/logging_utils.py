from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(run_dir: Path, level: str = "INFO") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"

    # Clear existing handlers to avoid duplicates on re-run
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    log_level = getattr(logging, level.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)

    root.setLevel(log_level)
    root.addHandler(fh)
    root.addHandler(ch)
