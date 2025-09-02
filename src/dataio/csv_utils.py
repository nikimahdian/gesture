# src/dataio/csv_utils.py
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

IMAGE_EXTS: Sequence[str] = (".jpg", ".jpeg", ".png")


def detect_delimiter(path: Path | str) -> str:
    """
    Robust delimiter detection with a preference for ';' if sniffing is inconclusive.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="ignore")
    sample = text[: 64 * 1024]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
        delim = dialect.delimiter or ";"
    except Exception:
        delim = ";"

    if delim in {",", ";"}:
        comma = sample.count(",")
        semi = sample.count(";")
        if comma == semi == 0:
            delim = ";"
        elif comma > semi:
            delim = ","
        else:
            delim = ";"

    logger.info("Detected delimiter '%s' for %s", delim, p)
    return delim


# ----- filename heuristics -----

_IMG_PAT = re.compile(r"\.(jpe?g|png)$", re.IGNORECASE)
_WIN_LIKE_STEM = re.compile(r"^win_\d{8}_\d{2}_\d{2}_\d{2}_pro_", re.IGNORECASE)


def _looks_like_filename(value: str) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    if _IMG_PAT.search(v):
        return True
    if "\\" in v or "/" in v:
        return True
    if _WIN_LIKE_STEM.match(v.replace(" ", "_")):
        return True
    underscores = v.count("_")
    digits = sum(ch.isdigit() for ch in v)
    return (underscores >= 3 and digits >= 6)


def guess_filename_column(df: pd.DataFrame) -> str:
    """
    Choose the first column whose values look like filenames / paths / stems.
    Works whether the CSV had headers or not (we read header=None upstream).
    """
    candidate_cols: list[tuple[float, str]] = []
    for col in df.columns:
        series = df[col]
        # sample up to 200 non-null values
        sample_vals = [str(x) for x in series.dropna().head(200).tolist()]
        if not sample_vals:
            continue
        hits = sum(_looks_like_filename(x) for x in sample_vals)
        score = hits / max(1, len(sample_vals))
        candidate_cols.append((score, str(col)))

    candidate_cols.sort(key=lambda t: (-t[0], df.columns.get_loc(t[1])))
    chosen = candidate_cols[0][1] if candidate_cols and candidate_cols[0][0] > 0 else str(df.columns[0])
    logger.info("Detected filename column: %s", chosen)
    return chosen


# ----- name cleaning & resolution -----

def strip_numeric_prefix(name: str) -> str:
    """Remove a single leading '123_' prefix exactly once."""
    return re.sub(r"^\d+_", "", name, count=1)


def _slug(s: str) -> str:
    """Normalize to compare names ignoring spaces/underscores/hyphens and case."""
    return re.sub(r"[\s_\-]+", "", s).lower()


def resolve_existing_image(stem: Path) -> Optional[Path]:
    """
    Resolve an image given a *stem* that could be:
      - a bare stem with no extension,
      - a full filename,
      - a directory containing many frames.

    Strategy:
      1) If 'stem' is an existing file with a valid image extension -> return it.
      2) If 'stem' has no suffix, probe '.jpg', '.jpeg', '.png'.
      3) If still missing:
         a) If 'stem' is a directory -> pick the middle frame within it.
         b) Else fuzzy-match a sibling directory (space/underscore/hyphen-insensitive),
            and pick the middle frame from it.
    """
    stem = Path(stem)

    # 1) Already an image file?
    if stem.is_file() and stem.suffix.lower() in IMAGE_EXTS:
        return stem

    # 2) Probe common extensions
    if stem.suffix == "":
        for ext in IMAGE_EXTS:
            p = stem.with_suffix(ext)
            if p.exists():
                return p

    # 3a) Directory containing frames
    if stem.exists() and stem.is_dir():
        imgs = sorted([*stem.glob("*.jpg"), *stem.glob("*.jpeg"), *stem.glob("*.png")])
        if imgs:
            return imgs[len(imgs) // 2]

    # 3b) Fuzzy match sibling directory
    parent = stem.parent
    if parent.exists():
        target = _slug(stem.name)
        for d in parent.iterdir():
            if d.is_dir() and _slug(d.name) == target:
                imgs = sorted([*d.glob("*.jpg"), *d.glob("*.jpeg"), *d.glob("*.png")])
                if imgs:
                    return imgs[len(imgs) // 2]

    return None
