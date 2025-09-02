# src/dataio/dataset.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .csv_utils import (
    detect_delimiter,
    guess_filename_column,
    resolve_existing_image,
    strip_numeric_prefix,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Canonical class order required by the project
CANONICAL_CLASSES: List[str] = ["LEFT", "RIGHT", "PALM", "THUMB_DOWN", "THUMB_UP"]
_NAME_TO_INDEX = {name: i for i, name in enumerate(CANONICAL_CLASSES)}

# Normalize free-form label text to canonical names
_NORMALIZE_MAP = {
    # swipes
    "left": "LEFT", "left swipe": "LEFT", "left_swipe": "LEFT", "left swipe_new": "LEFT", "left_swipe_new": "LEFT",
    "right": "RIGHT", "right swipe": "RIGHT", "right_swipe": "RIGHT", "right swipe_new": "RIGHT", "right_swipe_new": "RIGHT",
    # palm / stop
    "palm": "PALM", "open palm": "PALM", "openpalm": "PALM", "stop": "PALM", "stop gesture": "PALM", "stop_new": "PALM",
    # thumbs up/down
    "thumb up": "THUMB_UP", "thumb_up": "THUMB_UP", "thumbs up": "THUMB_UP", "thumbsup": "THUMB_UP", "thumbs_up": "THUMB_UP",
    "thumb down": "THUMB_DOWN", "thumb_down": "THUMB_DOWN", "thumbs down": "THUMB_DOWN", "thumbs_down": "THUMB_DOWN", "thumbsdown": "THUMB_DOWN",
    # variants already canonical-ish
    "thumb_up_new": "THUMB_UP", "thumbs_up_new": "THUMB_UP",
    "thumb_down_new": "THUMB_DOWN", "thumbs_down_new": "THUMB_DOWN",
    "palm_new": "PALM",
}


@dataclass
class Sample:
    path: Path
    image: Optional[np.ndarray]
    label_text: str
    label_numeric: int
    one_hot: List[int]


class GestureDataset:
    """
    Loads train/val splits from CSVs and resolves images or frame-folders.

    On disk:
      dataset_root/
        train.csv
        val.csv
        train/train/<file_or_folder>
        val/val/<file_or_folder>
    """

    def __init__(self, dataset_root: Path | str, load_images: bool = True) -> None:
        self.dataset_root = Path(dataset_root)
        self.load_images = load_images
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    # ---------- public API ----------

    def load_split(self, split: str) -> List[Sample]:
        assert split in {"train", "val"}
        df = self._read_csv(split)

        filename_col = guess_filename_column(df)
        text_col, numeric_col = self._detect_label_columns(df, exclude={filename_col})

        base_dir = self.dataset_root / split / split  # "train/train" or "val/val"

        parsed = kept = skipped = 0
        samples: List[Sample] = []

        for _idx, row in df.iterrows():
            parsed += 1

            raw_name = str(row.get(filename_col, "")).strip()
            if not raw_name or raw_name.lower() == "nan":
                skipped += 1
                continue

            # strip any `123_` prefix then drop extension; this may be a folder name
            clean = strip_numeric_prefix(raw_name)
            clean_leaf = Path(clean).with_suffix("").name
            stem = base_dir / clean_leaf

            img_path = resolve_existing_image(stem)
            if img_path is None:
                self.logger.warning(
                    "Image not found for %s -> searched under %s",
                    clean_leaf,
                    base_dir,
                )
                skipped += 1
                continue

            # label text / id
            label_text = self._derive_label_text(row, text_col, numeric_col)
            if label_text is None:
                skipped += 1
                continue
            label_text = self._normalize_label(label_text)
            if label_text not in _NAME_TO_INDEX:
                self.logger.warning("Unknown label '%s' (row %s) -> skip", label_text, _idx)
                skipped += 1
                continue

            label_numeric = _NAME_TO_INDEX[label_text]
            one_hot = [0] * len(CANONICAL_CLASSES)
            one_hot[label_numeric] = 1

            image = None
            if self.load_images:
                image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image is None:
                    self.logger.warning("cv2.imread failed for %s -> skip row", img_path)
                    skipped += 1
                    continue

            samples.append(
                Sample(
                    path=img_path,
                    image=image,
                    label_text=label_text,
                    label_numeric=label_numeric,
                    one_hot=one_hot,
                )
            )
            kept += 1

        self.logger.info("Split %s: parsed=%d kept=%d skipped=%d", split, parsed, kept, skipped)
        return samples

    def load_all(self) -> Tuple[List[Sample], List[Sample]]:
        return self.load_split("train"), self.load_split("val")

    # ---------- internals ----------

    def _read_csv(self, split: str) -> pd.DataFrame:
        """
        Always read with header=None to prevent the first row becoming headers.
        Treat all fields as strings; we'll detect labels/filenames downstream.
        """
        csv_path = self.dataset_root / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        delim = detect_delimiter(csv_path)
        df = pd.read_csv(
            csv_path,
            sep=delim,
            header=None,          # <- important
            engine="python",
            dtype=str,             # keep as strings, robust to mixed content
        )
        # simple column names: c0, c1, c2 ...
        df.columns = [f"c{i}" for i in range(len(df.columns))]
        self.logger.info("Loaded %d rows from %s with delimiter '%s'", len(df), csv_path, delim)
        return df

    def _detect_label_columns(
        self, df: pd.DataFrame, exclude: set[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find a text label column and/or numeric label column.
        Returns (text_col_name | None, numeric_col_name | None).
        """
        text_col: Optional[str] = None
        numeric_col: Optional[str] = None

        # 1) text-like column with small vocabulary that maps to our labels
        best_card = 10**9
        for col in df.columns:
            if col in exclude:
                continue
            s = df[col].fillna("")
            # try to normalize values against our map; count how many map cleanly
            normed = s.map(self._normalize_label)
            uniq = normed[normed != ""].nunique()
            hits = normed.isin(CANONICAL_CLASSES).sum()
            if 2 <= uniq <= 20 and hits >= max(2, int(0.5 * len(s))):
                if uniq < best_card:
                    text_col = col
                    best_card = uniq

        # 2) numeric-like column with small set {0..K}
        for col in df.columns:
            if col in exclude:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            valid = s.notna().sum()
            if valid == 0:
                continue
            uniq = int(s.dropna().nunique())
            # small integer label space (e.g., 0..4)
            if 2 <= uniq <= 10:
                numeric_col = col
                break

        return text_col, numeric_col

    def _derive_label_text(
        self, row: pd.Series, text_col: Optional[str], numeric_col: Optional[str]
    ) -> Optional[str]:
        if text_col and text_col in row.index:
            return str(row[text_col])

        if numeric_col and numeric_col in row.index:
            try:
                idx = int(float(row[numeric_col]))
            except Exception:
                idx = -1
            if 0 <= idx < len(CANONICAL_CLASSES):
                return CANONICAL_CLASSES[idx]

        return None

    @staticmethod
    def _normalize_label(raw: str) -> str:
        s = str(raw).strip()
        if not s:
            return s
        key = s.replace("-", " ").replace("_", " ").strip().lower()
        key = " ".join(key.split())
        return _NORMALIZE_MAP.get(key, s.upper())
