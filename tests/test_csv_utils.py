from pathlib import Path
import pandas as pd
from src.dataio.csv_utils import detect_delimiter, guess_filename_column, strip_numeric_prefix, resolve_existing_image

def test_strip_numeric_prefix():
    assert strip_numeric_prefix("12_WIN_abc") == "WIN_abc"
    assert strip_numeric_prefix("WIN_abc") == "WIN_abc"

def test_guess_filename_column(tmp_path: Path):
    df = pd.DataFrame({
        "meta": ["foo","bar","baz"],
        "file": ["A_B.jpg", "C_D.png", "E_F.jpeg"],
        "label": ["Left Swipe", "Palm", "Thumb_down"]
    })
    assert guess_filename_column(df) == "file"

def test_detect_delimiter(tmp_path: Path):
    p = tmp_path / "sample.csv"
    p.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    assert detect_delimiter(p) == ";"

def test_resolve_existing_image(tmp_path: Path):
    stem = tmp_path / "img"
    # create .jpeg
    (tmp_path / "img.jpeg").write_bytes(b"fake")
    assert resolve_existing_image(stem) == (tmp_path / "img.jpeg")
    # if ext already present and exists
    assert resolve_existing_image(tmp_path / "img.jpeg") == (tmp_path / "img.jpeg")
