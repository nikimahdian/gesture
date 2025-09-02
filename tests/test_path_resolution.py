from pathlib import Path
from src.dataio.csv_utils import resolve_existing_image

def test_path_resolution_with_extensions(tmp_path: Path):
    for ext in [".jpg",".jpeg",".png"]:
        p = tmp_path / f"f{ext}"
        p.write_bytes(b"fake")
        assert resolve_existing_image(tmp_path / "f") == p
        p.unlink()
