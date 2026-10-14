"""Build the public repository folder for Paper 2 from this working repository.

What goes in: code (src/), the manuscript source and PDF (paper/), figures, the result summaries (results/*.json:
per-system aggregates, validation and certification summaries), the price tables, the data-pipeline index and
summary (metadata only), the honestboard dependency pointer and a README. What stays out: the compact per-task
records derived from HAL's archives (data/hal/runs/) and the per-task boards (data/boards/), until the HAL team has
answered the licence question posted on the dataset's discussion board; and anything under notes/.

usage: python release/export_public.py [OUT_DIR]      (default: ../frontier-leaderboards-public beside this repo)
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "frontier-leaderboards-public"

KEEP_FILES = ["paper/main.tex", "paper/refs.bib", "paper/numbers.tex", "paper/main.pdf", "paper/supplement.tex", "paper/supplement.pdf",
              "data/hal/model_prices.json", "data/hal/cache_prices.json", "data/hal/index.json", "data/boards/hal_summary.json",
              "requirements.txt", "LICENSE", "README.md", "CITATION.cff"]
KEEP_GLOBS = ["paper/tab_*.tex", "src/*.py", "figures/*.pdf", "figures/*.png", "results/*.json", "release/export_public.py"]


def main():
    if OUT.exists():
        for child in OUT.iterdir():
            if child.name == ".git":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel in KEEP_FILES:
        src = ROOT / rel
        if src.exists():
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    for pat in KEEP_GLOBS:
        for src in ROOT.glob(pat):
            dst = OUT / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    (OUT / "data" / "hal" / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / "hal" / "runs" / "README.md").write_text(
        "The compact per-task records derived from HAL's trace archives are not redistributed here until the HAL team "
        "has answered the licence question (the dataset card states no licence). Rebuild them with `python src/hal_fetch.py` "
        "(it reads only the results member of each archive) and then `python src/build_hal_boards.py`.\n", encoding="utf-8")
    (OUT / ".gitignore").write_text("data/hal/runs/*.json\ndata/boards/hal_*.json\n*.aux\n*.log\n*.out\n*.bbl\n*.blg\n*.fls\n*.fdb_latexmk\n*.synctex.gz\n__pycache__/\n", encoding="utf-8")
    print(f"exported {copied} files to {OUT}")


if __name__ == "__main__":
    main()
