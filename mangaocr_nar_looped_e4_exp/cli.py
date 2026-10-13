"""mangaocr-nar-looped-e4-exp [--loop 1|2|3|all] [--device DEV] [--model-dir DIR] crop.png ...
-> one "path<TAB>text" line per crop ("path<TAB>loop1<TAB>loop2<TAB>loop3" with --loop all); all crops given in one call are run
as one batch."""
from __future__ import annotations

import argparse
import sys

from . import MangaOcrNarLoopedE4Exp


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mangaocr-nar-looped-e4-exp")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--loop", default="3", choices=("1", "2", "3", "all"))
    ap.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu (default: cuda if available, else cpu)")
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args(argv)
    loop = a.loop if a.loop == "all" else int(a.loop)
    ocr = MangaOcrNarLoopedE4Exp(loop, a.device, a.model_dir)
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(encoding="utf-8")
    for path, text in zip(a.images, ocr(list(a.images))):
        out.write(f"{path}\t" + ("\t".join(text) if loop == "all" else text) + "\n")


if __name__ == "__main__":
    main()
