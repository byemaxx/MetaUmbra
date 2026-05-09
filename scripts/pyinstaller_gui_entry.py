from __future__ import annotations

import multiprocessing as mp
import os

from metaumbra.gui import main


def _run_smoke_imports() -> None:
    import bz2
    import ctypes
    import decimal
    import lzma
    import metaumbra.digest
    import metaumbra.scoring
    import pandas.plotting
    import pyarrow.parquet
    import sqlite3
    import ssl


if __name__ == "__main__":
    mp.freeze_support()
    if os.environ.get("METAUMBRA_GUI_SMOKE_IMPORTS") == "1":
        _run_smoke_imports()
        raise SystemExit(0)
    main()
