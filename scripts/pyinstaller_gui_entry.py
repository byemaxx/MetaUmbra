from __future__ import annotations

import multiprocessing as mp
import os


def _run_smoke_imports() -> None:
    import bz2
    import ctypes
    import decimal
    import lzma
    import metaumbra._scoring.empirical
    import metaumbra._scoring.knockoff
    import metaumbra._scoring.pooled
    import metaumbra._scoring.ranking
    import metaumbra._scoring.stats
    import metaumbra._scoring.theoretical
    import metaumbra._scoring.unit_specific
    import metaumbra.digest
    import metaumbra.scoring
    import pandas.plotting
    import PySide6.QtCore
    import PySide6.QtGui
    import PySide6.QtWidgets
    import pyarrow.parquet
    import sqlite3
    import ssl


if __name__ == "__main__":
    mp.freeze_support()
    if os.environ.get("METAUMBRA_GUI_SMOKE_IMPORTS") == "1":
        _run_smoke_imports()
        raise SystemExit(0)
    from metaumbra.gui import main

    main()
