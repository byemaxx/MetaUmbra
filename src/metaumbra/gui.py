from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
import warnings
from dataclasses import asdict, fields
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metaumbra import __version__
    from metaumbra.workflows import (
        DigestConfig,
        ParquetExtractionConfig,
        ScoringConfig,
        run_digest_workflow,
        run_parquet_extraction_workflow,
        run_scoring_workflow,
    )
else:
    from . import __version__
    from .workflows import (
        DigestConfig,
        ParquetExtractionConfig,
        ScoringConfig,
        run_digest_workflow,
        run_parquet_extraction_workflow,
        run_scoring_workflow,
    )

try:
    from PySide6.QtCore import QEvent, QObject, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QProgressDialog,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QSpinBox,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_BINDING = "PySide6"
except ImportError as pyside_exc:
    try:
        from PyQt5.QtCore import QEvent, QObject, Qt, QThread, pyqtSignal as Signal, pyqtSlot as Slot
        from PyQt5.QtGui import QIcon
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QMenu,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QProgressDialog,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSplitter,
            QSpinBox,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        QT_BINDING = "PyQt5"
    except ImportError as pyqt_exc:
        raise SystemExit(
            "PySide6 or PyQt5 is required to run the GUI. Install one with: "
            "pip install PySide6 or pip install PyQt5\n"
            f"PySide6 import error: {pyside_exc}\n"
            f"PyQt5 import error: {pyqt_exc}"
        ) from pyqt_exc


def _qt_value(owner, scoped_name: str, member_name: str):
    namespace = getattr(owner, scoped_name, owner)
    return getattr(namespace, member_name)


def _exec_qt_object(obj) -> int:
    exec_method = getattr(obj, "exec", None) or getattr(obj, "exec_", None)
    return exec_method()


QT_ALIGN_LEFT = _qt_value(Qt, "AlignmentFlag", "AlignLeft")
QT_ALIGN_TOP = _qt_value(Qt, "AlignmentFlag", "AlignTop")
QT_ALIGN_VCENTER = _qt_value(Qt, "AlignmentFlag", "AlignVCenter")
QT_ELIDE_MIDDLE = _qt_value(Qt, "TextElideMode", "ElideMiddle")
QT_RICH_TEXT = _qt_value(Qt, "TextFormat", "RichText")
QT_TEXT_BROWSER_INTERACTION = _qt_value(Qt, "TextInteractionFlag", "TextBrowserInteraction")
QT_TOP_RIGHT_CORNER = _qt_value(Qt, "Corner", "TopRightCorner")
QT_VERTICAL = _qt_value(Qt, "Orientation", "Vertical")
QT_CUSTOM_CONTEXT_MENU = _qt_value(Qt, "ContextMenuPolicy", "CustomContextMenu")
QT_CHECKED = _qt_value(Qt, "CheckState", "Checked")
QT_UNCHECKED = _qt_value(Qt, "CheckState", "Unchecked")
QT_ITEM_IS_EDITABLE = _qt_value(Qt, "ItemFlag", "ItemIsEditable")
QEVENT_WHEEL = _qt_value(QEvent, "Type", "Wheel")
QSIZE_IGNORED = _qt_value(QSizePolicy, "Policy", "Ignored")
QSIZE_EXPANDING = _qt_value(QSizePolicy, "Policy", "Expanding")
QSIZE_PREFERRED = _qt_value(QSizePolicy, "Policy", "Preferred")
QSCROLL_NO_FRAME = _qt_value(QScrollArea, "Shape", "NoFrame")
QFORM_LABEL_ROLE = _qt_value(QFormLayout, "ItemRole", "LabelRole")
QDIALOG_ACCEPTED = _qt_value(QDialog, "DialogCode", "Accepted")
QDIALOG_BUTTON_OK = _qt_value(QDialogButtonBox, "StandardButton", "Ok")
QDIALOG_BUTTON_CANCEL = _qt_value(QDialogButtonBox, "StandardButton", "Cancel")
QMSG_OK = _qt_value(QMessageBox, "StandardButton", "Ok")
QMSG_YES = _qt_value(QMessageBox, "StandardButton", "Yes")
QMSG_NO = _qt_value(QMessageBox, "StandardButton", "No")

if QT_BINDING == "PyQt5":
    # PyQt5/sip emits this while creating Python subclasses of Qt classes.
    # It is internal to the binding, not a deprecated API call in this module.
    warnings.filterwarnings(
        "ignore",
        message=r"sipPyTypeDict\(\) is deprecated, the extension module should use sipPyTypeDictRef\(\) instead",
        category=DeprecationWarning,
    )


RPG_ENZYMES: list[tuple[str, str]] = [
    ("1", "Arg-C"),
    ("2", "Asp-N"),
    ("3", "BNPS-Skatole"),
    ("4", "Bromelain"),
    ("5", "Caspase 1"),
    ("6", "Caspase 2"),
    ("7", "Caspase 3"),
    ("8", "Caspase 4"),
    ("9", "Caspase 5"),
    ("10", "Caspase 6"),
    ("11", "Caspase 7"),
    ("12", "Caspase 8"),
    ("13", "Caspase 9"),
    ("14", "Caspase 10"),
    ("15", "Chymotrypsin high specificity"),
    ("16", "Chymotrypsin low specificity"),
    ("17", "Clostripain"),
    ("18", "CNBr"),
    ("19", "Enterokinase"),
    ("20", "Factor Xa"),
    ("21", "Ficin"),
    ("22", "Formic acid"),
    ("23", "Glu-C"),
    ("24", "Glutamyl endopeptidase"),
    ("25", "Granzyme B"),
    ("26", "Hydroxylamine"),
    ("27", "Iodosobenzoic acid"),
    ("28", "Lys-C"),
    ("29", "Lys-N"),
    ("30", "Neutrophil elastase"),
    ("31", "NTCB"),
    ("32", "Papain"),
    ("33", "Pepsin pH 1.3"),
    ("34", "Pepsin pH >=2"),
    ("35", "Proline-endopeptidase"),
    ("36", "Proteinase K"),
    ("37", "Staphylococcal peptidase I"),
    ("38", "Thermolysin"),
    ("39", "Thrombin (PeptideCutter)"),
    ("40", "Thrombin SG"),
    ("41", "Tobacco etch virus protease"),
    ("42", "Trypsin"),
    ("43", "Asp-N Endopeptidase"),
    ("44", "ProAlanase"),
    ("45", "Elastase"),
    ("46", "aLP"),
]

WINDOWS_MAX_PROCESS_POOL_WORKERS = 60
MAX_PROCESS_COUNT = min(
    WINDOWS_MAX_PROCESS_POOL_WORKERS if sys.platform == "win32" else 64,
    max(1, os.cpu_count() or 1),
)
DEFAULT_PROCESS_COUNT = min(MAX_PROCESS_COUNT, max(1, (os.cpu_count() or 1) - 1))
APP_VERSION = __version__
ICON_PATH = Path(__file__).resolve().parent / "assets" / "metaumbra_icon.png"
FORM_LABEL_MIN_WIDTH = 150
BROWSE_BUTTON_WIDTH = 96
PRIMARY_BUTTON_MIN_WIDTH = 240


def _default_user_config_dir() -> Path:
    return Path.home() / "MetaUmbra" / "config"


def _default_gui_state_path() -> Path:
    return _default_user_config_dir() / "gui_state.json"


class DropPathLineEdit(QLineEdit):
    def __init__(self, accept_mode: str = "path"):
        super().__init__()
        self.accept_mode = accept_mode
        self.setAcceptDrops(True)

    def _extract_local_path(self, event) -> str | None:
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return None
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            if not local_path:
                continue
            if self.accept_mode == "file" and not os.path.isfile(local_path):
                continue
            if self.accept_mode == "dir" and not os.path.isdir(local_path):
                continue
            return local_path
        return None

    def dragEnterEvent(self, event) -> None:
        if self._extract_local_path(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._extract_local_path(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        local_path = self._extract_local_path(event)
        if local_path:
            self.setText(local_path)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class ElidedLabel(QLabel):
    """A single-line label that elides long text instead of expanding the window."""

    def __init__(self, text: str = ""):
        super().__init__("")
        self._full_text = ""
        self.setSizePolicy(QSIZE_IGNORED, QSIZE_PREFERRED)
        self.setMinimumWidth(0)
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._refresh_elided_text()

    def text(self) -> str:  # type: ignore[override]
        return self._full_text

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_elided_text()

    def _refresh_elided_text(self) -> None:
        width = max(0, self.contentsRect().width())
        elided = self.fontMetrics().elidedText(
            self._full_text,
            QT_ELIDE_MIDDLE,
            width,
        )
        super().setText(elided)


class FileContentTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def _extract_local_files(self, event) -> list[str]:
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return []
        files: list[str] = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            if local_path and os.path.isfile(local_path):
                files.append(local_path)
        return files

    def _read_text_file(self, path: str) -> str:
        for encoding in ("utf-8-sig", "utf-16", "mbcs"):
            try:
                return Path(path).read_text(encoding=encoding)
            except UnicodeError:
                continue
        return Path(path).read_text(encoding="utf-8-sig", errors="replace")

    def dragEnterEvent(self, event) -> None:
        if self._extract_local_files(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._extract_local_files(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        files = self._extract_local_files(event)
        if not files:
            super().dropEvent(event)
            return

        try:
            self.setPlainText("\n".join(self._read_text_file(path) for path in files))
        except OSError as exc:
            QMessageBox.warning(self, "Cannot Read File", str(exc))
        event.acceptProposedAction()


class DirectoryDropListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def _extract_local_dirs(self, event) -> list[str]:
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return []
        dirs: list[str] = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            if local_path and os.path.isdir(local_path):
                dirs.append(local_path)
        return dirs

    def dragEnterEvent(self, event) -> None:
        if self._extract_local_dirs(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._extract_local_dirs(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        dirs = self._extract_local_dirs(event)
        if dirs:
            existing = {self.item(i).text() for i in range(self.count())}
            for path in dirs:
                if path not in existing:
                    self.addItem(path)
                    existing.add(path)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class WheelChangeGuard(QObject):
    def eventFilter(self, watched, event) -> bool:
        if event.type() != QEVENT_WHEEL:
            return super().eventFilter(watched, event)
        if isinstance(watched, (QComboBox, QSpinBox, QDoubleSpinBox)):
            event.ignore()
            return True
        return super().eventFilter(watched, event)


def _make_path_row(browse_text: str, browse_handler, accept_mode: str = "path") -> tuple[QWidget, QLineEdit]:
    wrapper = QWidget()
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    line_edit = DropPathLineEdit(accept_mode=accept_mode)
    line_edit.setClearButtonEnabled(True)
    if accept_mode == "file":
        line_edit.setPlaceholderText("Drop a file here or click Browse")
    elif accept_mode == "dir":
        line_edit.setPlaceholderText("Drop a folder here or click Browse")
    else:
        line_edit.setPlaceholderText("Drop a path here or click Browse")
    button = QPushButton(browse_text)
    button.setFixedWidth(BROWSE_BUTTON_WIDTH)
    button.clicked.connect(browse_handler)
    layout.addWidget(line_edit, 1)
    layout.addWidget(button)
    return wrapper, line_edit


def _parse_required_int(raw_value: str, field_name: str) -> int:
    try:
        return int(raw_value.strip())
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def _parse_optional_int(raw_value: str, field_name: str) -> int | None:
    value = raw_value.strip()
    if not value:
        return None
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer or blank.") from exc


def _parse_required_float(raw_value: str, field_name: str) -> float:
    try:
        return float(raw_value.strip())
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _parse_optional_float(raw_value: str, field_name: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number or blank.") from exc


def _parse_text_list(raw_text: str) -> list[str]:
    values = []
    for chunk in raw_text.replace(",", "\n").splitlines():
        item = chunk.strip()
        if item:
            values.append(item)
    return values


def _parse_range_pairs(raw_value: str) -> list[list[float]]:
    cleaned = raw_value.strip()
    if not cleaned:
        return []

    ranges: list[list[float]] = []
    for part in cleaned.split(","):
        segment = part.strip()
        if not segment:
            continue
        if "-" not in segment:
            raise ValueError("Stage 2 p ranges must look like '0.005-0.02, 0.02-0.08'.")
        left, right = segment.split("-", 1)
        low = float(left.strip())
        high = float(right.strip())
        if low > high:
            raise ValueError("Each Stage 2 p range must have low <= high.")
        ranges.append([low, high])
    return ranges


def _format_range_pairs(ranges: list[list[float]]) -> str:
    return ", ".join(f"{pair[0]}-{pair[1]}" for pair in ranges if len(pair) == 2)


def _create_process_spinbox() -> QSpinBox:
    spinbox = QSpinBox()
    spinbox.setRange(1, MAX_PROCESS_COUNT)
    spinbox.setValue(DEFAULT_PROCESS_COUNT)
    return spinbox


def _create_optional_process_spinbox() -> QSpinBox:
    spinbox = QSpinBox()
    spinbox.setRange(0, MAX_PROCESS_COUNT)
    spinbox.setSpecialValueText("Same as Processes")
    spinbox.setValue(0)
    return spinbox


def _set_combo_to_data(combo: QComboBox, value: str) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            combo.setCurrentIndex(index)
            return
    combo.addItem(f"Custom ({value})", value)
    combo.setCurrentIndex(combo.count() - 1)


def _make_wrapped_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def _create_scroll_form_host() -> tuple[QScrollArea, QVBoxLayout]:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QSCROLL_NO_FRAME)

    content = QWidget()
    content.setObjectName("FormCanvas")
    layout = QVBoxLayout(content)
    layout.setContentsMargins(20, 14, 20, 16)
    layout.setSpacing(14)
    layout.setAlignment(QT_ALIGN_TOP)

    scroll.setWidget(content)
    return scroll, layout


def _polish_form_layout(form: QFormLayout) -> None:
    form.setHorizontalSpacing(14)
    form.setVerticalSpacing(10)
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFORM_LABEL_ROLE)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, QLabel):
            widget.setMinimumWidth(FORM_LABEL_MIN_WIDTH)
            widget.setAlignment(QT_ALIGN_LEFT | QT_ALIGN_TOP)


def _set_compact_control_width(widget: QWidget, width: int = 150) -> None:
    widget.setMaximumWidth(width)


def _create_compact_grid() -> QGridLayout:
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(8)
    return grid


def _add_compact_field(
    grid: QGridLayout,
    row: int,
    column: int,
    label_text: str,
    widget: QWidget,
    widget_width: int | None = 150,
) -> QLabel:
    label = QLabel(label_text)
    label.setAlignment(QT_ALIGN_LEFT | QT_ALIGN_VCENTER)
    if widget_width is not None:
        _set_compact_control_width(widget, widget_width)
    grid.addWidget(label, row, column * 2)
    grid.addWidget(widget, row, column * 2 + 1)
    return label


def _create_editable_combo(default_text: str = "", placeholder: str = "") -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    if placeholder and combo.lineEdit() is not None:
        combo.lineEdit().setPlaceholderText(placeholder)
    if default_text:
        combo.setEditText(default_text)
    return combo


def _set_editable_combo_items(
    combo: QComboBox,
    items: list[str],
    preferred_text: str = "",
    preferred_index: int | None = None,
) -> None:
    current_text = combo.currentText().strip()
    combo.blockSignals(True)
    combo.clear()
    for item in items:
        combo.addItem(item)

    target_text = current_text
    if preferred_text and preferred_text in items:
        if not target_text or target_text not in items:
            target_text = preferred_text
    elif preferred_index is not None and 0 <= preferred_index < len(items):
        if not target_text or target_text not in items:
            target_text = items[preferred_index]
    elif not target_text and items:
        target_text = items[0]

    combo.setEditText(target_text)
    combo.blockSignals(False)


def _clear_editable_combo_items(combo: QComboBox, fallback_text: str = "") -> None:
    combo.blockSignals(True)
    combo.clear()
    combo.setEditText(fallback_text)
    combo.blockSignals(False)


PARQUET_EXTENSIONS = {".parquet", ".pq"}


def _is_parquet_path(path_value: str) -> bool:
    return Path(path_value.strip()).suffix.lower() in PARQUET_EXTENSIONS


def _suggest_parquet_output_tsv_path(parquet_path: str) -> str:
    input_path = Path(parquet_path.strip())
    if not input_path.name:
        return ""
    return str(input_path.with_name(f"{input_path.stem}_peptides_for_metaumbra.tsv"))


def _read_parquet_schema_columns(parquet_path: str) -> list[str]:
    path = parquet_path.strip()
    if not path or not os.path.isfile(path):
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []

    try:
        schema = pq.read_schema(Path(path).expanduser())
    except Exception:
        return []

    return list(schema.names)


def _pick_preferred_column(columns: list[str], candidates: list[str]) -> str:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        resolved = lower_map.get(candidate.lower())
        if resolved:
            return resolved
    return ""


def _read_table_columns(table_path: str) -> list[str]:
    path = table_path.strip()
    if not path or not os.path.isfile(path):
        return []

    if _is_parquet_path(path):
        return []

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if not sample.strip():
            return []
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        except csv.Error:
            dialect = csv.excel_tab
        reader = csv.reader(handle, dialect)
        for row in reader:
            cols = [col.strip() for col in row]
            if any(cols):
                return cols
    return []


def _normalize_column_key(name: str) -> str:
    return "".join(char.lower() for char in str(name) if char.isalnum())


def _resolve_table_column(columns: list[str], preferred: str, candidates: list[str]) -> str:
    if preferred and preferred in columns:
        return preferred
    lookup: dict[str, str] = {}
    for column in columns:
        key = _normalize_column_key(column)
        if key and key not in lookup:
            lookup[key] = column
    for candidate in [preferred, *candidates]:
        if not candidate:
            continue
        resolved = lookup.get(_normalize_column_key(candidate))
        if resolved:
            return resolved
    return ""


def _strip_raw_suffix_from_sample_ids(values):
    return values.astype("string").str.strip().str.replace(r"\.raw$", "", case=False, regex=True)


def _infer_decoy_flag_value_from_values(values, configured_value: str) -> str:
    configured = str(configured_value)
    if configured == "":
        return configured
    value_set: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text != "<NA>":
            value_set.add(text)
    if configured in value_set or configured != "+":
        return configured
    for candidate in ("True", "true", "1", "decoy", "Decoy", "DECOY", "T", "t"):
        if candidate in value_set:
            return candidate
    return configured


def _drop_duplicate_pairs_with_pyarrow(df, first_col: str, second_col: str):
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to inspect parquet sample IDs.") from exc
    table = pa.Table.from_pandas(df[[first_col, second_col]], preserve_index=False)
    return table.group_by([first_col, second_col], use_threads=True).aggregate([]).to_pandas()


def _read_parquet_sample_unit_preview_rows_fast(
    path: str,
    columns: list[str],
    sample_id_col: str,
    peptide_seq_col: str,
    intensity_col: str,
    peptide_error_col: str,
    peptide_error_cutoff: float,
    peptide_decoy_flag_col: str,
    decoy_flag_value: str,
    intensity_min_value: float,
) -> list[dict[str, object]]:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    sample_col = _resolve_table_column(columns, sample_id_col, ["Run", "File.Name", "Raw.File", "Sample", "Sample.Name"])
    seq_col = _resolve_table_column(
        columns,
        peptide_seq_col,
        ["Stripped.Sequence", "Base Sequence", "Sequence", "Peptide.Sequence", "PeptideSequence"],
    )
    intensity_col = _resolve_table_column(columns, intensity_col, ["Precursor.Quantity", "Precursor.Normalised", "Intensity"])
    error_col = _resolve_table_column(columns, peptide_error_col, ["Q.Value", "QValue", "Qval", "QVal", "PEP", "FDR"])
    decoy_col = _resolve_table_column(columns, peptide_decoy_flag_col, ["Reverse", "Target/Decoy", "TargetDecoy", "Decoy"])
    required = [sample_col, seq_col, intensity_col]
    if not all(required):
        raise ValueError(
            "Unable to resolve sample, peptide sequence, and intensity columns from the parquet table."
        )

    read_cols = list(dict.fromkeys(required + [col for col in [decoy_col, error_col] if col]))
    raw_table = pq.read_table(path, columns=read_cols, use_threads=True)
    if decoy_col:
        decoy = pc.utf8_trim_whitespace(pc.cast(raw_table[decoy_col], pa.string(), safe=False))
        effective_decoy_value = _infer_decoy_flag_value_from_values(pc.unique(decoy).to_pylist(), decoy_flag_value)
        decoy_keep = pc.or_(pc.is_null(decoy), pc.not_equal(decoy, effective_decoy_value))
        raw_table = raw_table.filter(decoy_keep)

    sample = pc.utf8_trim_whitespace(pc.cast(raw_table[sample_col], pa.string(), safe=False))
    sample = pc.replace_substring_regex(sample, pattern=r"(?i)\.raw$", replacement="")
    peptide = pc.utf8_trim_whitespace(pc.cast(raw_table[seq_col], pa.string(), safe=False))
    intensity = pc.cast(raw_table[intensity_col], pa.float64(), safe=False)

    normalized = pa.table(
        {
            "sample_id": sample,
            "peptide": peptide,
            "intensity": intensity,
            "row_id": pa.array(np.arange(raw_table.num_rows, dtype=np.int64)),
        }
    )

    sample_valid = pc.and_(pc.is_valid(normalized["sample_id"]), pc.not_equal(normalized["sample_id"], ""))
    sample_rows = normalized.filter(sample_valid).select(["sample_id", "row_id"])
    if sample_rows.num_rows == 0:
        return []

    total_by_sample = sample_rows.group_by(["sample_id"], use_threads=True).aggregate(
        [("sample_id", "count"), ("row_id", "min")]
    )
    total_df = total_by_sample.to_pandas().rename(
        columns={"sample_id_count": "n_total_rows", "row_id_min": "_first_row"}
    )

    valid_mask = pc.and_(
        sample_valid,
        pc.and_(
            pc.and_(pc.is_valid(normalized["peptide"]), pc.not_equal(normalized["peptide"], "")),
            pc.and_(
                pc.is_valid(normalized["intensity"]),
                pc.greater(normalized["intensity"], float(intensity_min_value)),
            ),
        ),
    )
    if error_col:
        error = pc.cast(raw_table[error_col], pa.float64(), safe=False)
        valid_mask = pc.and_(
            valid_mask,
            pc.and_(pc.is_valid(error), pc.less_equal(error, float(peptide_error_cutoff))),
        )

    valid_pairs = normalized.filter(valid_mask).select(["sample_id", "peptide"])
    if valid_pairs.num_rows:
        unique_pairs = valid_pairs.group_by(["sample_id", "peptide"], use_threads=True).aggregate([])
        valid_by_sample = unique_pairs.group_by(["sample_id"], use_threads=True).aggregate([("peptide", "count")])
        valid_df = valid_by_sample.to_pandas().rename(columns={"peptide_count": "n_valid_peptides"})
    else:
        valid_df = pd.DataFrame({"sample_id": [], "n_valid_peptides": []})

    preview_df = total_df.merge(valid_df, on="sample_id", how="left")
    preview_df["n_valid_peptides"] = preview_df["n_valid_peptides"].fillna(0).astype(int)
    preview_df["n_total_rows"] = preview_df["n_total_rows"].astype(int)
    preview_df = preview_df.sort_values("_first_row", kind="mergesort")

    return [
        {
            "included": bool(int(row.n_valid_peptides) > 0),
            "sample_id": str(row.sample_id),
            "analysis_unit_id": str(row.sample_id),
            "n_total_rows": int(row.n_total_rows),
            "n_valid_peptides": int(row.n_valid_peptides),
        }
        for row in preview_df.itertuples(index=False)
    ]


def _read_delimited_table_for_columns(path: str, columns: list[str]):
    import pandas as pd

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        sep = dialect.delimiter
    except csv.Error:
        sep = "\t"
    return pd.read_csv(path, sep=sep, usecols=columns, dtype="string")


def _read_sample_unit_preview_rows(
    peptide_table_path: str,
    sample_id_col: str,
    peptide_seq_col: str,
    intensity_col: str,
    peptide_error_col: str,
    peptide_error_cutoff: float,
    peptide_decoy_flag_col: str,
    decoy_flag_value: str,
    intensity_min_value: float,
    intensity_min_quantile: float,
) -> list[dict[str, object]]:
    import pandas as pd

    path = str(Path(peptide_table_path).expanduser())
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Please choose an existing peptide table first.")

    if not 0.0 <= float(intensity_min_quantile) <= 1.0:
        raise ValueError("Minimum within-sample intensity quantile must be between 0 and 1.")

    is_parquet_input = _is_parquet_path(path)
    if is_parquet_input:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to inspect parquet sample IDs.") from exc
        columns = list(pq.read_schema(path).names)
        if float(intensity_min_quantile) == 0.0:
            try:
                return _read_parquet_sample_unit_preview_rows_fast(
                    path=path,
                    columns=columns,
                    sample_id_col=sample_id_col,
                    peptide_seq_col=peptide_seq_col,
                    intensity_col=intensity_col,
                    peptide_error_col=peptide_error_col,
                    peptide_error_cutoff=peptide_error_cutoff,
                    peptide_decoy_flag_col=peptide_decoy_flag_col,
                    decoy_flag_value=decoy_flag_value,
                    intensity_min_value=intensity_min_value,
                )
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
                pass
        sample_col = _resolve_table_column(columns, sample_id_col, ["Run", "File.Name", "Raw.File", "Sample", "Sample.Name"])
        seq_col = _resolve_table_column(
            columns,
            peptide_seq_col,
            ["Stripped.Sequence", "Base Sequence", "Sequence", "Peptide.Sequence", "PeptideSequence"],
        )
        intensity_col = _resolve_table_column(columns, intensity_col, ["Precursor.Quantity", "Precursor.Normalised", "Intensity"])
        error_col = _resolve_table_column(columns, peptide_error_col, ["Q.Value", "QValue", "Qval", "QVal", "PEP", "FDR"])
        decoy_col = _resolve_table_column(columns, peptide_decoy_flag_col, ["Reverse", "Target/Decoy", "TargetDecoy", "Decoy"])
        required = [sample_col, seq_col, intensity_col]
        if not all(required):
            raise ValueError(
                "Unable to resolve sample, peptide sequence, and intensity columns from the parquet table."
            )
        read_cols = list(dict.fromkeys(required + [col for col in [decoy_col, error_col] if col]))
        df = pq.read_table(path, columns=read_cols, use_threads=True).to_pandas()
    else:
        columns = _read_table_columns(path)
        sample_col = _resolve_table_column(columns, sample_id_col, ["Run", "File.Name", "Raw.File", "Sample", "Sample.Name"])
        seq_col = _resolve_table_column(
            columns,
            peptide_seq_col,
            ["Stripped.Sequence", "Base Sequence", "Sequence", "Peptide.Sequence", "PeptideSequence"],
        )
        intensity_col = _resolve_table_column(columns, intensity_col, ["Precursor.Quantity", "Precursor.Normalised", "Intensity"])
        error_col = _resolve_table_column(columns, peptide_error_col, ["Q.Value", "QValue", "Qval", "QVal", "PEP", "FDR"])
        decoy_col = _resolve_table_column(columns, peptide_decoy_flag_col, ["Reverse", "Target/Decoy", "TargetDecoy", "Decoy"])
        required = [sample_col, seq_col, intensity_col]
        if not all(required):
            raise ValueError(
                "Unable to resolve sample, peptide sequence, and intensity columns from the peptide table."
            )
        read_cols = list(dict.fromkeys(required + [col for col in [decoy_col, error_col] if col]))
        df = _read_delimited_table_for_columns(path, read_cols)

    df = df.copy()
    df[sample_col] = df[sample_col].astype("string").str.strip()
    if is_parquet_input:
        df[sample_col] = _strip_raw_suffix_from_sample_ids(df[sample_col])
    df[seq_col] = df[seq_col].astype("string").str.strip()
    if decoy_col and decoy_col in df.columns:
        df[decoy_col] = df[decoy_col].astype("string").str.strip()
        effective_decoy_value = _infer_decoy_flag_value_from_values(df[decoy_col].dropna().unique()[:50], decoy_flag_value)
        df = df[(df[decoy_col] != effective_decoy_value) | (df[decoy_col].isna())].copy()
    df[intensity_col] = pd.to_numeric(df[intensity_col], errors="coerce")
    if error_col and error_col in df.columns:
        df[error_col] = pd.to_numeric(df[error_col], errors="coerce")

    sample_mask = df[sample_col].notna() & (df[sample_col] != "")
    total_by_sample = df.loc[sample_mask].groupby(sample_col).size().astype(int)

    valid = df[
        sample_mask
        & df[seq_col].notna()
        & (df[seq_col] != "")
        & df[intensity_col].notna()
        & (df[intensity_col] > float(intensity_min_value))
    ].copy()
    if error_col and error_col in valid.columns:
        valid = valid[valid[error_col].notna() & (valid[error_col] <= float(peptide_error_cutoff))].copy()
    quantile = float(intensity_min_quantile)
    if quantile > 0.0 and len(valid) > 0:
        thresholds = valid.groupby(sample_col)[intensity_col].transform(lambda values: values.quantile(quantile))
        valid = valid[valid[intensity_col] >= thresholds].copy()

    pair_source = valid[[sample_col, seq_col]]
    if is_parquet_input:
        valid_pairs = _drop_duplicate_pairs_with_pyarrow(pair_source, sample_col, seq_col)
    else:
        valid_pairs = pair_source.drop_duplicates()
    valid_by_sample = valid_pairs.groupby(sample_col).size().astype(int) if len(valid_pairs) else pd.Series(dtype=int)
    sample_ids = [str(value) for value in pd.unique(df.loc[sample_mask, sample_col])]
    return [
        {
            "included": bool(int(valid_by_sample.get(sample_id, 0)) > 0),
            "sample_id": sample_id,
            "analysis_unit_id": sample_id,
            "n_total_rows": int(total_by_sample.get(sample_id, 0)),
            "n_valid_peptides": int(valid_by_sample.get(sample_id, 0)),
        }
        for sample_id in sample_ids
    ]


def _read_sample_unit_mapping_table(path: str, sample_col: str, unit_col: str) -> dict[str, str]:
    import pandas as pd

    path = str(Path(path).expanduser())
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Metadata table not found: {path}")
    sep = "," if Path(path).suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep, dtype="string")
    missing = [column for column in [sample_col, unit_col] if column not in df.columns]
    if missing:
        raise ValueError(f"Metadata table is missing columns: {missing}")
    df = df[[sample_col, unit_col]].copy()
    df[sample_col] = df[sample_col].astype("string").str.strip()
    df[unit_col] = df[unit_col].astype("string").str.strip()
    df = df[df[sample_col].notna() & (df[sample_col] != "") & df[unit_col].notna() & (df[unit_col] != "")]
    df = df.drop_duplicates(subset=[sample_col], keep="first")
    return dict(zip(df[sample_col].astype(str), df[unit_col].astype(str)))


def _read_metadata_rows_by_sample(path: str, sample_col: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    import pandas as pd

    path = str(Path(path).expanduser())
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Metadata table not found: {path}")
    sep = "," if Path(path).suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep, dtype="string")
    if sample_col not in df.columns:
        raise ValueError(f"Metadata table is missing sample column: {sample_col}")
    columns = [str(column) for column in df.columns]
    df = df.copy()
    for column in columns:
        df[column] = df[column].astype("string").fillna("").str.strip()
    df = df[df[sample_col] != ""].drop_duplicates(subset=[sample_col], keep="first")
    records = {
        str(row[sample_col]): {column: str(row[column]) for column in columns}
        for row in df.to_dict(orient="records")
    }
    return columns, records


def _initial_dialog_path(current_value: str = "", fallback_dir: str = "", default_name: str = "") -> str:
    current = current_value.strip()
    if current:
        return current
    if fallback_dir:
        return str(Path(fallback_dir) / default_name) if default_name else fallback_dir
    return default_name


def _remember_dialog_directory(path_value: str) -> str:
    path = path_value.strip()
    if not path:
        return ""
    path_obj = Path(path)
    if path_obj.exists() and path_obj.is_dir():
        return str(path_obj)
    return str(path_obj.parent)


def _require_existing_file(path_value: str, field_name: str) -> None:
    if not path_value.strip():
        return
    if not os.path.isfile(path_value):
        raise ValueError(f"{field_name} does not exist or is not a file: {path_value}")
    if not os.access(path_value, os.R_OK):
        raise ValueError(f"{field_name} is not readable: {path_value}")


def _require_existing_directory(path_value: str, field_name: str) -> None:
    if not path_value.strip():
        return
    if not os.path.isdir(path_value):
        raise ValueError(f"{field_name} does not exist or is not a directory: {path_value}")
    if not os.access(path_value, os.R_OK):
        raise ValueError(f"{field_name} is not readable: {path_value}")


def _require_output_parent_directory(path_value: str, field_name: str) -> None:
    if not path_value.strip():
        return
    parent = Path(path_value).expanduser().parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError(f"Parent directory for {field_name} does not exist: {parent}")


def _require_directory_parent(path_value: str, field_name: str) -> None:
    if not path_value.strip():
        return
    parent = Path(path_value).expanduser().parent
    if str(parent) in ("", "."):
        parent = Path.cwd()
    if not parent.exists() or not parent.is_dir():
        raise ValueError(f"Parent directory for {field_name} does not exist: {parent}")


def _choose_directory(parent: QWidget, title: str, initial_path: str = "") -> str:
    return QFileDialog.getExistingDirectory(parent, title, initial_path or "")


class CollapsibleOptions(QWidget):
    def __init__(self, title: str = "More Options"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.toggle = QCheckBox(title)
        self.toggle.setObjectName("SectionToggle")
        self.body = QGroupBox()
        self.body.setProperty("subtle", True)
        self.body.setVisible(False)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self.body.setVisible)


class DigestTab(QWidget):
    def __init__(self):
        super().__init__()
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(10)
        self._last_browse_dir = ""
        self._last_auto_output_dir = ""

        scroll, layout = _create_scroll_form_host()
        outer_layout.addWidget(scroll, 1)

        required_box = QGroupBox("Required")
        required_box.setProperty("elevated", True)
        required_form = QFormLayout(required_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Digest a directory of FASTA files", "directory")
        self.mode_combo.addItem("Digest one FASTA file", "file")
        required_form.addRow("Mode", self.mode_combo)

        self.file_inputs = QGroupBox("Single File Input")
        self.file_inputs.setProperty("subtle", True)
        file_form = QFormLayout(self.file_inputs)
        file_input_row, self.input_file_edit = _make_path_row("Browse", self._browse_input_file, accept_mode="file")
        file_output_row, self.output_file_edit = _make_path_row("Browse", self._browse_output_file, accept_mode="file")
        file_form.addRow("Input FASTA file", file_input_row)
        file_form.addRow("Output TSV file", file_output_row)

        self.dir_inputs = QGroupBox("Directory Input")
        self.dir_inputs.setProperty("subtle", True)
        dir_form = QFormLayout(self.dir_inputs)
        dir_input_row, self.input_dir_edit = _make_path_row("Browse", self._browse_input_dir, accept_mode="dir")
        dir_output_row, self.output_dir_edit = _make_path_row("Browse", self._browse_output_dir, accept_mode="dir")
        self.output_dir_edit.setPlaceholderText("Auto-filled from input folder")
        dir_form.addRow("Input FASTA directory", dir_input_row)
        dir_form.addRow("Output TSV directory", dir_output_row)

        self.enzyme_combo = QComboBox()
        for enzyme_id, enzyme_name in RPG_ENZYMES:
            self.enzyme_combo.addItem(f"{enzyme_name} ({enzyme_id})", enzyme_id)
        _set_combo_to_data(self.enzyme_combo, "42")
        required_form.addRow(self.file_inputs)
        required_form.addRow(self.dir_inputs)
        required_form.addRow("Enzyme", self.enzyme_combo)
        _polish_form_layout(required_form)
        _polish_form_layout(file_form)
        _polish_form_layout(dir_form)
        layout.addWidget(required_box)

        self.more_options = CollapsibleOptions()
        options_layout = QVBoxLayout(self.more_options.body)
        self.min_length_edit = QLineEdit("7")
        self.max_length_edit = QLineEdit("30")
        self.max_miscleavages_edit = QLineEdit("2")
        self.processes_spin = _create_process_spinbox()
        self.short_header_checkbox = QCheckBox("Shorten FASTA header at first space")
        self.short_header_checkbox.setChecked(True)
        self.verbose_checkbox = QCheckBox("Verbose logging")
        self.verbose_checkbox.setChecked(True)
        self.skip_existing_checkbox = QCheckBox("Skip existing output files in directory mode")
        self.skip_existing_checkbox.setChecked(True)
        digest_grid = _create_compact_grid()
        _add_compact_field(digest_grid, 0, 0, "Minimum length", self.min_length_edit, 110)
        _add_compact_field(digest_grid, 0, 1, "Maximum length", self.max_length_edit, 110)
        _add_compact_field(digest_grid, 0, 2, "Miscleavages", self.max_miscleavages_edit, 110)
        _add_compact_field(digest_grid, 1, 0, "Processes", self.processes_spin, 110)
        options_layout.addLayout(digest_grid)

        digest_flags_grid = _create_compact_grid()
        digest_flags_grid.addWidget(self.short_header_checkbox, 0, 0)
        digest_flags_grid.addWidget(self.verbose_checkbox, 0, 1)
        digest_flags_grid.addWidget(self.skip_existing_checkbox, 1, 0, 1, 2)
        options_layout.addLayout(digest_flags_grid)
        options_layout.addWidget(
            _make_wrapped_label(
                "Single-file mode: number of worker processes used within one FASTA. "
                "Directory mode: number of FASTA files processed in parallel. "
                f"Default processes = CPU cores minus one. Maximum allowed here is {MAX_PROCESS_COUNT}."
            )
        )
        layout.addWidget(self.more_options)
        layout.addStretch(1)

        self.mode_combo.currentIndexChanged.connect(self._sync_mode_visibility)
        self.input_dir_edit.textChanged.connect(self._update_auto_output_dir_from_input_dir)
        self._sync_mode_visibility()

    def _suggest_output_dir_path(self, input_dir_path: str) -> str:
        input_path = Path(input_dir_path.strip())
        if not input_path.name:
            return ""
        return str(input_path.with_name(f"{input_path.name}_digested"))

    def _update_auto_output_dir_from_input_dir(self) -> None:
        input_dir_path = self.input_dir_edit.text().strip()
        current_output = self.output_dir_edit.text().strip()

        if not input_dir_path:
            if current_output == self._last_auto_output_dir:
                self.output_dir_edit.clear()
            self._last_auto_output_dir = ""
            return

        suggested_output = self._suggest_output_dir_path(input_dir_path)
        if not suggested_output:
            return

        if not current_output or current_output == self._last_auto_output_dir:
            self.output_dir_edit.setText(suggested_output)
        self._last_auto_output_dir = suggested_output

    def _sync_mode_visibility(self) -> None:
        mode = self.mode_combo.currentData()
        self.file_inputs.setVisible(mode == "file")
        self.dir_inputs.setVisible(mode == "directory")

    def _browse_input_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FASTA file",
            _initial_dialog_path(self.input_file_edit.text(), self._last_browse_dir),
            "FASTA files (*.fa *.faa *.fasta *.fna);;All files (*.*)",
        )
        if path:
            self.input_file_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)
            if not self.output_file_edit.text().strip():
                self.output_file_edit.setText(str(Path(path).with_suffix(".tsv")))

    def _browse_output_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output TSV file",
            _initial_dialog_path(self.output_file_edit.text(), self._last_browse_dir, "output.tsv"),
            "TSV files (*.tsv);;All files (*.*)",
        )
        if path:
            self.output_file_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _browse_input_dir(self) -> None:
        path = _choose_directory(
            self,
            "Select FASTA directory",
            _initial_dialog_path(self.input_dir_edit.text(), self._last_browse_dir),
        )
        if path:
            self.input_dir_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _browse_output_dir(self) -> None:
        path = _choose_directory(
            self,
            "Select output directory",
            _initial_dialog_path(self.output_dir_edit.text(), self._last_browse_dir),
        )
        if path:
            self.output_dir_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def build_config(self, require_required_fields: bool = True) -> DigestConfig:
        config = DigestConfig(
            input_mode=self.mode_combo.currentData(),
            input_file=self.input_file_edit.text().strip(),
            input_dir=self.input_dir_edit.text().strip(),
            output_file=self.output_file_edit.text().strip(),
            output_dir=self.output_dir_edit.text().strip(),
            enzyme_id=str(self.enzyme_combo.currentData()),
            min_length=_parse_required_int(self.min_length_edit.text(), "Minimum peptide length"),
            max_length=_parse_required_int(self.max_length_edit.text(), "Maximum peptide length"),
            max_num_miscleavages=_parse_required_int(
                self.max_miscleavages_edit.text(), "Maximum miscleavages"
            ),
            processes=int(self.processes_spin.value()),
            short_header=self.short_header_checkbox.isChecked(),
            verbose=self.verbose_checkbox.isChecked(),
            skip_existing=self.skip_existing_checkbox.isChecked(),
        )

        if require_required_fields:
            if config.min_length > config.max_length:
                raise ValueError("Minimum peptide length cannot be greater than maximum peptide length.")
            if config.input_mode == "file":
                if not config.input_file:
                    raise ValueError("Please choose an input FASTA file.")
                if not config.output_file:
                    raise ValueError("Please choose an output TSV file.")
                _require_existing_file(config.input_file, "Input FASTA file")
                _require_output_parent_directory(config.output_file, "output TSV file")
            else:
                if not config.input_dir:
                    raise ValueError("Please choose an input FASTA directory.")
                if not config.output_dir:
                    raise ValueError("Please choose an output TSV directory.")
                _require_existing_directory(config.input_dir, "Input FASTA directory")
                _require_directory_parent(config.output_dir, "output TSV directory")
        return config

    def load_config(self, config: DigestConfig) -> None:
        mode_index = 1 if config.input_mode == "file" else 0
        self.mode_combo.setCurrentIndex(mode_index)
        self.input_file_edit.setText(config.input_file)
        self.input_dir_edit.setText(config.input_dir)
        self.output_file_edit.setText(config.output_file)
        self.output_dir_edit.setText(config.output_dir)
        _set_combo_to_data(self.enzyme_combo, config.enzyme_id)
        self.min_length_edit.setText(str(config.min_length))
        self.max_length_edit.setText(str(config.max_length))
        self.max_miscleavages_edit.setText(str(config.max_num_miscleavages))
        self.processes_spin.setValue(config.processes if config.processes is not None else DEFAULT_PROCESS_COUNT)
        self.short_header_checkbox.setChecked(config.short_header)
        self.verbose_checkbox.setChecked(config.verbose)
        self.skip_existing_checkbox.setChecked(config.skip_existing)
        self._last_browse_dir = _remember_dialog_directory(
            config.input_file or config.input_dir or config.output_file or config.output_dir
        )
        self._last_auto_output_dir = self._suggest_output_dir_path(config.input_dir)
        self._sync_mode_visibility()


class ParquetExtractionDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, initial_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Import Parquet Peptide Table")
        self.resize(760, 380)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(14, 14, 14, 14)
        outer_layout.setSpacing(10)
        self._last_auto_output_tsv = ""
        self._last_browse_dir = initial_dir

        scroll, layout = _create_scroll_form_host()

        required_box = QGroupBox("Required")
        required_box.setProperty("elevated", True)
        required_form = QFormLayout(required_box)
        input_row, self.input_parquet_edit = _make_path_row("Browse", self._browse_input_parquet, accept_mode="file")
        output_row, self.output_tsv_edit = _make_path_row("Browse", self._browse_output_tsv, accept_mode="file")
        self.input_parquet_edit.setPlaceholderText("Drop a DIA-NN report.parquet file here or click Browse")
        self.output_tsv_edit.setPlaceholderText("Auto-filled as <parquet_stem>_peptides_for_metaumbra.tsv")
        required_form.addRow("Input parquet report", input_row)
        required_form.addRow("Output peptide TSV", output_row)
        _polish_form_layout(required_form)
        layout.addWidget(required_box)

        self.more_options = CollapsibleOptions()
        options_layout = QVBoxLayout(self.more_options.body)
        self.input_columns_edit = QLineEdit("Run, Stripped.Sequence, Evidence, Q.Value")
        self.output_columns_edit = QLineEdit("Run, Sequence, Evidence, Q.Value")
        self.batch_size_edit = QLineEdit("65536")
        self.force_checkbox = QCheckBox("Overwrite output TSV if it already exists")
        batch_grid = _create_compact_grid()
        _add_compact_field(batch_grid, 0, 0, "Batch size", self.batch_size_edit, 120)
        batch_grid.addWidget(self.force_checkbox, 0, 2, 1, 2)
        options_layout.addLayout(batch_grid)
        options_form = QFormLayout()
        options_form.addRow("Input columns", self.input_columns_edit)
        options_form.addRow("Output columns", self.output_columns_edit)
        _polish_form_layout(options_form)
        options_layout.addLayout(options_form)
        options_layout.addWidget(
            _make_wrapped_label(
                "Default mapping converts DIA-NN Stripped.Sequence to MetaUmbra Sequence."
            )
        )
        layout.addWidget(self.more_options)
        layout.addStretch(1)
        outer_layout.addWidget(scroll, 1)

        self.button_box = QDialogButtonBox(QDIALOG_BUTTON_OK | QDIALOG_BUTTON_CANCEL)
        self.button_box.button(QDIALOG_BUTTON_OK).setText("Extract")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        outer_layout.addWidget(self.button_box)

        self.input_parquet_edit.textChanged.connect(self._update_auto_output_tsv_from_input_parquet)

    def _suggest_output_tsv_path(self, parquet_path: str) -> str:
        return _suggest_parquet_output_tsv_path(parquet_path)

    def _update_auto_output_tsv_from_input_parquet(self) -> None:
        parquet_path = self.input_parquet_edit.text().strip()
        current_output = self.output_tsv_edit.text().strip()

        if not parquet_path:
            if current_output == self._last_auto_output_tsv:
                self.output_tsv_edit.clear()
            self._last_auto_output_tsv = ""
            return

        suggested_output = self._suggest_output_tsv_path(parquet_path)
        if not suggested_output:
            return

        if not current_output or current_output == self._last_auto_output_tsv:
            self.output_tsv_edit.setText(suggested_output)
        self._last_auto_output_tsv = suggested_output

    def _browse_input_parquet(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select parquet report",
            _initial_dialog_path(self.input_parquet_edit.text(), self._last_browse_dir),
            "Parquet files (*.parquet);;All files (*.*)",
        )
        if path:
            self.input_parquet_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _browse_output_tsv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output peptide TSV",
            _initial_dialog_path(self.output_tsv_edit.text(), self._last_browse_dir, "peptides_for_metaumbra.tsv"),
            "TSV files (*.tsv);;All files (*.*)",
        )
        if path:
            self.output_tsv_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def build_config(self, require_required_fields: bool = True) -> ParquetExtractionConfig:
        config = ParquetExtractionConfig(
            input_parquet_path=self.input_parquet_edit.text().strip(),
            output_tsv_path=self.output_tsv_edit.text().strip(),
            input_columns=_parse_text_list(self.input_columns_edit.text()),
            output_columns=_parse_text_list(self.output_columns_edit.text()),
            batch_size=_parse_required_int(self.batch_size_edit.text(), "Batch size"),
            force=self.force_checkbox.isChecked(),
        )

        if require_required_fields:
            if not config.input_parquet_path:
                raise ValueError("Please choose an input parquet report.")
            _require_existing_file(config.input_parquet_path, "Input parquet report")
            if not config.output_tsv_path:
                raise ValueError("Please choose an output peptide TSV file.")
            _require_output_parent_directory(config.output_tsv_path, "output peptide TSV file")
            if Path(config.output_tsv_path).expanduser().exists() and not config.force:
                raise ValueError("Output peptide TSV already exists. Enable overwrite to replace it.")
            if not config.input_columns:
                raise ValueError("Please provide at least one input column.")
            if len(config.input_columns) != len(config.output_columns):
                raise ValueError("Input columns and output columns must have the same number of entries.")
            if config.batch_size <= 0:
                raise ValueError("Batch size must be a positive integer.")
        return config

    def accept(self) -> None:
        try:
            self.build_config(require_required_fields=True)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return
        super().accept()


class SortableTableWidgetItem(QTableWidgetItem):
    def __init__(self, text: object, sort_value: object | None = None):
        super().__init__(str(text))
        self._sort_value = sort_value if sort_value is not None else str(text)

    def __lt__(self, other) -> bool:
        other_value = getattr(other, "_sort_value", other.text() if other is not None else "")
        try:
            return float(self._sort_value) < float(other_value)
        except (TypeError, ValueError):
            return str(self._sort_value).casefold() < str(other_value).casefold()


class SampleUnitMappingDialog(QDialog):
    def __init__(
        self,
        rows: list[dict[str, object]],
        metadata_path: str = "",
        metadata_sample_col: str = "sample_id",
        metadata_unit_col: str = "analysis_unit_id",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Configure Sample / Unit Mapping")
        self.resize(860, 560)
        self._metadata_path = metadata_path.strip()
        self._metadata_sample_col = metadata_sample_col or "sample_id"
        self._metadata_unit_col = metadata_unit_col or "analysis_unit_id"
        self._metadata_columns: list[str] = _read_table_columns(self._metadata_path) if self._metadata_path else []
        self._metadata_rows_by_sample: dict[str, dict[str, str]] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        metadata_row = QHBoxLayout()
        self.metadata_sample_col_combo = _create_editable_combo(
            self._metadata_sample_col,
            "Metadata sample ID column",
        )
        self.metadata_unit_col_combo = _create_editable_combo(
            self._metadata_unit_col,
            "Metadata column to use as analysis_unit_id",
        )
        if self._metadata_columns:
            preferred_sample = _pick_preferred_column(
                self._metadata_columns,
                [self._metadata_sample_col, "sample_id", "SampleID", "sample", "Run", "File.Name", "Raw.File"],
            )
            preferred_unit = _pick_preferred_column(
                self._metadata_columns,
                [self._metadata_unit_col, "analysis_unit_id", "unit_id", "Unit", "Group", "Condition"],
            )
            _set_editable_combo_items(
                self.metadata_sample_col_combo,
                self._metadata_columns,
                preferred_text=preferred_sample or self._metadata_sample_col,
                preferred_index=0,
            )
            _set_editable_combo_items(
                self.metadata_unit_col_combo,
                self._metadata_columns,
                preferred_text=preferred_unit or self._metadata_unit_col,
                preferred_index=1,
            )
        apply_metadata_button = QPushButton("Apply Metadata Unit")
        apply_metadata_button.clicked.connect(lambda: self._apply_metadata_unit_column(show_errors=True))
        has_metadata = bool(self._metadata_path and self._metadata_columns)
        self.metadata_sample_col_combo.setEnabled(has_metadata)
        self.metadata_unit_col_combo.setEnabled(has_metadata)
        apply_metadata_button.setEnabled(has_metadata)
        metadata_row.addWidget(QLabel("Metadata sample"))
        metadata_row.addWidget(self.metadata_sample_col_combo, 1)
        metadata_row.addWidget(QLabel("Unit column"))
        metadata_row.addWidget(self.metadata_unit_col_combo, 1)
        metadata_row.addWidget(apply_metadata_button)
        layout.addLayout(metadata_row)

        top_row = QHBoxLayout()
        self.group_name_edit = QLineEdit()
        self.group_name_edit.setPlaceholderText("analysis_unit_id for selected rows")
        set_group_button = QPushButton("Set Selected Group")
        set_group_button.clicked.connect(self._set_selected_group)
        reset_button = QPushButton("Reset To Sample IDs")
        reset_button.clicked.connect(self._reset_to_sample_ids)
        import_button = QPushButton("Import Mapping")
        import_button.clicked.connect(self._import_mapping)
        export_button = QPushButton("Export Mapping")
        export_button.clicked.connect(self._export_mapping)
        top_row.addWidget(QLabel("Group"))
        top_row.addWidget(self.group_name_edit, 1)
        top_row.addWidget(set_group_button)
        top_row.addWidget(reset_button)
        top_row.addWidget(import_button)
        top_row.addWidget(export_button)
        layout.addLayout(top_row)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Include", "sample_id", "analysis_unit_id", "n_total_rows", "n_valid_peptides"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows if hasattr(QTableWidget, "SelectionBehavior") else 1)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(QT_CUSTOM_CONTEXT_MENU)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        header = self.table.horizontalHeader()
        if header is not None:
            try:
                header.setStretchLastSection(True)
            except Exception:
                pass
            try:
                header.setSortIndicatorShown(True)
            except Exception:
                pass
        layout.addWidget(self.table, 1)

        self.summary_label = QLabel("")
        layout.addWidget(self.summary_label)

        buttons = QDialogButtonBox(QDIALOG_BUTTON_OK | QDIALOG_BUTTON_CANCEL)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_rows(rows)
        self.table.setSortingEnabled(True)
        if self._metadata_path:
            self._apply_metadata_unit_column(show_errors=True)
        self.metadata_unit_col_combo.currentIndexChanged.connect(
            lambda _index: self._apply_metadata_unit_column(show_errors=False)
        )
        self.table.itemChanged.connect(self._handle_item_changed)

    def _readonly_item(self, text: object) -> QTableWidgetItem:
        item = SortableTableWidgetItem(text)
        item.setFlags(item.flags() & ~QT_ITEM_IS_EDITABLE)
        return item

    def _load_rows(self, rows: list[dict[str, object]]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            included = bool(row.get("included", True))
            include_item = SortableTableWidgetItem("", sort_value=1 if included else 0)
            include_item.setFlags(include_item.flags() & ~QT_ITEM_IS_EDITABLE)
            include_item.setCheckState(QT_CHECKED if included else QT_UNCHECKED)
            self.table.setItem(row_index, 0, include_item)
            sample_id = str(row.get("sample_id", ""))
            self.table.setItem(row_index, 1, self._readonly_item(sample_id))
            self.table.setItem(row_index, 2, SortableTableWidgetItem(str(row.get("analysis_unit_id", sample_id))))
            total_rows = int(row.get("n_total_rows", 0) or 0)
            valid_peptides = int(row.get("n_valid_peptides", 0) or 0)
            total_item = SortableTableWidgetItem(total_rows, sort_value=total_rows)
            total_item.setFlags(total_item.flags() & ~QT_ITEM_IS_EDITABLE)
            valid_item = SortableTableWidgetItem(valid_peptides, sort_value=valid_peptides)
            valid_item.setFlags(valid_item.flags() & ~QT_ITEM_IS_EDITABLE)
            self.table.setItem(row_index, 3, total_item)
            self.table.setItem(row_index, 4, valid_item)
        self.table.setSortingEnabled(True)
        self._update_summary()

    def _selected_row_indexes(self) -> list[int]:
        rows = {index.row() for index in self.table.selectedIndexes()}
        return sorted(row for row in rows if 0 <= row < self.table.rowCount())

    def _row_sample_id(self, row: int) -> str:
        sample_item = self.table.item(row, 1)
        return sample_item.text().strip() if sample_item is not None else ""

    def _set_row_included(self, row: int, included: bool) -> None:
        include_item = self.table.item(row, 0)
        if include_item is None:
            return
        include_item.setCheckState(QT_CHECKED if included else QT_UNCHECKED)
        if hasattr(include_item, "_sort_value"):
            include_item._sort_value = 1 if included else 0

    def _set_all_included(self, included: bool) -> None:
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            self._set_row_included(row, included)
        self.table.blockSignals(False)
        self.table.setSortingEnabled(sorting_enabled)
        self._update_summary()

    def _invert_included(self) -> None:
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            include_item = self.table.item(row, 0)
            checked = include_item is not None and include_item.checkState() == QT_CHECKED
            self._set_row_included(row, not checked)
        self.table.blockSignals(False)
        self.table.setSortingEnabled(sorting_enabled)
        self._update_summary()

    def _metadata_group_values(self, column: str) -> list[str]:
        values = {
            str(record.get(column, "")).strip()
            for record in self._metadata_rows_by_sample.values()
            if str(record.get(column, "")).strip()
        }
        return sorted(values, key=lambda value: value.casefold())

    def _set_included_by_metadata_group(self, column: str, value: str, mode: str) -> None:
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            sample_id = self._row_sample_id(row)
            record = self._metadata_rows_by_sample.get(sample_id, {})
            matches = str(record.get(column, "")).strip() == value
            if mode == "only":
                self._set_row_included(row, matches)
            elif mode == "include" and matches:
                self._set_row_included(row, True)
            elif mode == "exclude" and matches:
                self._set_row_included(row, False)
        self.table.blockSignals(False)
        self.table.setSortingEnabled(sorting_enabled)
        self._update_summary()

    def _add_metadata_group_actions(self, menu: QMenu, title: str, column: str, mode: str) -> None:
        submenu = menu.addMenu(title)
        values = self._metadata_group_values(column)
        if not values:
            action = submenu.addAction("No metadata groups available")
            action.setEnabled(False)
            return
        for value in values[:40]:
            submenu.addAction(value, lambda _checked=False, selected=value: self._set_included_by_metadata_group(column, selected, mode))
        if len(values) > 40:
            action = submenu.addAction(f"Showing first 40 of {len(values)} groups")
            action.setEnabled(False)

    def _show_context_menu(self, position) -> None:
        menu = QMenu(self)
        menu.addAction("Check All Samples", lambda: self._set_all_included(True))
        menu.addAction("Uncheck All Samples", lambda: self._set_all_included(False))
        menu.addAction("Invert Checked Samples", self._invert_included)
        unit_col = self.metadata_unit_col_combo.currentText().strip()
        if self._metadata_rows_by_sample and unit_col:
            menu.addSeparator()
            self._add_metadata_group_actions(menu, f"Check {unit_col} Group", unit_col, "include")
            self._add_metadata_group_actions(menu, f"Check Only {unit_col} Group", unit_col, "only")
            self._add_metadata_group_actions(menu, f"Uncheck {unit_col} Group", unit_col, "exclude")
        exec_method = getattr(menu, "exec", None) or getattr(menu, "exec_", None)
        exec_method(self.table.viewport().mapToGlobal(position))

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0 and hasattr(item, "_sort_value"):
            item._sort_value = 1 if item.checkState() == QT_CHECKED else 0
        self._update_summary()

    def _set_selected_group(self) -> None:
        group_name = self.group_name_edit.text().strip()
        if not group_name:
            QMessageBox.warning(self, "Missing Group", "Enter an analysis_unit_id first.")
            return
        rows = self._selected_row_indexes()
        if not rows:
            QMessageBox.warning(self, "No Rows Selected", "Select one or more sample rows first.")
            return
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        for row in rows:
            item = self.table.item(row, 2)
            if item is not None:
                item.setText(group_name)
        self.table.setSortingEnabled(sorting_enabled)
        self._update_summary()

    def _reset_to_sample_ids(self) -> None:
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        for row in range(self.table.rowCount()):
            sample_item = self.table.item(row, 1)
            unit_item = self.table.item(row, 2)
            if sample_item is not None and unit_item is not None:
                unit_item.setText(sample_item.text())
        self.table.setSortingEnabled(sorting_enabled)
        self._update_summary()

    def _apply_mapping(self, mapping: dict[str, str]) -> None:
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        for row in range(self.table.rowCount()):
            sample_item = self.table.item(row, 1)
            unit_item = self.table.item(row, 2)
            if sample_item is None or unit_item is None:
                continue
            sample_id = sample_item.text().strip()
            if sample_id in mapping:
                unit_item.setText(mapping[sample_id])
        self.table.setSortingEnabled(sorting_enabled)
        self._update_summary()

    def _apply_metadata_unit_column(self, show_errors: bool = False) -> None:
        if not self._metadata_path:
            return
        sample_col = self.metadata_sample_col_combo.currentText().strip() or self._metadata_sample_col
        unit_col = self.metadata_unit_col_combo.currentText().strip() or self._metadata_unit_col
        if not sample_col or not unit_col:
            return
        try:
            columns, rows_by_sample = _read_metadata_rows_by_sample(self._metadata_path, sample_col)
            if unit_col not in columns:
                raise ValueError(f"Metadata table is missing unit column: {unit_col}")
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(self, "Metadata Import Failed", str(exc))
            return
        self._metadata_sample_col = sample_col
        self._metadata_unit_col = unit_col
        self._metadata_columns = columns
        self._metadata_rows_by_sample = rows_by_sample
        mapping = {
            sample_id: str(record.get(unit_col, "")).strip()
            for sample_id, record in rows_by_sample.items()
            if str(record.get(unit_col, "")).strip()
        }
        self._apply_mapping(mapping)

    def _import_mapping(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import sample mapping",
            "",
            "TSV/CSV files (*.tsv *.txt *.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            self._apply_mapping(
                _read_sample_unit_mapping_table(
                    path,
                    self._metadata_sample_col,
                    self._metadata_unit_col,
                )
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))

    def _export_mapping(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export sample mapping",
            "sample_unit_mapping.tsv",
            "TSV files (*.tsv);;All files (*.*)",
        )
        if not path:
            return
        try:
            rows = self.mapping_rows()
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "analysis_unit_id", "n_valid_peptides", "n_total_rows", "included"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _update_summary(self) -> None:
        rows = self.mapping_rows()
        included = [row for row in rows if row["included"]]
        units = {row["analysis_unit_id"] for row in included if row["analysis_unit_id"]}
        self.summary_label.setText(
            f"{len(included)} included sample(s), {len(units)} analysis unit(s)."
        )

    def mapping_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in range(self.table.rowCount()):
            include_item = self.table.item(row, 0)
            sample_item = self.table.item(row, 1)
            unit_item = self.table.item(row, 2)
            total_item = self.table.item(row, 3)
            valid_item = self.table.item(row, 4)
            sample_id = sample_item.text().strip() if sample_item is not None else ""
            unit_id = unit_item.text().strip() if unit_item is not None else ""
            if not unit_id:
                unit_id = sample_id
            rows.append(
                {
                    "sample_id": sample_id,
                    "analysis_unit_id": unit_id,
                    "n_total_rows": int(total_item.text()) if total_item is not None and total_item.text().isdigit() else 0,
                    "n_valid_peptides": int(valid_item.text()) if valid_item is not None and valid_item.text().isdigit() else 0,
                    "included": bool(include_item is None or include_item.checkState() == QT_CHECKED),
                }
            )
        return rows

    def accept(self) -> None:
        for row in self.mapping_rows():
            if row["included"] and (not row["sample_id"] or not row["analysis_unit_id"]):
                QMessageBox.critical(self, "Invalid Mapping", "Included rows require sample_id and analysis_unit_id.")
                return
        super().accept()


class ScoringTab(QWidget):
    def __init__(self):
        super().__init__()
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(10)
        self._last_auto_output_tsv = ""
        self._last_browse_dir = ""
        self._sample_unit_mapping_rows: list[dict[str, object]] = []
        self._sample_unit_mapping_source_path = ""
        self._sample_unit_preview_cache_key: tuple[object, ...] | None = None
        self._sample_unit_preview_cache_rows: list[dict[str, object]] = []

        scroll, layout = _create_scroll_form_host()
        outer_layout.addWidget(scroll, 1)

        required_box = QGroupBox("Required")
        required_box.setProperty("elevated", True)
        required_form = QFormLayout(required_box)
        peptide_row, self.peptide_table_edit = _make_path_row("Browse", self._browse_peptide_table, accept_mode="file")
        self.peptide_table_edit.setPlaceholderText("Drop a peptide TSV or DIA-NN report.parquet")
        self.import_parquet_button = QPushButton("Import Parquet...")
        self.import_parquet_button.setToolTip("Extract a MetaUmbra peptide TSV from a DIA-NN parquet report.")
        peptide_row.layout().addWidget(self.import_parquet_button)
        required_form.addRow("Observed peptide table", peptide_row)
        lineage_row, self.genome_lineage_table_edit = _make_path_row("Browse", self._browse_genome_lineage_table, accept_mode="file")
        required_form.addRow("Genome-Lineage table (optional)", lineage_row)
        output_row, self.output_tsv_edit = _make_path_row("Browse", self._browse_output_tsv, accept_mode="file")
        required_form.addRow("Output result TSV", output_row)

        genome_box = QGroupBox("Genome Digest Directories")
        genome_layout = QVBoxLayout(genome_box)
        self.genome_dir_list = DirectoryDropListWidget()
        self.genome_dir_list.setMinimumHeight(84)
        self.genome_dir_list.setMaximumHeight(96)
        button_row = QHBoxLayout()
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_genome_dir)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._remove_genome_dir)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_genome_dirs)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)
        genome_layout.addWidget(self.genome_dir_list)
        genome_layout.addLayout(button_row)
        required_form.addRow(genome_box)
        _polish_form_layout(required_form)
        layout.addWidget(required_box)

        mapping_box = QGroupBox("Column Mapping")
        mapping_box.setProperty("elevated", True)
        mapping_layout = QVBoxLayout(mapping_box)
        self.peptide_seq_col_edit = _create_editable_combo("Sequence", "Choose or type the peptide sequence column")
        self.peptide_score_col_edit = _create_editable_combo("Evidence", "Choose or type the peptide score column")
        self.peptide_seq_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.peptide_score_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        mapping_grid = _create_compact_grid()
        _add_compact_field(mapping_grid, 0, 0, "Sequence column", self.peptide_seq_col_edit, None)
        self.peptide_score_col_label = _add_compact_field(
            mapping_grid, 0, 1, "Score column", self.peptide_score_col_edit, None
        )
        mapping_layout.addLayout(mapping_grid)
        self.lineage_columns_box = QGroupBox("Genome-Lineage Columns")
        self.lineage_columns_box.setProperty("subtle", True)
        lineage_columns_grid = _create_compact_grid()
        self.genome_lineage_genome_id_col_edit = _create_editable_combo(
            placeholder="Required if table is provided, e.g. Genome_id"
        )
        self.genome_lineage_lineage_col_edit = _create_editable_combo(
            placeholder="Required if table is provided, e.g. Lineage"
        )
        self.genome_lineage_genome_id_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.genome_lineage_lineage_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        lineage_columns_grid.addWidget(QLabel("Lineage genome ID column"), 0, 0)
        lineage_columns_grid.addWidget(self.genome_lineage_genome_id_col_edit, 0, 1)
        lineage_columns_grid.addWidget(QLabel("Lineage column"), 0, 2)
        lineage_columns_grid.addWidget(self.genome_lineage_lineage_col_edit, 0, 3)
        self.lineage_columns_box.setVisible(False)
        lineage_columns_box_layout = QVBoxLayout(self.lineage_columns_box)
        lineage_columns_box_layout.addLayout(lineage_columns_grid)
        mapping_layout.addWidget(self.lineage_columns_box)
        layout.addWidget(mapping_box)

        self.more_options = CollapsibleOptions()
        options_layout = QVBoxLayout(self.more_options.body)

        columns_box = QGroupBox("Peptide Row Filtering")
        columns_box.setProperty("subtle", True)
        columns_grid = _create_compact_grid()
        self.peptide_error_col_edit = _create_editable_combo("Q.Value")
        self.peptide_error_cutoff_edit = QLineEdit("0.05")
        self.peptide_decoy_flag_col_edit = _create_editable_combo("Reverse")
        self.sample_id_col_edit = _create_editable_combo("Run")
        self.intensity_col_edit = _create_editable_combo("Precursor.Quantity")
        self.decoy_flag_value_edit = _create_editable_combo("+", "Decoy flag value")
        self.decoy_flag_value_edit.addItems(["+", "decoy", "1", "True", "FALSE", "T", "F"])
        self.decoy_flag_value_edit.setEditText("+")
        self.peptide_error_cutoff_edit.setToolTip("Filters input peptide rows by the selected error/FDR column.")
        self.peptide_error_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.peptide_decoy_flag_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.sample_id_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.intensity_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        _add_compact_field(columns_grid, 0, 0, "Error column", self.peptide_error_col_edit, 170)
        _add_compact_field(columns_grid, 0, 1, "Peptide error cutoff", self.peptide_error_cutoff_edit, 90)
        self.peptide_decoy_flag_col_label = _add_compact_field(
            columns_grid, 0, 2, "Decoy flag column", self.peptide_decoy_flag_col_edit, 170
        )
        self.decoy_flag_value_label = _add_compact_field(
            columns_grid, 0, 3, "Decoy flag value", self.decoy_flag_value_edit, 90
        )
        columns_layout = QVBoxLayout(columns_box)
        columns_layout.addLayout(columns_grid)
        layout.addWidget(columns_box)

        self.unit_aware_checkbox = QCheckBox("Enable unit-aware multi-sample scoring")
        self.unit_aware_checkbox.setToolTip(
            "Interpret the observed peptide table as long-format sample evidence and score genome presence per analysis unit."
        )
        unit_aware_row = QWidget()
        unit_aware_row.setObjectName("InlineOptionRow")
        unit_aware_row_layout = QHBoxLayout(unit_aware_row)
        unit_aware_row_layout.setContentsMargins(10, 6, 10, 6)
        unit_aware_row_layout.addWidget(self.unit_aware_checkbox)
        unit_aware_row_layout.addStretch(1)
        layout.addWidget(unit_aware_row)

        self.unit_box = QGroupBox("Unit-aware Sample Definition")
        self.unit_box.setProperty("subtle", True)
        unit_layout = QVBoxLayout(self.unit_box)
        self.export_unit_derived_tables_checkbox = QCheckBox("Export derived unit-aware tables")
        self.export_unit_derived_tables_checkbox.setToolTip(
            "Write optional unit-aware call-count, significant-call, genome-union, and matrix tables under the artifacts folder."
        )
        self.export_unit_derived_tables_checkbox.setChecked(True)
        self.intensity_min_value_spin = QSpinBox()
        # Limit to signed 32-bit int max to avoid libshiboken overflow
        self.intensity_min_value_spin.setRange(0, 2147483647)
        self.intensity_min_value_spin.setSingleStep(10)
        self.intensity_min_value_spin.setValue(0)
        self.intensity_min_value_spin.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.intensity_min_value_spin.setToolTip(
            "Minimum intensity required for sample-level peptide presence."
        )
        self.intensity_min_quantile_spin = QDoubleSpinBox()
        # Accept percentage input 0-100 for user convenience; stored/used as fraction (0-1)
        self.intensity_min_quantile_spin.setRange(0.0, 100.0)
        self.intensity_min_quantile_spin.setDecimals(2)
        self.intensity_min_quantile_spin.setSingleStep(1.0)
        self.intensity_min_quantile_spin.setValue(0.0)
        self.intensity_min_quantile_spin.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.intensity_min_quantile_spin.setToolTip(
            "Percentage of lowest-intensity rows to remove within each sample. Use 5 for the lowest 5%."
        )
        metadata_row, self.metadata_table_edit = _make_path_row("Browse", self._browse_metadata_table, accept_mode="file")
        self.metadata_sample_id_col_edit = _create_editable_combo("sample_id", "Choose or type the metadata sample column")
        self.metadata_analysis_unit_col_edit = _create_editable_combo(
            "analysis_unit_id",
            "Choose or type the metadata analysis-unit column",
        )
        self.metadata_sample_id_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.metadata_analysis_unit_col_edit.setSizePolicy(QSIZE_EXPANDING, QSIZE_PREFERRED)
        self.configure_sample_mapping_button = QPushButton("Configure Sample / Unit Mapping")
        self.configure_sample_mapping_button.clicked.connect(self._configure_sample_unit_mapping)
        self.sample_mapping_status_label = QLabel("No custom sample mapping configured.")

        self.sample_filter_box = QGroupBox("Sample Columns And Intensity Filters")
        self.sample_filter_box.setProperty("subtle", True)
        sample_filter_grid = _create_compact_grid()
        _add_compact_field(sample_filter_grid, 0, 0, "Sample ID column", self.sample_id_col_edit, None)
        _add_compact_field(sample_filter_grid, 0, 1, "Intensity column", self.intensity_col_edit, None)
        _add_compact_field(sample_filter_grid, 1, 0, "Minimum intensity", self.intensity_min_value_spin, None)
        _add_compact_field(sample_filter_grid, 1, 1, "Drop lowest percent (%)", self.intensity_min_quantile_spin, None)
        sample_filter_layout = QVBoxLayout(self.sample_filter_box)
        sample_filter_layout.addLayout(sample_filter_grid)
        unit_layout.addWidget(self.sample_filter_box)

        self.metadata_box = QGroupBox("Sample / Unit Mapping")
        self.metadata_box.setProperty("subtle", True)
        metadata_layout = QVBoxLayout(self.metadata_box)
        metadata_grid = _create_compact_grid()
        metadata_table_label = QLabel("Metadata table")
        metadata_table_label.setAlignment(QT_ALIGN_LEFT | QT_ALIGN_VCENTER)
        metadata_grid.addWidget(metadata_table_label, 0, 0)
        metadata_grid.addWidget(metadata_row, 0, 1, 1, 5)
        _add_compact_field(metadata_grid, 1, 0, "Metadata sample ID column", self.metadata_sample_id_col_edit, None)
        _add_compact_field(
            metadata_grid,
            1,
            1,
            "Metadata analysis unit column",
            self.metadata_analysis_unit_col_edit,
            None,
        )
        metadata_layout.addLayout(metadata_grid)
        unit_layout.addWidget(self.metadata_box)

        unit_output_grid = _create_compact_grid()
        unit_output_grid.addWidget(self.configure_sample_mapping_button, 0, 0, 1, 2)
        unit_output_grid.addWidget(self.sample_mapping_status_label, 0, 2, 1, 2)
        unit_layout.addLayout(unit_output_grid)
        layout.addWidget(self.unit_box)

        unique_box = QGroupBox("Unique Evidence Settings")
        unique_box.setProperty("subtle", True)
        unique_box.setToolTip(
            "Unique p-value strength is controlled by unique p-value mode. Alpha controls apply only to alpha-upper-bound."
        )
        unique_layout = QVBoxLayout(unique_box)
        self.unique_pvalue_mode_combo = QComboBox()
        self.unique_pvalue_mode_combo.addItem("Empirical background", "empirical-background")
        self.unique_pvalue_mode_combo.addItem("Alpha upper bound", "alpha-upper-bound")
        self.unique_pvalue_mode_combo.addItem("Hypergeometric opportunity", "hypergeometric-opportunity")
        self.unique_pvalue_mode_combo.setMinimumWidth(260)
        _set_compact_control_width(self.unique_pvalue_mode_combo, 260)
        self.unique_peptide_error_source_combo = QComboBox()
        self.unique_peptide_error_source_combo.addItem("Global alpha", "global-alpha")
        self.unique_peptide_error_source_combo.addItem("Peptide error column", "peptide-error-column")
        _set_compact_control_width(self.unique_peptide_error_source_combo, 190)
        self.single_peptide_error_rate_upper_bound_edit = QLineEdit("0.05")
        self.unique_count_power_spin = QDoubleSpinBox()
        self.unique_count_power_spin.setRange(0.01, 1.0)
        self.unique_count_power_spin.setSingleStep(0.05)
        self.unique_count_power_spin.setDecimals(2)
        self.unique_count_power_spin.setValue(1.0)
        self.theoretical_opportunity_processes_spin = _create_optional_process_spinbox()
        self.theoretical_opportunity_processes_spin.setToolTip(
            "Optional worker count for theoretical opportunity sharding.\n"
            "Use 0 to reuse the main Processes setting."
        )
        self.rebuild_theoretical_opportunity_cache_checkbox = QCheckBox("Rebuild theoretical opportunity cache")
        self.single_peptide_error_rate_upper_bound_edit.setToolTip(
            "Global alpha used by alpha-upper-bound mode when unique peptide error source is Global alpha."
        )
        self.unique_pvalue_mode_combo.setToolTip(
            "Empirical background estimates the sample-specific weak-genome unique peptide background and tests whether each genome's unique peptide count exceeds that background."
        )
        self.unique_peptide_error_source_combo.setToolTip(
            "Choose epsilon_i for alpha-upper-bound mode."
        )
        self.unique_count_power_spin.setToolTip(
            "Power exponent for alpha-upper-bound mode: U_eff = U_raw^power. Set to 1.0 for raw unique count."
        )
        theoretical_cache_row, self.theoretical_opportunity_cache_edit = _make_path_row(
            "Browse", self._browse_theoretical_opportunity_cache, accept_mode="file"
        )
        unique_grid = _create_compact_grid()
        _add_compact_field(unique_grid, 0, 0, "Unique p-value mode", self.unique_pvalue_mode_combo, 260)
        self.unique_peptide_error_source_label = _add_compact_field(
            unique_grid,
            0,
            1,
            "Unique peptide error source",
            self.unique_peptide_error_source_combo,
            190,
        )
        self.unique_alpha_label = QLabel("Unique evidence alpha")
        self.unique_alpha_label.setAlignment(QT_ALIGN_LEFT | QT_ALIGN_VCENTER)
        _set_compact_control_width(self.single_peptide_error_rate_upper_bound_edit, 130)
        unique_grid.addWidget(self.unique_alpha_label, 0, 4)
        unique_grid.addWidget(self.single_peptide_error_rate_upper_bound_edit, 0, 5)
        self.unique_count_power_label = QLabel("Unique count power")
        self.unique_count_power_label.setAlignment(QT_ALIGN_LEFT | QT_ALIGN_VCENTER)
        _set_compact_control_width(self.unique_count_power_spin, 110)
        unique_grid.addWidget(self.unique_count_power_label, 1, 0)
        unique_grid.addWidget(self.unique_count_power_spin, 1, 1)
        self.theoretical_opportunity_processes_label = QLabel("Opportunity processes")
        self.theoretical_opportunity_processes_label.setAlignment(QT_ALIGN_LEFT | QT_ALIGN_VCENTER)
        _set_compact_control_width(self.theoretical_opportunity_processes_spin, 160)
        unique_grid.addWidget(self.theoretical_opportunity_processes_label, 0, 4)
        unique_grid.addWidget(self.theoretical_opportunity_processes_spin, 0, 5)
        unique_layout.addLayout(unique_grid)
        unique_form = QFormLayout()
        self.theoretical_opportunity_cache_label = QLabel("Theoretical opportunity cache")
        unique_form.addRow(self.theoretical_opportunity_cache_label, theoretical_cache_row)
        unique_form.addRow("", self.rebuild_theoretical_opportunity_cache_checkbox)
        _polish_form_layout(unique_form)
        unique_layout.addLayout(unique_form)
        options_layout.addWidget(unique_box)

        runtime_box = QGroupBox("Knockoff Runtime")
        runtime_box.setProperty("subtle", True)
        runtime_layout = QVBoxLayout(runtime_box)
        self.processes_spin = _create_process_spinbox()
        self.knockoff_mc_iterations_edit = QLineEdit("500")
        self.knockoff_stage2_mc_iterations_edit = QLineEdit("2000")
        self.knockoff_stage2_ranges_edit = QLineEdit("0.01-0.05")
        self.knockoff_random_seed_edit = QLineEdit("1")
        self.knockoff_top_n_targets_spin = QSpinBox()
        self.knockoff_top_n_targets_spin.setRange(0, 1000000)
        self.knockoff_top_n_targets_spin.setValue(0)
        self.knockoff_top_n_targets_spin.setToolTip(
            "Only run knockoff inference for the top N genomes ranked by evidence.\n"
            "Use 0 to evaluate all candidate genomes.\n"
            "This is mainly a speed optimization for very large runs."
        )
        runtime_grid = _create_compact_grid()
        _add_compact_field(runtime_grid, 0, 0, "Processes", self.processes_spin, 110)
        _add_compact_field(runtime_grid, 0, 1, "Top genomes", self.knockoff_top_n_targets_spin, 110)
        _add_compact_field(runtime_grid, 1, 0, "Random seed", self.knockoff_random_seed_edit, 110)
        _add_compact_field(runtime_grid, 1, 1, "Knockoff MC", self.knockoff_mc_iterations_edit, 110)
        _add_compact_field(runtime_grid, 1, 2, "Stage 2 MC", self.knockoff_stage2_mc_iterations_edit, 110)
        runtime_layout.addLayout(runtime_grid)
        runtime_form = QFormLayout()
        runtime_form.addRow("Stage 2 p ranges", self.knockoff_stage2_ranges_edit)
        runtime_form.addRow(
            _make_wrapped_label(
                f"Default processes = CPU cores minus one. Maximum allowed here is {MAX_PROCESS_COUNT}."
            )
        )
        runtime_form.addRow(
            _make_wrapped_label(
                "If the knockoff limit is 0, all candidate genomes are evaluated. Set a positive value only to speed up very large runs."
            )
        )
        _polish_form_layout(runtime_form)
        runtime_layout.addLayout(runtime_form)
        options_layout.addWidget(runtime_box)

        self.unique_pvalue_mode_combo.currentIndexChanged.connect(self._sync_unique_mode_visibility)
        self.unique_peptide_error_source_combo.currentIndexChanged.connect(self._sync_unique_mode_visibility)

        cache_output_box = QGroupBox("Cache And Output Artifacts")
        cache_output_box.setProperty("subtle", True)
        cache_output_layout = QVBoxLayout(cache_output_box)
        cache_row, self.cache_path_edit = _make_path_row("Browse", self._browse_cache_path, accept_mode="file")
        self.export_peptide_contrib_topn_spin = QSpinBox()
        self.export_peptide_contrib_topn_spin.setRange(0, 1000000)
        self.export_peptide_contrib_topn_spin.setValue(0)
        self.save_cache_checkbox = QCheckBox("Save matched-peptide cache")
        self.save_cache_checkbox.setChecked(True)
        self.use_cache_if_exists_checkbox = QCheckBox("Reuse cache if it already exists")
        self.compute_coverage_checkbox = QCheckBox("Compute coverage columns")
        self.compute_coverage_checkbox.setChecked(True)
        self.export_temp_checkbox = QCheckBox("Export temp artifacts")
        self.export_temp_checkbox.setChecked(True)
        self.return_full_table_checkbox = QCheckBox("Return full internal table")
        cache_form = QFormLayout()
        cache_form.addRow("Matched peptide cache", cache_row)
        _polish_form_layout(cache_form)
        cache_output_layout.addLayout(cache_form)
        artifact_grid = _create_compact_grid()
        _add_compact_field(artifact_grid, 0, 0, "Export contrib top-N", self.export_peptide_contrib_topn_spin, 120)
        artifact_grid.addWidget(self.save_cache_checkbox, 1, 0, 1, 2)
        artifact_grid.addWidget(self.use_cache_if_exists_checkbox, 1, 2, 1, 2)
        artifact_grid.addWidget(self.compute_coverage_checkbox, 2, 0, 1, 2)
        artifact_grid.addWidget(self.export_temp_checkbox, 2, 2, 1, 2)
        artifact_grid.addWidget(self.return_full_table_checkbox, 3, 0, 1, 2)
        artifact_grid.addWidget(self.export_unit_derived_tables_checkbox, 3, 2, 1, 2)
        cache_output_layout.addLayout(artifact_grid)
        options_layout.addWidget(cache_output_box)

        genome_filter_box = QGroupBox("Genome Selection Filters")
        genome_filter_box.setProperty("subtle", True)
        genome_filter_layout = QHBoxLayout(genome_filter_box)

        exclude_filter_panel = QWidget()
        exclude_filter_layout = QVBoxLayout(exclude_filter_panel)
        exclude_filter_layout.setContentsMargins(0, 0, 0, 0)
        exclude_filter_layout.setSpacing(8)
        exclude_filter_header = QHBoxLayout()
        exclude_filter_header.setContentsMargins(0, 0, 0, 0)
        exclude_filter_label = QLabel("Excluded genomes")
        self.load_last_excluded_genomes_button = QPushButton("Load Last")
        self.clear_excluded_genomes_button = QPushButton("Clear All")
        exclude_filter_header.addWidget(exclude_filter_label)
        exclude_filter_header.addStretch(1)
        exclude_filter_header.addWidget(self.load_last_excluded_genomes_button)
        exclude_filter_header.addWidget(self.clear_excluded_genomes_button)
        self.exclude_text = FileContentTextEdit()
        self.exclude_text.setPlaceholderText(
            "Excluded genome IDs.\n"
            "One genome ID per line, or comma-separated.\n"
            "Genome IDs should match digest TSV filenames without the .tsv suffix."
        )

        selected_filter_panel = QWidget()
        selected_filter_layout = QVBoxLayout(selected_filter_panel)
        selected_filter_layout.setContentsMargins(0, 0, 0, 0)
        selected_filter_layout.setSpacing(8)
        selected_filter_header = QHBoxLayout()
        selected_filter_header.setContentsMargins(0, 0, 0, 0)
        selected_filter_label = QLabel("Only run these genome IDs")
        self.load_last_selected_genomes_button = QPushButton("Load Last")
        self.clear_selected_genomes_button = QPushButton("Clear All")
        selected_filter_header.addWidget(selected_filter_label)
        selected_filter_header.addStretch(1)
        selected_filter_header.addWidget(self.load_last_selected_genomes_button)
        selected_filter_header.addWidget(self.clear_selected_genomes_button)
        self.selected_genomes_text = FileContentTextEdit()
        self.selected_genomes_text.setPlaceholderText(
            "Only run these genome IDs.\n"
            "Leave empty to run all candidates.\n"
            "One genome ID per line, or comma-separated."
        )
        exclude_filter_layout.addLayout(exclude_filter_header)
        exclude_filter_layout.addWidget(self.exclude_text, 1)
        selected_filter_layout.addLayout(selected_filter_header)
        selected_filter_layout.addWidget(self.selected_genomes_text, 1)
        genome_filter_layout.addWidget(exclude_filter_panel, 1)
        genome_filter_layout.addWidget(selected_filter_panel, 1)
        options_layout.addWidget(genome_filter_box)
        layout.addWidget(self.more_options)
        layout.addStretch(1)
        self.peptide_table_edit.textChanged.connect(self._update_auto_output_tsv_from_peptide_table)
        self.peptide_table_edit.textChanged.connect(self._update_peptide_table_column_options)
        self.genome_lineage_table_edit.textChanged.connect(self._update_genome_lineage_column_options)
        self.genome_lineage_table_edit.textChanged.connect(self._sync_genome_lineage_column_visibility)
        self.metadata_table_edit.textChanged.connect(self._update_metadata_table_column_options)
        self.unit_aware_checkbox.toggled.connect(self._sync_unit_aware_visibility)
        self.load_last_excluded_genomes_button.clicked.connect(self._load_last_excluded_genomes)
        self.clear_excluded_genomes_button.clicked.connect(self.exclude_text.clear)
        self.load_last_selected_genomes_button.clicked.connect(self._load_last_selected_genomes)
        self.clear_selected_genomes_button.clicked.connect(self.selected_genomes_text.clear)
        self._sync_unit_aware_visibility()

    def _suggest_output_tsv_path(self, peptide_table_path: str) -> str:
        peptide_path = Path(peptide_table_path.strip())
        if not peptide_path.name:
            return ""
        return str(peptide_path.with_name(f"{peptide_path.stem}_MetaUmbra_Genome_Presence.tsv"))

    def _update_auto_output_tsv_from_peptide_table(self) -> None:
        peptide_table_path = self.peptide_table_edit.text().strip()
        current_output = self.output_tsv_edit.text().strip()

        if not peptide_table_path:
            if current_output == self._last_auto_output_tsv:
                self.output_tsv_edit.clear()
            self._last_auto_output_tsv = ""
            return

        suggested_output = self._suggest_output_tsv_path(peptide_table_path)
        if not suggested_output:
            return

        if not current_output or current_output == self._last_auto_output_tsv:
            self.output_tsv_edit.setText(suggested_output)
        self._last_auto_output_tsv = suggested_output

    def _update_peptide_table_column_options(self) -> None:
        path_value = self.peptide_table_edit.text()
        if _is_parquet_path(path_value):
            columns = _read_parquet_schema_columns(path_value)
            if not columns:
                _clear_editable_combo_items(self.peptide_seq_col_edit, "Sequence")
                _clear_editable_combo_items(self.peptide_score_col_edit, "Evidence")
                _clear_editable_combo_items(self.peptide_error_col_edit, "Q.Value")
                _clear_editable_combo_items(self.peptide_decoy_flag_col_edit, "Reverse")
                _clear_editable_combo_items(self.sample_id_col_edit, "Run")
                _clear_editable_combo_items(self.intensity_col_edit, "Precursor.Quantity")
                return
            preferred_seq = _pick_preferred_column(
                columns,
                ["Stripped.Sequence", "StrippedSequence", "Sequence", "Peptide.Sequence", "PeptideSequence"],
            )
            preferred_sample = _pick_preferred_column(
                columns,
                ["Run", "File.Name", "Raw.File", "Sample", "Sample.Name"],
            )
            preferred_intensity = _pick_preferred_column(
                columns,
                ["Precursor.Quantity", "Precursor.Normalised", "Intensity"],
            )
            preferred_score = _pick_preferred_column(columns, ["Evidence", "Score", "CScore"])
            preferred_error = _pick_preferred_column(
                columns,
                ["Q.Value", "QValue", "Qval", "QVal", "FDR", "PEP"],
            )
            preferred_decoy = _pick_preferred_column(
                columns,
                ["Reverse", "Target/Decoy", "TargetDecoy", "Decoy"],
            )
            _set_editable_combo_items(
                self.peptide_seq_col_edit,
                columns,
                preferred_text=preferred_seq or "Sequence",
            )
            _set_editable_combo_items(
                self.peptide_score_col_edit,
                columns,
                preferred_text=preferred_score or "Evidence",
            )
            _set_editable_combo_items(
                self.peptide_error_col_edit,
                columns,
                preferred_text=preferred_error or "Q.Value",
            )
            _set_editable_combo_items(
                self.peptide_decoy_flag_col_edit,
                columns,
                preferred_text=preferred_decoy or "Reverse",
            )
            _set_editable_combo_items(
                self.sample_id_col_edit,
                columns,
                preferred_text=preferred_sample or "Run",
            )
            _set_editable_combo_items(
                self.intensity_col_edit,
                columns,
                preferred_text=preferred_intensity or "Precursor.Quantity",
            )
            return
        columns = _read_table_columns(path_value)
        if not columns:
            _clear_editable_combo_items(self.peptide_seq_col_edit, "Sequence")
            _clear_editable_combo_items(self.peptide_score_col_edit, "Evidence")
            _clear_editable_combo_items(self.peptide_error_col_edit, "Q.Value")
            _clear_editable_combo_items(self.peptide_decoy_flag_col_edit, "Reverse")
            _clear_editable_combo_items(self.sample_id_col_edit, "Run")
            _clear_editable_combo_items(self.intensity_col_edit, "Precursor.Quantity")
            return
        _set_editable_combo_items(self.peptide_seq_col_edit, columns, preferred_text="Sequence")
        preferred_sample = _pick_preferred_column(columns, ["Run", "File.Name", "Raw.File", "Sample", "Sample.Name"])
        preferred_intensity = _pick_preferred_column(columns, ["Precursor.Quantity", "Precursor.Normalised", "Intensity"])
        preferred_score = _pick_preferred_column(columns, ["Evidence", "score", "Score"]) or "Evidence"
        preferred_error = _pick_preferred_column(columns, ["Q.Value", "QValue", "Qval", "QVal", "FDR", "PEP"])
        preferred_decoy = _pick_preferred_column(
            columns,
            ["Reverse", "Target/Decoy", "TargetDecoy", "Decoy"],
        )
        _set_editable_combo_items(self.peptide_score_col_edit, columns, preferred_text=preferred_score)
        _set_editable_combo_items(
            self.peptide_error_col_edit,
            columns,
            preferred_text=preferred_error or "Q.Value",
        )
        _set_editable_combo_items(
            self.peptide_decoy_flag_col_edit,
            columns,
            preferred_text=preferred_decoy or "Reverse",
        )
        _set_editable_combo_items(self.sample_id_col_edit, columns, preferred_text=preferred_sample or "Run")
        _set_editable_combo_items(
            self.intensity_col_edit,
            columns,
            preferred_text=preferred_intensity or "Precursor.Quantity",
        )

    def _update_genome_lineage_column_options(self) -> None:
        columns = _read_table_columns(self.genome_lineage_table_edit.text())
        if not columns:
            _clear_editable_combo_items(self.genome_lineage_genome_id_col_edit)
            _clear_editable_combo_items(self.genome_lineage_lineage_col_edit)
            return
        _set_editable_combo_items(
            self.genome_lineage_genome_id_col_edit,
            columns,
            preferred_text="Genome_id",
        )
        _set_editable_combo_items(
            self.genome_lineage_lineage_col_edit,
            columns,
            preferred_text="Lineage",
            preferred_index=1,
        )

    def _update_metadata_table_column_options(self) -> None:
        columns = _read_table_columns(self.metadata_table_edit.text())
        if not columns:
            _clear_editable_combo_items(self.metadata_sample_id_col_edit, "sample_id")
            _clear_editable_combo_items(self.metadata_analysis_unit_col_edit, "analysis_unit_id")
            return
        preferred_sample = _pick_preferred_column(
            columns,
            ["sample_id", "SampleID", "sample", "Sample", "Run", "File.Name", "Raw.File", "Sample.Name"],
        )
        preferred_unit = _pick_preferred_column(
            columns,
            [
                "analysis_unit_id",
                "AnalysisUnitID",
                "analysis_unit",
                "unit_id",
                "Unit",
                "Group",
                "group",
                "Condition",
                "condition",
            ],
        )
        _set_editable_combo_items(
            self.metadata_sample_id_col_edit,
            columns,
            preferred_text=preferred_sample or "sample_id",
            preferred_index=0,
        )
        _set_editable_combo_items(
            self.metadata_analysis_unit_col_edit,
            columns,
            preferred_text=preferred_unit or "analysis_unit_id",
            preferred_index=1,
        )

    def _sync_genome_lineage_column_visibility(self) -> None:
        self.lineage_columns_box.setVisible(bool(self.genome_lineage_table_edit.text().strip()))

    def _browse_peptide_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select peptide table",
            _initial_dialog_path(self.peptide_table_edit.text(), self._last_browse_dir),
            "TSV files (*.tsv *.txt);;Parquet files (*.parquet *.pq);;All files (*.*)",
        )
        if path:
            self.peptide_table_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _browse_output_tsv(self) -> None:
        current_value = self.output_tsv_edit.text().strip()
        initial_path = _initial_dialog_path(current_value, self._last_browse_dir, "genome_presence.tsv")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output result TSV",
            initial_path,
            "TSV files (*.tsv);;All files (*.*)",
        )
        if path:
            self.output_tsv_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)
            if not self.cache_path_edit.text().strip():
                out_path = Path(path)
                suggested = out_path.with_name(f"{out_path.stem}_artifacts") / "matched_peptides.pkl"
                self.cache_path_edit.setText(str(suggested))
            if not self.theoretical_opportunity_cache_edit.text().strip():
                out_path = Path(path)
                suggested = out_path.with_name(f"{out_path.stem}_artifacts") / "theoretical_opportunity_cache.pkl"
                self.theoretical_opportunity_cache_edit.setText(str(suggested))

    def _browse_genome_lineage_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select genome-Lineage table",
            _initial_dialog_path(self.genome_lineage_table_edit.text(), self._last_browse_dir),
            "TSV files (*.tsv *.txt);;All files (*.*)",
        )
        if path:
            self.genome_lineage_table_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _browse_metadata_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select sample metadata table",
            _initial_dialog_path(self.metadata_table_edit.text(), self._last_browse_dir),
            "TSV/CSV files (*.tsv *.txt *.csv);;All files (*.*)",
        )
        if path:
            self.metadata_table_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _update_sample_mapping_status(self) -> None:
        if not self._sample_unit_mapping_rows:
            self.sample_mapping_status_label.setText("No custom sample mapping configured.")
            return
        included = [row for row in self._sample_unit_mapping_rows if bool(row.get("included", True))]
        units = {str(row.get("analysis_unit_id", "")).strip() for row in included if str(row.get("analysis_unit_id", "")).strip()}
        self.sample_mapping_status_label.setText(
            f"Custom mapping: {len(included)} included sample(s), {len(units)} unit(s)."
        )

    def _configure_sample_unit_mapping(self) -> None:
        loading_dialog = QProgressDialog(
            "Loading sample/unit mapping preview...\nThis can take a while for large parquet or long tables.",
            "",
            0,
            0,
            self,
        )
        loading_dialog.setWindowTitle("Loading Samples")
        loading_dialog.setMinimumDuration(0)
        loading_dialog.setCancelButton(None)
        loading_dialog.setAutoClose(False)
        loading_dialog.setAutoReset(False)
        self.configure_sample_mapping_button.setEnabled(False)
        loading_dialog.show()
        QApplication.processEvents()
        dialog = None
        try:
            peptide_table_path = self.peptide_table_edit.text().strip()
            peptide_error_cutoff = _parse_required_float(
                self.peptide_error_cutoff_edit.text(), "Peptide error cutoff"
            )
            intensity_min_value = int(self.intensity_min_value_spin.value())
            # spin provides percent (0-100); convert to fraction for backend (0-1)
            intensity_min_quantile = float(self.intensity_min_quantile_spin.value()) / 100.0
            try:
                peptide_stat = Path(peptide_table_path).expanduser().stat()
                table_signature = (int(peptide_stat.st_size), int(peptide_stat.st_mtime_ns))
            except OSError:
                table_signature = (None, None)
            cache_key = (
                peptide_table_path,
                table_signature,
                self.sample_id_col_edit.currentText().strip(),
                self.peptide_seq_col_edit.currentText().strip(),
                self.intensity_col_edit.currentText().strip(),
                self.peptide_error_col_edit.currentText().strip(),
                float(peptide_error_cutoff),
                self.peptide_decoy_flag_col_edit.currentText().strip(),
                self.decoy_flag_value_edit.currentText().strip(),
                int(intensity_min_value),
                float(intensity_min_quantile),
            )
            if self._sample_unit_preview_cache_key == cache_key:
                rows = [dict(row) for row in self._sample_unit_preview_cache_rows]
            else:
                rows = _read_sample_unit_preview_rows(
                    peptide_table_path=peptide_table_path,
                    sample_id_col=self.sample_id_col_edit.currentText().strip(),
                    peptide_seq_col=self.peptide_seq_col_edit.currentText().strip(),
                    intensity_col=self.intensity_col_edit.currentText().strip(),
                    peptide_error_col=self.peptide_error_col_edit.currentText().strip(),
                    peptide_error_cutoff=peptide_error_cutoff,
                    peptide_decoy_flag_col=self.peptide_decoy_flag_col_edit.currentText().strip(),
                    decoy_flag_value=self.decoy_flag_value_edit.currentText().strip(),
                    intensity_min_value=intensity_min_value,
                    intensity_min_quantile=intensity_min_quantile,
                )
                self._sample_unit_preview_cache_key = cache_key
                self._sample_unit_preview_cache_rows = [dict(row) for row in rows]
            if self._sample_unit_mapping_rows and self._sample_unit_mapping_source_path == peptide_table_path:
                existing = {
                    str(row.get("sample_id", "")): row
                    for row in self._sample_unit_mapping_rows
                    if str(row.get("sample_id", ""))
                }
                for row in rows:
                    previous = existing.get(str(row.get("sample_id", "")))
                    if previous:
                        row["analysis_unit_id"] = previous.get("analysis_unit_id", row["sample_id"])
                        row["included"] = bool(previous.get("included", row.get("included", True)))
            loading_dialog.setLabelText("Opening sample/unit mapping editor...")
            QApplication.processEvents()
            dialog = SampleUnitMappingDialog(
                rows=rows,
                metadata_path=self.metadata_table_edit.text().strip(),
                metadata_sample_col=self.metadata_sample_id_col_edit.currentText().strip() or "sample_id",
                metadata_unit_col=self.metadata_analysis_unit_col_edit.currentText().strip() or "analysis_unit_id",
                parent=self,
            )
        except Exception as exc:
            loading_dialog.close()
            self.configure_sample_mapping_button.setEnabled(True)
            QMessageBox.critical(self, "Cannot Read Samples", str(exc))
            return
        finally:
            loading_dialog.close()
            self.configure_sample_mapping_button.setEnabled(True)

        if dialog is None:
            return
        if _exec_qt_object(dialog) != QDIALOG_ACCEPTED:
            return
        self._sample_unit_mapping_rows = dialog.mapping_rows()
        self._sample_unit_mapping_source_path = peptide_table_path
        self.unit_aware_checkbox.setChecked(True)
        self._update_sample_mapping_status()

    def _materialize_sample_unit_mapping(self, config: ScoringConfig) -> None:
        if not self._sample_unit_mapping_rows:
            return
        if self._sample_unit_mapping_source_path and self._sample_unit_mapping_source_path != config.peptide_table_path:
            raise ValueError(
                "The custom sample/unit mapping was created for a different peptide table. "
                "Open Configure Sample / Unit Mapping again for the current input."
            )
        out_path = Path(config.output_tsv_path).expanduser()
        mapping_path = out_path.with_name(f"{out_path.stem}_gui_sample_unit_mapping.tsv")
        rows = [
            {
                "sample_id": str(row.get("sample_id", "")).strip(),
                "analysis_unit_id": str(row.get("analysis_unit_id", "")).strip(),
                "included": "true" if bool(row.get("included", True)) else "false",
                "n_valid_peptides": int(row.get("n_valid_peptides", 0) or 0),
                "n_total_rows": int(row.get("n_total_rows", 0) or 0),
            }
            for row in self._sample_unit_mapping_rows
            if str(row.get("sample_id", "")).strip()
        ]
        with open(mapping_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "analysis_unit_id", "included", "n_valid_peptides", "n_total_rows"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        config.metadata_table_path = str(mapping_path)
        config.metadata_sample_id_col = "sample_id"
        config.metadata_analysis_unit_col = "analysis_unit_id"

    def _browse_cache_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select matched peptide cache",
            _initial_dialog_path(self.cache_path_edit.text(), self._last_browse_dir, "matched_peptides.pkl"),
            "Pickle files (*.pkl);;All files (*.*)",
        )
        if path:
            self.cache_path_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _browse_theoretical_opportunity_cache(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select theoretical opportunity cache",
            _initial_dialog_path(
                self.theoretical_opportunity_cache_edit.text(),
                self._last_browse_dir,
                "theoretical_opportunity_cache.pkl",
            ),
            "Pickle files (*.pkl);;All files (*.*)",
        )
        if path:
            self.theoretical_opportunity_cache_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _sync_unique_mode_visibility(self) -> None:
        mode = str(self.unique_pvalue_mode_combo.currentData() or "empirical-background")
        error_source = str(self.unique_peptide_error_source_combo.currentData() or "global-alpha")
        show_alpha_mode = mode == "alpha-upper-bound"
        show_alpha = show_alpha_mode and error_source == "global-alpha"
        show_effective_count = show_alpha_mode
        self.unique_peptide_error_source_label.setVisible(show_alpha_mode)
        self.unique_peptide_error_source_combo.setVisible(show_alpha_mode)
        show_exact_cache = mode in {"hypergeometric-opportunity", "empirical-background"}
        self.unique_alpha_label.setVisible(show_alpha)
        self.single_peptide_error_rate_upper_bound_edit.setVisible(show_alpha)
        self.unique_count_power_label.setVisible(show_effective_count)
        self.unique_count_power_spin.setVisible(show_effective_count)
        self.theoretical_opportunity_cache_label.setVisible(show_exact_cache)
        self.theoretical_opportunity_cache_edit.parentWidget().setVisible(show_exact_cache)
        self.rebuild_theoretical_opportunity_cache_checkbox.setVisible(show_exact_cache)
        self.theoretical_opportunity_processes_label.setVisible(show_exact_cache)
        self.theoretical_opportunity_processes_spin.setVisible(show_exact_cache)

    def _sync_unit_aware_visibility(self) -> None:
        unit_aware = self.unit_aware_checkbox.isChecked()
        self.unit_box.setVisible(unit_aware)
        self.sample_filter_box.setVisible(unit_aware)
        self.metadata_box.setVisible(unit_aware)
        self.export_unit_derived_tables_checkbox.setVisible(unit_aware)
        self.configure_sample_mapping_button.setVisible(unit_aware)
        self.sample_mapping_status_label.setVisible(unit_aware)
        self._sync_unique_mode_visibility()

    def _load_last_excluded_genomes(self) -> None:
        found, values = self._read_saved_genome_filter_values("exclude_genome_ids")
        if found:
            self.exclude_text.setPlainText("\n".join(values))
            return
        QMessageBox.information(self, "No Saved Filters", "No saved excluded genome IDs were found.")

    def _load_last_selected_genomes(self) -> None:
        found, values = self._read_saved_genome_filter_values("selected_genome_ids")
        if found:
            self.selected_genomes_text.setPlainText("\n".join(values))
            return
        QMessageBox.information(self, "No Saved Filters", "No saved selected genome IDs were found.")

    def _read_saved_genome_filter_values(self, key: str) -> tuple[bool, list[str]]:
        state_path = _default_gui_state_path()
        if not state_path.exists():
            return False, []
        try:
            with open(state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot Load Saved Filters", str(exc))
            return False, []

        filters = state.get("genome_selection_filters", {})
        if not isinstance(filters, dict) or key not in filters:
            return False, []
        raw_values = filters.get(key, [])
        if isinstance(raw_values, str):
            return True, _parse_text_list(raw_values)
        if isinstance(raw_values, list):
            return True, [str(value).strip() for value in raw_values if str(value).strip()]
        return True, []

    def _add_genome_dir(self) -> None:
        path = _choose_directory(
            self,
            "Select digested genome directory",
            _initial_dialog_path("", self._last_browse_dir),
        )
        if path:
            existing = {self.genome_dir_list.item(i).text() for i in range(self.genome_dir_list.count())}
            if path not in existing:
                self.genome_dir_list.addItem(path)
            self._last_browse_dir = _remember_dialog_directory(path)

    def _remove_genome_dir(self) -> None:
        for item in self.genome_dir_list.selectedItems():
            row = self.genome_dir_list.row(item)
            self.genome_dir_list.takeItem(row)

    def _clear_genome_dirs(self) -> None:
        self.genome_dir_list.clear()

    def build_config(self, require_required_fields: bool = True) -> ScoringConfig:
        config = ScoringConfig(
            peptide_table_path=self.peptide_table_edit.text().strip(),
            genome_lineage_table_path=self.genome_lineage_table_edit.text().strip(),
            genome_lineage_genome_id_col=self.genome_lineage_genome_id_col_edit.currentText().strip(),
            genome_lineage_lineage_col=self.genome_lineage_lineage_col_edit.currentText().strip(),
            genome_digest_dirs=[self.genome_dir_list.item(i).text() for i in range(self.genome_dir_list.count())],
            selected_genome_ids=_parse_text_list(self.selected_genomes_text.toPlainText()),
            output_tsv_path=self.output_tsv_edit.text().strip(),
            peptide_seq_col=self.peptide_seq_col_edit.currentText().strip(),
            peptide_score_col=self.peptide_score_col_edit.currentText().strip(),
            peptide_error_col=self.peptide_error_col_edit.currentText().strip(),
            peptide_error_cutoff=_parse_required_float(
                self.peptide_error_cutoff_edit.text(), "Peptide error cutoff"
            ),
            single_peptide_error_rate_upper_bound=_parse_required_float(
                self.single_peptide_error_rate_upper_bound_edit.text(),
                "Unique evidence alpha",
            ),
            unique_pvalue_mode=str(self.unique_pvalue_mode_combo.currentData() or "empirical-background"),
            unique_peptide_error_source=str(self.unique_peptide_error_source_combo.currentData() or "global-alpha"),
            unique_count_power=float(self.unique_count_power_spin.value()),
            unit_aware=self.unit_aware_checkbox.isChecked(),
            sample_id_col=self.sample_id_col_edit.currentText().strip(),
            intensity_col=self.intensity_col_edit.currentText().strip(),
            intensity_min_value=int(self.intensity_min_value_spin.value()),
                intensity_min_quantile=float(self.intensity_min_quantile_spin.value()) / 100.0,
            metadata_table_path=self.metadata_table_edit.text().strip(),
            metadata_sample_id_col=self.metadata_sample_id_col_edit.currentText().strip(),
            metadata_analysis_unit_col=self.metadata_analysis_unit_col_edit.currentText().strip(),
            export_unit_derived_tables=self.export_unit_derived_tables_checkbox.isChecked(),
            theoretical_opportunity_cache_path=self.theoretical_opportunity_cache_edit.text().strip(),
            rebuild_theoretical_opportunity_cache=self.rebuild_theoretical_opportunity_cache_checkbox.isChecked(),
            num_workers_for_theoretical_opportunity=(
                int(self.theoretical_opportunity_processes_spin.value())
                if self.theoretical_opportunity_processes_spin.value() > 0
                else None
            ),
            peptide_decoy_flag_col=self.peptide_decoy_flag_col_edit.currentText().strip(),
            decoy_flag_value=self.decoy_flag_value_edit.currentText().strip(),
            exclude_genome_ids=_parse_text_list(self.exclude_text.toPlainText()),
            num_workers=int(self.processes_spin.value()),
            knockoff_mc_iterations=_parse_required_int(
                self.knockoff_mc_iterations_edit.text(), "Knockoff MC iterations"
            ),
            knockoff_stage2_mc_iterations=_parse_optional_int(
                self.knockoff_stage2_mc_iterations_edit.text(), "Stage 2 MC iterations"
            ),
            knockoff_stage2_p_exist_ranges=_parse_range_pairs(self.knockoff_stage2_ranges_edit.text()),
            knockoff_random_seed=_parse_required_int(self.knockoff_random_seed_edit.text(), "Random seed"),
            knockoff_top_n_targets=(
                int(self.knockoff_top_n_targets_spin.value())
                if self.knockoff_top_n_targets_spin.value() > 0
                else None
            ),
            matched_peptides_cache_path=self.cache_path_edit.text().strip(),
            save_matched_peptides_cache=self.save_cache_checkbox.isChecked(),
            use_cache_if_exists=self.use_cache_if_exists_checkbox.isChecked(),
            compute_coverage=self.compute_coverage_checkbox.isChecked(),
            export_temp=self.export_temp_checkbox.isChecked(),
            export_peptide_contrib_topN=int(self.export_peptide_contrib_topn_spin.value()),
            return_full_table=self.return_full_table_checkbox.isChecked(),
        )
        if not config.unit_aware:
            config.export_unit_derived_tables = False

        if require_required_fields:
            if not config.peptide_table_path:
                raise ValueError("Please choose an observed peptide table.")
            _require_existing_file(config.peptide_table_path, "Observed peptide table")
            if config.genome_lineage_table_path:
                _require_existing_file(config.genome_lineage_table_path, "Genome-Lineage table")
                if not config.genome_lineage_genome_id_col:
                    raise ValueError("Please provide the genome ID column name for the genome-Lineage table.")
                if not config.genome_lineage_lineage_col:
                    raise ValueError("Please provide the Lineage column name for the genome-Lineage table.")
            if not config.output_tsv_path:
                raise ValueError("Please choose an output result TSV file.")
            _require_output_parent_directory(config.output_tsv_path, "output result TSV file")
            if not config.peptide_seq_col:
                raise ValueError("Please provide the sequence column name.")
            if not config.peptide_score_col:
                raise ValueError("Please provide the score column name.")
            if config.unit_aware:
                if not config.sample_id_col:
                    raise ValueError("Please provide the sample ID column name for unit-aware scoring.")
                if not config.intensity_col:
                    raise ValueError("Please provide the intensity column name for unit-aware scoring.")
                if config.intensity_min_quantile < 0 or config.intensity_min_quantile > 1:
                    raise ValueError("Minimum within-sample intensity quantile must be between 0 and 1.")
                if config.metadata_table_path and not self._sample_unit_mapping_rows:
                    _require_existing_file(config.metadata_table_path, "sample metadata table")
                    if not config.metadata_sample_id_col:
                        raise ValueError("Please provide the metadata sample ID column name.")
                    if not config.metadata_analysis_unit_col:
                        raise ValueError("Please provide the metadata analysis unit column name.")
            if not config.genome_digest_dirs:
                raise ValueError("Please add at least one genome digest directory.")
            for genome_dir in config.genome_digest_dirs:
                _require_existing_directory(genome_dir, "Genome digest directory")
            if config.matched_peptides_cache_path:
                _require_output_parent_directory(config.matched_peptides_cache_path, "matched peptide cache")
            if config.theoretical_opportunity_cache_path:
                _require_output_parent_directory(config.theoretical_opportunity_cache_path, "theoretical opportunity cache")
            if config.unit_aware:
                self._materialize_sample_unit_mapping(config)
        return config

    def load_config(self, config: ScoringConfig) -> None:
        self.peptide_table_edit.setText(config.peptide_table_path)
        self.genome_lineage_table_edit.setText(config.genome_lineage_table_path)
        self.genome_lineage_genome_id_col_edit.setEditText(config.genome_lineage_genome_id_col)
        self.genome_lineage_lineage_col_edit.setEditText(config.genome_lineage_lineage_col)
        self._sync_genome_lineage_column_visibility()
        self.output_tsv_edit.setText(config.output_tsv_path)
        self.peptide_seq_col_edit.setEditText(config.peptide_seq_col)
        self.peptide_score_col_edit.setEditText(config.peptide_score_col)
        self.peptide_error_col_edit.setEditText(config.peptide_error_col)
        self.sample_id_col_edit.setEditText(config.sample_id_col)
        self.intensity_col_edit.setEditText(config.intensity_col)
        self.peptide_error_cutoff_edit.setText(str(config.peptide_error_cutoff))
        self.single_peptide_error_rate_upper_bound_edit.setText(
            str(config.single_peptide_error_rate_upper_bound)
        )
        _set_combo_to_data(self.unique_pvalue_mode_combo, str(config.unique_pvalue_mode))
        _set_combo_to_data(self.unique_peptide_error_source_combo, str(config.unique_peptide_error_source))
        self.unique_count_power_spin.setValue(float(config.unique_count_power))
        self.unit_aware_checkbox.setChecked(bool(config.unit_aware))
        self.export_unit_derived_tables_checkbox.setChecked(bool(config.export_unit_derived_tables))
        # Clamp to QSpinBox maximum to avoid overflow
        try:
            iv = int(config.intensity_min_value)
        except Exception:
            iv = 0
        if iv > 2147483647:
            iv = 2147483647
        self.intensity_min_value_spin.setValue(iv)
        # config stores fraction (0-1); display percent (0-100)
        try:
            q = float(config.intensity_min_quantile)
        except Exception:
            q = 0.0
        if q < 0.0:
            q = 0.0
        if q > 1.0:
            q = 1.0
        self.intensity_min_quantile_spin.setValue(q * 100.0)
        self.metadata_table_edit.setText(config.metadata_table_path)
        self.metadata_sample_id_col_edit.setEditText(config.metadata_sample_id_col)
        self.metadata_analysis_unit_col_edit.setEditText(config.metadata_analysis_unit_col)
        self._sample_unit_mapping_rows = []
        self._sample_unit_mapping_source_path = config.peptide_table_path
        self._update_sample_mapping_status()
        self.theoretical_opportunity_cache_edit.setText(config.theoretical_opportunity_cache_path)
        self.rebuild_theoretical_opportunity_cache_checkbox.setChecked(bool(config.rebuild_theoretical_opportunity_cache))
        self.theoretical_opportunity_processes_spin.setValue(
            int(config.num_workers_for_theoretical_opportunity)
            if config.num_workers_for_theoretical_opportunity is not None
            else 0
        )
        self._sync_unit_aware_visibility()
        self.peptide_decoy_flag_col_edit.setEditText(config.peptide_decoy_flag_col)
        self.decoy_flag_value_edit.setEditText(config.decoy_flag_value)
        self.processes_spin.setValue(config.num_workers if config.num_workers is not None else DEFAULT_PROCESS_COUNT)
        self.knockoff_mc_iterations_edit.setText(str(config.knockoff_mc_iterations))
        self.knockoff_stage2_mc_iterations_edit.setText(
            "" if config.knockoff_stage2_mc_iterations is None else str(config.knockoff_stage2_mc_iterations)
        )
        self.knockoff_stage2_ranges_edit.setText(_format_range_pairs(config.knockoff_stage2_p_exist_ranges))
        self.knockoff_random_seed_edit.setText(str(config.knockoff_random_seed))
        self.knockoff_top_n_targets_spin.setValue(
            config.knockoff_top_n_targets if config.knockoff_top_n_targets is not None else 0
        )
        self.cache_path_edit.setText(config.matched_peptides_cache_path)
        self.export_peptide_contrib_topn_spin.setValue(int(config.export_peptide_contrib_topN))
        self.save_cache_checkbox.setChecked(config.save_matched_peptides_cache)
        self.use_cache_if_exists_checkbox.setChecked(config.use_cache_if_exists)
        self.compute_coverage_checkbox.setChecked(config.compute_coverage)
        self.export_temp_checkbox.setChecked(config.export_temp)
        self.return_full_table_checkbox.setChecked(config.return_full_table)
        self._sync_unit_aware_visibility()
        self._last_browse_dir = _remember_dialog_directory(
            config.peptide_table_path
            or config.genome_lineage_table_path
            or config.output_tsv_path
            or config.matched_peptides_cache_path
        )

        self.genome_dir_list.clear()
        for path in config.genome_digest_dirs:
            self.genome_dir_list.addItem(path)

        self.exclude_text.setPlainText("\n".join(config.exclude_genome_ids))
        self.selected_genomes_text.setPlainText("\n".join(config.selected_genome_ids))


class WorkflowWorker(QObject):
    log_message = Signal(str)
    finished = Signal(str, dict)
    failed = Signal(str, str)

    def __init__(self, task_name: str, config):
        super().__init__()
        self.task_name = task_name
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            if self.task_name == "digest":
                result = run_digest_workflow(self.config, self.log_message.emit)
            elif self.task_name == "parquet_extraction":
                result = run_parquet_extraction_workflow(self.config, self.log_message.emit)
            else:
                result = run_scoring_workflow(self.config, self.log_message.emit)
            self.finished.emit(self.task_name, result)
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.failed.emit(self.task_name, error_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"MetaUmbra GUI v{APP_VERSION}")
        self.resize(1200, 900)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.worker_thread: QThread | None = None
        self.worker: WorkflowWorker | None = None
        self._stop_requested = False
        self._task_result_handled = False
        self._last_config_dir = ""
        self._build_ui()
        self._apply_styles()
        self._set_status_state("idle", "Ready")
        self._load_gui_state()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("AppRoot")
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        top_bar = QWidget()
        top_bar.setObjectName("TopBar")
        button_row = QHBoxLayout(top_bar)
        button_row.setContentsMargins(14, 12, 14, 12)
        button_row.setSpacing(10)
        self.app_title_label = QLabel(f"MetaUmbra v{APP_VERSION}")
        self.app_title_label.setObjectName("AppTitle")
        self.load_button = QPushButton("Load Config")
        self.save_button = QPushButton("Save Config")
        self.clear_settings_button = QPushButton("Clear Settings")
        self.clear_log_button = QPushButton("Clear Log")
        self.about_button = QPushButton("About")
        self.about_button.setObjectName("AboutButton")
        self.about_button.setFixedHeight(28)
        self.about_button.setMaximumWidth(72)
        self.state_badge = QLabel("Idle")
        self.state_badge.setObjectName("StatusBadge")
        self.status_label = ElidedLabel("Ready")
        self.status_label.setObjectName("StatusDetail")
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setVisible(False)
        self.busy_bar.setFixedWidth(180)
        # button_row.addWidget(self.app_title_label)
        button_row.addSpacing(8)
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.clear_settings_button)
        button_row.addWidget(self.clear_log_button)
        button_row.addStretch(1)
        button_row.addWidget(self.busy_bar)
        button_row.addWidget(self.state_badge)
        button_row.addWidget(self.status_label, 1)
        button_row.addSpacing(8)
        button_row.addWidget(self.about_button)
        root_layout.addWidget(top_bar)

        splitter = QSplitter(QT_VERTICAL)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("PrimaryTabs")
        self.digest_tab = DigestTab()
        self.scoring_tab = ScoringTab()
        self.tabs.addTab(self.scoring_tab, "Genome Presence Scoring")
        self.tabs.addTab(self.digest_tab, "Digest FASTA")
        self.tab_actions = QWidget()
        self.tab_actions.setObjectName("TabActions")
        tab_actions_layout = QHBoxLayout(self.tab_actions)
        tab_actions_layout.setContentsMargins(0, 0, 0, 0)
        tab_actions_layout.setSpacing(0)
        self.run_current_button = QPushButton()
        self.run_current_button.setObjectName("PrimaryRunButton")
        self.run_current_button.setProperty("accent", True)
        self.run_current_button.setMinimumHeight(38)
        self.run_current_button.setMinimumWidth(PRIMARY_BUTTON_MIN_WIDTH)
        tab_actions_layout.addWidget(self.run_current_button)
        self.tabs.setCornerWidget(self.tab_actions, QT_TOP_RIGHT_CORNER)

        log_panel = QGroupBox("Run Log")
        log_panel.setProperty("elevated", True)
        log_layout = QVBoxLayout(log_panel)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setObjectName("RunLog")
        log_layout.addWidget(self.log_output)

        splitter.addWidget(self.tabs)
        splitter.addWidget(log_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([920, 210])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self.load_button.clicked.connect(self._load_config)
        self.save_button.clicked.connect(self._save_config)
        self.clear_settings_button.clicked.connect(self._clear_gui_settings)
        self.clear_log_button.clicked.connect(self.log_output.clear)
        self.about_button.clicked.connect(self._show_about_dialog)
        self.run_current_button.clicked.connect(self._handle_run_button_clicked)
        self.scoring_tab.import_parquet_button.clicked.connect(self._open_parquet_import_dialog)
        self.tabs.currentChanged.connect(self._sync_primary_run_button)
        self._sync_primary_run_button()

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def _show_about_dialog(self) -> None:
        about_box = QMessageBox(self)
        about_box.setWindowTitle(f"About MetaUmbra v{APP_VERSION}")
        about_box.setTextFormat(QT_RICH_TEXT)
        about_box.setStandardButtons(QMSG_OK)
        about_box.setText(
            (
                f'<div style="min-width: 520px;">'
                f'<h2 style="margin-bottom: 4px;">MetaUmbra GUI v{APP_VERSION}</h2>'
                '<p style="margin-top: 0; color: #52606d;">'
                "Statistically controlled genome-level presence inference from "
                "metaproteomic peptides.</p>"
                "<hr>"
                "<p><b>Workflows</b></p>"
                "<p>In-silico FASTA digestion, DIA-NN parquet peptide-table import, "
                "and genome presence scoring with a peptide-space knockoff null model.</p>"
                "<p><b>Publication</b></p>"
                "<p>Wu Q, Ning Z, Zhang A, et al.<br>"
                '<a href="https://www.biorxiv.org/content/10.64898/2026.04.29.721689">'
                "MetaUmbra: Statistically Controlled Genome-Level Presence Inference "
                "from Metaproteomic Peptides</a><br>"
                "bioRxiv, 2026: 2026.04.29.721689.</p>"
                "<p><b>Links</b></p>"
                '<a href="https://github.com/byemaxx/MetaUmbra">GitHub Repository</a></p>'
                '<p><a href="https://github.com/byemaxx/MetaUmbra/blob/main/docs/usage.md">'
                "Open Document</a><br>"
                "</div>"
            )
        )
        about_label = about_box.findChild(QLabel, "qt_msgbox_label")
        if about_label is not None:
            about_label.setMinimumWidth(520)
            about_label.setWordWrap(True)
            about_label.setOpenExternalLinks(True)
            about_label.setTextInteractionFlags(QT_TEXT_BROWSER_INTERACTION)
        _exec_qt_object(about_box)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4f6f8;
                color: #1f2933;
            }
            QDialog {
                background: #f4f6f8;
                color: #1f2933;
            }
            QWidget#AppRoot {
                background: #f4f6f8;
            }
            QWidget#TopBar {
                background: #ffffff;
                border: 1px solid #d6dce3;
                border-radius: 3px;
            }
            QLabel#AppTitle {
                color: #1f2933;
                font-size: 15px;
                font-weight: 800;
                padding-right: 2px;
            }
            QTabWidget#PrimaryTabs::pane {
                border: 1px solid #d6dce3;
                border-radius: 3px;
                background: #ffffff;
                top: 0px;
                margin-top: 6px;
            }
            QTabWidget#PrimaryTabs::tab-bar {
                left: 6px;
            }
            QWidget#TabActions {
                background: transparent;
                margin-right: 10px;
            }
            QTabBar::tab {
                background: #e9edf2;
                color: #52606d;
                border: 1px solid #d6dce3;
                padding: 9px 16px;
                min-width: 220px;
                margin-right: 4px;
                margin-top: 2px;
                font-weight: 600;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #1f2933;
                border-color: #c8d0da;
                border-bottom-color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background: #f1f4f7;
                color: #364452;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d6dce3;
                border-radius: 3px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox[elevated="true"] {
                border-color: #cfd6df;
                background: #ffffff;
            }
            QGroupBox[subtle="true"] {
                background: #f8fafc;
                border-color: #dde3ea;
                border-radius: 3px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #334155;
            }
            QWidget#FormCanvas {
                background: transparent;
            }
            QWidget#InlineOptionRow {
                background: #f8fafc;
                border: 1px solid #dde3ea;
                border-radius: 3px;
            }
            QLabel#StatusBadge {
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: 700;
                background: #edf1f5;
                color: #455565;
            }
            QLabel#StatusBadge[statusState="running"] {
                background: #e6f0ff;
                color: #1d5fbf;
            }
            QLabel#StatusBadge[statusState="done"] {
                background: #e4f4ea;
                color: #1b7a43;
            }
            QLabel#StatusBadge[statusState="failed"] {
                background: #fde8e8;
                color: #b42318;
            }
            QLabel#StatusDetail {
                color: #596879;
                padding-left: 6px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QListWidget {
                background: #ffffff;
                border: 1px solid #c7d0da;
                border-radius: 3px;
                padding: 6px 9px;
                selection-background-color: #cfe0ff;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus, QListWidget:focus {
                border-color: #2563eb;
                background: #ffffff;
            }
            QComboBox QLineEdit {
                border: none;
                background: transparent;
                padding: 0;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border: none;
                border-left: 1px solid #e1e7ef;
            }
            QComboBox::down-arrow {
                width: 9px;
                height: 9px;
            }
            QPlainTextEdit#RunLog {
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
                line-height: 1.3;
                background: #f9fafb;
                color: #263545;
                border-color: #d6dce3;
            }
            QPushButton {
                background: #f9fafb;
                border: 1px solid #bdc7d2;
                border-radius: 3px;
                padding: 7px 14px;
                color: #1f2933;
            }
            QPushButton:hover {
                background: #eef2f6;
                border-color: #aeb9c6;
            }
            QPushButton[accent="true"] {
                background: #1f66d1;
                color: white;
                border: 1px solid #1f66d1;
                font-weight: 700;
                border-radius: 3px;
            }
            QPushButton#PrimaryRunButton {
                padding-left: 18px;
                padding-right: 18px;
            }
            QPushButton[accent="true"]:hover {
                background: #1a58b5;
                border-color: #1a58b5;
            }
            QPushButton#PrimaryRunButton[stopAction="true"] {
                color: #b42318;
                border-color: #d7a3a0;
                background: #fff7f7;
            }
            QPushButton#PrimaryRunButton[stopAction="true"]:hover {
                background: #fde8e8;
                border-color: #c97b76;
            }
            QPushButton#AboutButton {
                padding: 4px 10px;
                color: #52606d;
                background: #ffffff;
                border-color: #d6dce3;
                font-size: 12px;
            }
            QPushButton#AboutButton:hover {
                background: #f1f4f7;
                border-color: #c7d0da;
                color: #334155;
            }
            QPushButton:disabled {
                background: #edf0f3;
                color: #8a98a8;
                border-color: #d3dae2;
            }
            QCheckBox#SectionToggle {
                font-weight: 700;
                color: #334155;
                spacing: 10px;
                padding: 2px 0 2px 2px;
            }
            QCheckBox#SectionToggle::indicator {
                width: 16px;
                height: 16px;
                border-radius: 2px;
                border: 1px solid #b8c3cf;
                background: #ffffff;
            }
            QCheckBox#SectionToggle::indicator:checked {
                background: #1f66d1;
                border-color: #1f66d1;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QProgressBar {
                background: #e9edf2;
                border: 1px solid #cfd6df;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: #1f66d1;
                border-radius: 2px;
            }
            QSplitter::handle {
                background: #d7dde5;
                height: 5px;
            }
            """
        )

    def _set_status_state(self, state: str, detail: str) -> None:
        labels = {
            "idle": "Idle",
            "running": "Running",
            "done": "Done",
            "failed": "Failed",
        }
        self.state_badge.setProperty("statusState", state)
        self.state_badge.setText(labels.get(state, state.title()))
        self.state_badge.style().unpolish(self.state_badge)
        self.state_badge.style().polish(self.state_badge)
        self.status_label.setText(detail)

    def _set_busy_state(self, is_busy: bool, status_text: str) -> None:
        self.busy_bar.setVisible(is_busy)
        self._set_status_state("running" if is_busy else "idle", status_text)
        self.load_button.setEnabled(not is_busy)
        self.save_button.setEnabled(not is_busy)
        self.scoring_tab.setEnabled(not is_busy)
        self.digest_tab.setEnabled(not is_busy)
        self.tabs.tabBar().setEnabled(not is_busy)
        self._sync_primary_run_button()

    def _current_tab_key(self) -> str:
        return "scoring" if self.tabs.currentWidget() is self.scoring_tab else "digest"

    def _sync_primary_run_button(self, index: int | None = None) -> None:
        if self.worker_thread is not None:
            self.run_current_button.setText("Stop Task")
            self.run_current_button.setToolTip("Terminate the currently running task.")
            self.run_current_button.setProperty("stopAction", True)
            self.run_current_button.setProperty("accent", False)
        elif self.tabs.currentWidget() is self.scoring_tab:
            self.run_current_button.setText("Run Genome Presence Scoring")
            self.run_current_button.setToolTip("Run genome presence scoring with the current inputs and mappings.")
            self.run_current_button.setProperty("stopAction", False)
            self.run_current_button.setProperty("accent", True)
        else:
            self.run_current_button.setText("Run Digest")
            self.run_current_button.setToolTip("Run FASTA digestion with the current inputs and settings.")
            self.run_current_button.setProperty("stopAction", False)
            self.run_current_button.setProperty("accent", True)
        self.run_current_button.setEnabled(True)
        self.run_current_button.style().unpolish(self.run_current_button)
        self.run_current_button.style().polish(self.run_current_button)

    def _select_tab_by_key(self, tab_key: str) -> None:
        if tab_key == "scoring":
            self.tabs.setCurrentWidget(self.scoring_tab)
        else:
            self.tabs.setCurrentWidget(self.digest_tab)
        self._sync_primary_run_button()

    @staticmethod
    def _summarize_error_text(error_text: str) -> str:
        lines = [line.strip() for line in error_text.splitlines() if line.strip()]
        if not lines:
            return "Unknown error."
        for line in reversed(lines):
            if line != "Traceback (most recent call last):":
                return line
        return lines[-1]

    def _start_parquet_extraction_task(
        self,
        config: ParquetExtractionConfig,
        status_text: str,
        log_header: str,
        extra_log_lines: list[str] | None = None,
    ) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Task Running", "A task is already running. Please wait for it to finish.")
            return

        self._append_log("")
        if log_header:
            self._append_log(log_header)
        if extra_log_lines:
            for line in extra_log_lines:
                self._append_log(line)

        self._stop_requested = False
        self._task_result_handled = False

        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker("parquet_extraction", config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self._append_log)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.failed.connect(self._on_task_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._on_worker_thread_finished)
        self._set_busy_state(True, status_text)
        self.worker_thread.start()

    def _open_parquet_import_dialog(self) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Task Running", "A task is already running. Please wait for it to finish.")
            return

        dialog = ParquetExtractionDialog(self, initial_dir=self.scoring_tab._last_browse_dir)
        if _exec_qt_object(dialog) != QDIALOG_ACCEPTED:
            return

        config = dialog.build_config(require_required_fields=True)
        self._start_parquet_extraction_task(
            config,
            "Parquet peptide extraction in progress",
            "=== Starting parquet import task ===",
        )

    def _handle_run_button_clicked(self) -> None:
        if self.worker_thread is not None:
            self._terminate_current_task()
            return
        self._run_current_tab()

    def _run_current_tab(self) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Task Running", "A task is already running. Please wait for it to finish.")
            return

        try:
            if self.tabs.currentWidget() is self.scoring_tab:
                task_name = "scoring"
                config = self.scoring_tab.build_config(require_required_fields=True)
                busy_text = "Genome presence scoring in progress"
            else:
                task_name = "digest"
                config = self.digest_tab.build_config(require_required_fields=True)
                busy_text = "Digest workflow in progress"
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return

        self._save_gui_state()

        self._append_log("")
        self._append_log(f"=== Starting {task_name} task ===")
        self._stop_requested = False
        self._task_result_handled = False

        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(task_name, config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self._append_log)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.failed.connect(self._on_task_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._on_worker_thread_finished)
        self._set_busy_state(True, busy_text)
        self.worker_thread.start()

    @Slot(str, dict)
    def _on_task_finished(self, task_name: str, payload: dict) -> None:
        if self._stop_requested:
            return
        self._task_result_handled = True
        summary = self._format_summary(task_name, payload)
        if task_name == "parquet_extraction":
            output_path = str(payload.get("output", ""))
            if output_path:
                self.scoring_tab.peptide_table_edit.setText(output_path)
                self.scoring_tab._last_browse_dir = _remember_dialog_directory(output_path)
                self.tabs.setCurrentWidget(self.scoring_tab)
        self.busy_bar.setVisible(False)
        self._set_status_state("done", summary)
        self._append_log(summary)
        QMessageBox.information(self, "Task Complete", summary)
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.scoring_tab.setEnabled(True)
        self.digest_tab.setEnabled(True)
        self.tabs.tabBar().setEnabled(True)
        self._set_run_buttons_enabled(True)

    @Slot(str, str)
    def _on_task_failed(self, task_name: str, error_text: str) -> None:
        if self._stop_requested:
            return
        self._task_result_handled = True
        self._append_log(error_text)
        error_summary = self._summarize_error_text(error_text)
        self.busy_bar.setVisible(False)
        self._set_status_state("failed", error_summary)
        QMessageBox.critical(
            self,
            "Task Failed",
            f"{task_name} failed.\n\n{error_summary}\n\nSee the run log for the full traceback.",
        )
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.scoring_tab.setEnabled(True)
        self.digest_tab.setEnabled(True)
        self.tabs.tabBar().setEnabled(True)
        self._set_run_buttons_enabled(True)

    @Slot()
    def _on_worker_thread_finished(self) -> None:
        if self._stop_requested and not self._task_result_handled:
            message = "Task terminated by user. Partial output files may remain."
            self._append_log(message)
            self.busy_bar.setVisible(False)
            self._set_status_state("failed", message)
            self.load_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.scoring_tab.setEnabled(True)
            self.digest_tab.setEnabled(True)
            self.tabs.tabBar().setEnabled(True)
            self._set_run_buttons_enabled(True)
            self._task_result_handled = True

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        self._stop_requested = False
        self._sync_primary_run_button()

    def _set_run_buttons_enabled(self, enabled: bool) -> None:
        self.run_current_button.setEnabled(enabled)

    def _terminate_current_task(self) -> None:
        if self.worker_thread is None:
            return

        reply = QMessageBox.warning(
            self,
            "Terminate Task",
            "Terminate the current task now?\n\nPartial output files may remain and should be checked before reuse.",
            QMSG_YES | QMSG_NO,
            QMSG_NO,
        )
        if reply != QMSG_YES:
            return

        self._stop_requested = True
        self.run_current_button.setEnabled(False)
        self._set_status_state("running", "Terminating current task")
        self._append_log("Termination requested by user.")
        self.worker_thread.requestInterruption()
        self.worker_thread.terminate()

    def _run_digest_tab(self) -> None:
        self.tabs.setCurrentWidget(self.digest_tab)
        self._run_current_tab()

    def _run_scoring_tab(self) -> None:
        self.tabs.setCurrentWidget(self.scoring_tab)
        self._run_current_tab()

    def _format_summary(self, task_name: str, payload: dict) -> str:
        if task_name == "parquet_extraction":
            return (
                f"Parquet extraction completed: {payload.get('rows', 0)} rows written to "
                f"{payload.get('output', '')} in {payload.get('elapsed_seconds', 0)} s."
            )

        if task_name == "digest":
            if payload.get("mode") == "file":
                return (
                    f"Digest completed: {payload.get('peptides', 0)} peptides written in "
                    f"{payload.get('elapsed_seconds', 0)} s."
                )
            return (
                f"Digest completed: {payload.get('files_processed', 0)} files processed, "
                f"{payload.get('peptides', 0)} peptides total in {payload.get('elapsed_seconds', 0)} s."
            )

        return (
            f"Scoring completed: {payload.get('rows', 0)} genomes written to "
            f"{payload.get('output', '')} in {payload.get('elapsed_seconds', 0)} s."
        )

    def _save_config(self) -> None:
        try:
            payload = {
                "selected_tab": self._current_tab_key(),
                "digest": asdict(self.digest_tab.build_config(require_required_fields=False)),
                "scoring": asdict(self.scoring_tab.build_config(require_required_fields=False)),
                "scoring_sample_unit_mapping_rows": self.scoring_tab._sample_unit_mapping_rows,
                "scoring_sample_unit_mapping_source_path": self.scoring_tab._sample_unit_mapping_source_path,
            }
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Save Config", str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GUI config",
            _initial_dialog_path("", self._last_config_dir, "metaumbra_gui_config.json"),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self._last_config_dir = _remember_dialog_directory(path)
        self._set_status_state("idle", f"Saved config: {path}")

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load GUI config",
            _initial_dialog_path("", self._last_config_dir),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.digest_tab.load_config(DigestConfig(**payload.get("digest", {})))
        scoring_payload = payload.get("scoring", {})
        if not isinstance(scoring_payload, dict):
            scoring_payload = {}
        scoring_fields = {field.name for field in fields(ScoringConfig)}
        self.scoring_tab.load_config(ScoringConfig(**{k: v for k, v in scoring_payload.items() if k in scoring_fields}))
        if isinstance(payload.get("scoring_sample_unit_mapping_rows"), list):
            self.scoring_tab._sample_unit_mapping_rows = list(payload["scoring_sample_unit_mapping_rows"])
            self.scoring_tab._sample_unit_mapping_source_path = str(
                payload.get("scoring_sample_unit_mapping_source_path") or self.scoring_tab.peptide_table_edit.text().strip()
            )
            self.scoring_tab._update_sample_mapping_status()
        selected_tab = payload.get("selected_tab", "scoring")
        if isinstance(selected_tab, int):
            selected_tab = "scoring" if selected_tab == 0 else "digest"
        if selected_tab == "parquet_extraction":
            selected_tab = "scoring"
        self._select_tab_by_key(str(selected_tab))
        self._last_config_dir = _remember_dialog_directory(path)
        self._set_status_state("idle", f"Loaded config: {path}")

    def closeEvent(self, event) -> None:
        self._save_gui_state()
        if self.worker_thread is None:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Task Running",
            "A task is still running. Close anyway?",
            QMSG_YES | QMSG_NO,
            QMSG_NO,
        )
        if reply == QMSG_YES:
            event.accept()
        else:
            event.ignore()

    def _save_gui_state(self) -> None:
        try:
            state_dir = _default_user_config_dir()
            state_dir.mkdir(parents=True, exist_ok=True)

            state = {
                "version": 2,
                "genome_digest": {
                    "directories": [
                        self.scoring_tab.genome_dir_list.item(i).text()
                        for i in range(self.scoring_tab.genome_dir_list.count())
                    ],
                    "theoretical_opportunity_cache_path": (
                        self.scoring_tab.theoretical_opportunity_cache_edit.text().strip()
                    ),
                },
                "genome_lineage": {
                    "table_path": self.scoring_tab.genome_lineage_table_edit.text().strip(),
                    "genome_id_col": self.scoring_tab.genome_lineage_genome_id_col_edit.currentText().strip(),
                    "lineage_col": self.scoring_tab.genome_lineage_lineage_col_edit.currentText().strip(),
                },
                "genome_selection_filters": {
                    "exclude_genome_ids": _parse_text_list(self.scoring_tab.exclude_text.toPlainText()),
                    "selected_genome_ids": _parse_text_list(self.scoring_tab.selected_genomes_text.toPlainText()),
                },
            }
            with open(_default_gui_state_path(), "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            logging.warning(f"Failed to save GUI state: {exc}")
            self._append_log(f"Warning: Failed to save GUI state: {exc}")

    def _load_gui_state(self) -> None:
        state_path = _default_gui_state_path()
        if not state_path.exists():
            return
        
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            logging.warning(f"Failed to load GUI state: {exc}")
            self._append_log(f"Warning: Failed to load GUI state: {exc}")
            return

        try:
            genome_digest_state = state.get("genome_digest", {})
            restored_digest_dirs: list[str] = []
            saved_digest_dirs: list[str] = []
            if isinstance(genome_digest_state, dict):
                raw_dirs = genome_digest_state.get("directories", [])
                if isinstance(raw_dirs, list):
                    saved_digest_dirs = [str(d) for d in raw_dirs if str(d).strip()]

                self.scoring_tab._clear_genome_dirs()
                for d in saved_digest_dirs:
                    if Path(d).exists() and Path(d).is_dir():
                        self.scoring_tab.genome_dir_list.addItem(d)
                        restored_digest_dirs.append(d)

                if saved_digest_dirs and restored_digest_dirs == saved_digest_dirs:
                    cache_path = str(genome_digest_state.get("theoretical_opportunity_cache_path") or "").strip()
                    self.scoring_tab.theoretical_opportunity_cache_edit.setText(cache_path)

            genome_lineage_state = state.get("genome_lineage", {})
            if isinstance(genome_lineage_state, dict):
                p = str(genome_lineage_state.get("table_path") or "").strip()
                if p and Path(p).exists() and Path(p).is_file():
                    self.scoring_tab.genome_lineage_table_edit.setText(p)

                genome_id_col = genome_lineage_state.get("genome_id_col")
                if genome_id_col is not None:
                    self.scoring_tab.genome_lineage_genome_id_col_edit.setEditText(str(genome_id_col))
                lineage_col = genome_lineage_state.get("lineage_col")
                if lineage_col is not None:
                    self.scoring_tab.genome_lineage_lineage_col_edit.setEditText(str(lineage_col))

            self.scoring_tab._sync_unit_aware_visibility()
            
            self.scoring_tab._sync_genome_lineage_column_visibility()
        except Exception as exc:
            logging.warning(f"Error applying GUI state: {exc}")
            self._append_log(f"Warning: Error applying GUI state: {exc}")

    def _clear_gui_settings(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear Settings",
            "Are you sure you want to clear all GUI settings? This will delete the saved state and cannot be undone.",
            QMSG_YES | QMSG_NO,
            QMSG_NO
        )
        if reply == QMSG_YES:
            state_path = _default_gui_state_path()
            if state_path.exists():
                try:
                    state_path.unlink()
                except Exception as exc:
                    logging.warning(f"Failed to delete GUI state file: {exc}")
            QMessageBox.information(self, "Settings Cleared", "Saved GUI settings have been cleared. Please close and re-open the application for it to take effect.")


def main() -> None:
    app = QApplication([])
    wheel_guard = WheelChangeGuard(app)
    app.installEventFilter(wheel_guard)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow()
    window._wheel_guard = wheel_guard
    window.show()
    _exec_qt_object(app)


if __name__ == "__main__":
    mp.freeze_support()
    main()

