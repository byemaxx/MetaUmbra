from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import traceback
from dataclasses import asdict
from pathlib import Path

try:
    from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSplitter,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PySide6 is required to run the GUI. Install it with: pip install PySide6") from exc

from taxaseeker_workflows import (
    DigestConfig,
    ScoringConfig,
    run_digest_workflow,
    run_scoring_workflow,
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

DEFAULT_PROCESS_COUNT = min(64, max(1, (os.cpu_count() or 1) - 1))
MAX_PROCESS_COUNT = min(64, max(1, os.cpu_count() or 1))
ICON_PATH = Path(__file__).resolve().parent / "assets" / "taxaseeker_icon.png"
FORM_LABEL_MIN_WIDTH = 150
BROWSE_BUTTON_WIDTH = 96
PRIMARY_BUTTON_MIN_WIDTH = 240


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
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)

    content = QWidget()
    content.setObjectName("FormCanvas")
    layout = QVBoxLayout(content)
    layout.setContentsMargins(20, 14, 20, 16)
    layout.setSpacing(14)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    scroll.setWidget(content)
    return scroll, layout


def _create_action_bar(hint_text: str, button_text: str) -> tuple[QWidget, QPushButton]:
    action_bar = QWidget()
    action_bar.setObjectName("ActionBar")
    action_layout = QHBoxLayout(action_bar)
    action_layout.setContentsMargins(18, 12, 18, 12)
    action_layout.setSpacing(12)

    action_hint = QLabel(hint_text)
    action_hint.setObjectName("ActionHint")
    action_hint.setWordWrap(True)
    action_layout.addWidget(action_hint, 1)

    run_button = QPushButton(button_text)
    run_button.setProperty("accent", True)
    run_button.setMinimumHeight(42)
    run_button.setMinimumWidth(PRIMARY_BUTTON_MIN_WIDTH)
    action_layout.addWidget(run_button, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return action_bar, run_button


def _polish_form_layout(form: QFormLayout) -> None:
    form.setHorizontalSpacing(14)
    form.setVerticalSpacing(10)
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, QLabel):
            widget.setMinimumWidth(FORM_LABEL_MIN_WIDTH)
            widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)


def _create_editable_combo(default_text: str = "", placeholder: str = "") -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    if placeholder and combo.lineEdit() is not None:
        combo.lineEdit().setPlaceholderText(placeholder)
    if default_text:
        combo.setEditText(default_text)
    return combo


def _set_editable_combo_items(combo: QComboBox, items: list[str], preferred_text: str = "") -> None:
    current_text = combo.currentText().strip()
    combo.blockSignals(True)
    combo.clear()
    for item in items:
        combo.addItem(item)

    target_text = current_text
    if preferred_text and preferred_text in items:
        if not target_text or target_text not in items:
            target_text = preferred_text
    elif not target_text and items:
        target_text = items[0]

    combo.setEditText(target_text)
    combo.blockSignals(False)


def _clear_editable_combo_items(combo: QComboBox, fallback_text: str = "") -> None:
    combo.blockSignals(True)
    combo.clear()
    combo.setEditText(fallback_text)
    combo.blockSignals(False)


def _read_table_columns(table_path: str) -> list[str]:
    path = table_path.strip()
    if not path or not os.path.isfile(path):
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
        self.output_dir_edit.setPlaceholderText("Suggested: <input_folder>_digested")
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
        options_form = QFormLayout(self.more_options.body)
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
        options_form.addRow("Minimum peptide length", self.min_length_edit)
        options_form.addRow("Maximum peptide length", self.max_length_edit)
        options_form.addRow("Maximum miscleavages", self.max_miscleavages_edit)
        options_form.addRow("Processes", self.processes_spin)
        options_form.addRow(self.short_header_checkbox)
        options_form.addRow(self.verbose_checkbox)
        options_form.addRow(self.skip_existing_checkbox)
        options_form.addRow(
            _make_wrapped_label(
                f"Default processes = CPU cores minus one. Maximum allowed here is {MAX_PROCESS_COUNT}."
            )
        )
        _polish_form_layout(options_form)
        layout.addWidget(self.more_options)
        layout.addStretch(1)

        action_bar, self.run_button = _create_action_bar(
            "Review the inputs above, then start the digest workflow.",
            "Run Digest",
        )
        outer_layout.addWidget(action_bar)

        self.mode_combo.currentIndexChanged.connect(self._sync_mode_visibility)
        self._sync_mode_visibility()

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
        self._sync_mode_visibility()


class ScoringTab(QWidget):
    def __init__(self):
        super().__init__()
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(10)
        self._last_auto_output_tsv = ""
        self._last_browse_dir = ""

        scroll, layout = _create_scroll_form_host()
        outer_layout.addWidget(scroll, 1)

        required_box = QGroupBox("Required")
        required_box.setProperty("elevated", True)
        required_form = QFormLayout(required_box)
        peptide_row, self.peptide_table_edit = _make_path_row("Browse", self._browse_peptide_table, accept_mode="file")
        required_form.addRow("Observed peptide table", peptide_row)
        lineage_row, self.genome_lineage_table_edit = _make_path_row("Browse", self._browse_genome_lineage_table, accept_mode="file")
        required_form.addRow("Genome-Lineage table (optional)", lineage_row)
        output_row, self.output_tsv_edit = _make_path_row("Browse", self._browse_output_tsv, accept_mode="file")
        required_form.addRow("Output result TSV", output_row)

        genome_box = QGroupBox("Genome Digest Directories")
        genome_layout = QVBoxLayout(genome_box)
        self.genome_dir_list = DirectoryDropListWidget()
        self.genome_dir_list.setMinimumHeight(120)
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
        mapping_form = QFormLayout(mapping_box)
        self.peptide_seq_col_edit = _create_editable_combo("Sequence", "Choose or type the peptide sequence column")
        self.peptide_score_col_edit = _create_editable_combo("score", "Choose or type the peptide score column")
        mapping_form.addRow("Sequence column", self.peptide_seq_col_edit)
        mapping_form.addRow("Score column", self.peptide_score_col_edit)
        self.lineage_columns_box = QGroupBox("Genome-Lineage Columns")
        self.lineage_columns_box.setProperty("subtle", True)
        lineage_columns_form = QFormLayout(self.lineage_columns_box)
        self.genome_lineage_genome_id_col_edit = _create_editable_combo(
            placeholder="Required if table is provided, e.g. Genome_id"
        )
        self.genome_lineage_lineage_col_edit = _create_editable_combo(
            placeholder="Required if table is provided, e.g. Lineage"
        )
        lineage_columns_form.addRow("Lineage genome ID column", self.genome_lineage_genome_id_col_edit)
        lineage_columns_form.addRow("Lineage column", self.genome_lineage_lineage_col_edit)
        self.lineage_columns_box.setVisible(False)
        _polish_form_layout(lineage_columns_form)
        mapping_form.addRow(self.lineage_columns_box)
        _polish_form_layout(mapping_form)
        layout.addWidget(mapping_box)

        self.more_options = CollapsibleOptions()
        options_layout = QVBoxLayout(self.more_options.body)

        columns_box = QGroupBox("Peptide Table Columns")
        columns_box.setProperty("subtle", True)
        columns_form = QFormLayout(columns_box)
        self.peptide_error_col_edit = QLineEdit("Q.Value")
        self.peptide_decoy_flag_col_edit = QLineEdit("Reverse")
        self.decoy_flag_value_edit = QLineEdit("+")
        columns_form.addRow("Error column", self.peptide_error_col_edit)
        columns_form.addRow("Decoy flag column", self.peptide_decoy_flag_col_edit)
        columns_form.addRow("Decoy flag value", self.decoy_flag_value_edit)
        _polish_form_layout(columns_form)
        options_layout.addWidget(columns_box)

        runtime_box = QGroupBox("Runtime And Knockoff Settings")
        runtime_box.setProperty("subtle", True)
        runtime_form = QFormLayout(runtime_box)
        self.processes_spin = _create_process_spinbox()
        self.peptide_error_cutoff_edit = QLineEdit("0.05")
        self.knockoff_mc_iterations_edit = QLineEdit("500")
        self.knockoff_stage2_mc_iterations_edit = QLineEdit("2000")
        self.knockoff_stage2_ranges_edit = QLineEdit("0.005-0.02, 0.02-0.08")
        self.knockoff_random_seed_edit = QLineEdit("1")
        self.knockoff_top_n_targets_spin = QSpinBox()
        self.knockoff_top_n_targets_spin.setRange(0, 1000000)
        self.knockoff_top_n_targets_spin.setValue(0)
        self.knockoff_top_n_targets_spin.setToolTip(
            "Only run knockoff inference for the top N genomes ranked by evidence.\n"
            "Use 0 to evaluate all candidate genomes.\n"
            "This is mainly a speed optimization for very large runs."
        )
        cache_row, self.cache_path_edit = _make_path_row("Browse", self._browse_cache_path, accept_mode="file")
        self.export_peptide_contrib_topn_spin = QSpinBox()
        self.export_peptide_contrib_topn_spin.setRange(0, 1000000)
        self.export_peptide_contrib_topn_spin.setValue(0)
        knockoff_limit_label = QLabel("Limit knockoff to top-ranked genomes (0 = all)")
        knockoff_limit_label.setToolTip(
            "Only run knockoff inference for the top N genomes ranked by evidence.\n"
            "Use 0 to evaluate all candidate genomes.\n"
            "This is mainly a speed optimization for very large runs."
        )
        runtime_form.addRow("Processes", self.processes_spin)
        runtime_form.addRow("Peptide error cutoff", self.peptide_error_cutoff_edit)
        runtime_form.addRow("Knockoff MC iterations", self.knockoff_mc_iterations_edit)
        runtime_form.addRow("Stage 2 MC iterations", self.knockoff_stage2_mc_iterations_edit)
        runtime_form.addRow("Stage 2 p ranges", self.knockoff_stage2_ranges_edit)
        runtime_form.addRow("Random seed", self.knockoff_random_seed_edit)
        runtime_form.addRow(knockoff_limit_label, self.knockoff_top_n_targets_spin)
        runtime_form.addRow("Matched peptide cache", cache_row)
        runtime_form.addRow("Export peptide contrib top-N", self.export_peptide_contrib_topn_spin)
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
        options_layout.addWidget(runtime_box)

        exclude_box = QGroupBox("Excluded Genome IDs")
        exclude_box.setProperty("subtle", True)
        exclude_layout = QVBoxLayout(exclude_box)
        self.exclude_text = QTextEdit()
        self.exclude_text.setPlaceholderText("One genome ID per line, or comma-separated.")
        exclude_layout.addWidget(self.exclude_text)
        options_layout.addWidget(exclude_box)

        flags_box = QGroupBox("Flags")
        flags_box.setProperty("subtle", True)
        flags_layout = QVBoxLayout(flags_box)
        self.save_cache_checkbox = QCheckBox("Save matched-peptide cache")
        self.save_cache_checkbox.setChecked(True)
        self.use_cache_if_exists_checkbox = QCheckBox("Reuse cache if it already exists")
        self.use_peptide_error_checkbox = QCheckBox("Use peptide-level error for unique evidence p-value")
        self.compute_coverage_checkbox = QCheckBox("Compute coverage columns")
        self.compute_coverage_checkbox.setChecked(True)
        self.export_temp_checkbox = QCheckBox("Export temp artifacts")
        self.export_temp_checkbox.setChecked(True)
        self.return_full_table_checkbox = QCheckBox("Return full internal table")
        flags_layout.addWidget(self.save_cache_checkbox)
        flags_layout.addWidget(self.use_cache_if_exists_checkbox)
        flags_layout.addWidget(self.use_peptide_error_checkbox)
        flags_layout.addWidget(self.compute_coverage_checkbox)
        flags_layout.addWidget(self.export_temp_checkbox)
        flags_layout.addWidget(self.return_full_table_checkbox)
        options_layout.addWidget(flags_box)
        layout.addWidget(self.more_options)
        layout.addStretch(1)

        action_bar, self.run_button = _create_action_bar(
            "Run genome presence scoring with the selected inputs and mappings.",
            "Run Genome Presence Scoring",
        )
        outer_layout.addWidget(action_bar)
        self.peptide_table_edit.textChanged.connect(self._update_auto_output_tsv_from_peptide_table)
        self.peptide_table_edit.textChanged.connect(self._update_peptide_table_column_options)
        self.genome_lineage_table_edit.textChanged.connect(self._update_genome_lineage_column_options)
        self.genome_lineage_table_edit.textChanged.connect(self._sync_genome_lineage_column_visibility)

    def _suggest_output_tsv_path(self, peptide_table_path: str) -> str:
        peptide_path = Path(peptide_table_path.strip())
        if not peptide_path.name:
            return ""
        return str(peptide_path.with_name(f"{peptide_path.stem}_genome_presence.tsv"))

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
        columns = _read_table_columns(self.peptide_table_edit.text())
        if not columns:
            _clear_editable_combo_items(self.peptide_seq_col_edit, "Sequence")
            _clear_editable_combo_items(self.peptide_score_col_edit, "score")
            return
        _set_editable_combo_items(self.peptide_seq_col_edit, columns, preferred_text="Sequence")
        preferred_score = "score" if "score" in columns else "Score"
        _set_editable_combo_items(self.peptide_score_col_edit, columns, preferred_text=preferred_score)

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
        )

    def _sync_genome_lineage_column_visibility(self) -> None:
        self.lineage_columns_box.setVisible(bool(self.genome_lineage_table_edit.text().strip()))

    def _browse_peptide_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select peptide table",
            _initial_dialog_path(self.peptide_table_edit.text(), self._last_browse_dir),
            "TSV files (*.tsv *.txt);;All files (*.*)",
        )
        if path:
            self.peptide_table_edit.setText(path)
            self._last_browse_dir = _remember_dialog_directory(path)
            if not self.cache_path_edit.text().strip():
                self.cache_path_edit.setText(str(Path(path).with_name("matched_peptides.pkl")))

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
            output_tsv_path=self.output_tsv_edit.text().strip(),
            peptide_seq_col=self.peptide_seq_col_edit.currentText().strip(),
            peptide_score_col=self.peptide_score_col_edit.currentText().strip(),
            peptide_error_col=self.peptide_error_col_edit.text().strip(),
            peptide_error_cutoff=_parse_required_float(
                self.peptide_error_cutoff_edit.text(), "Peptide error cutoff"
            ),
            peptide_decoy_flag_col=self.peptide_decoy_flag_col_edit.text().strip(),
            decoy_flag_value=self.decoy_flag_value_edit.text().strip(),
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
            use_peptide_error_for_unique_pvalue=self.use_peptide_error_checkbox.isChecked(),
            compute_coverage=self.compute_coverage_checkbox.isChecked(),
            export_temp=self.export_temp_checkbox.isChecked(),
            export_peptide_contrib_topN=int(self.export_peptide_contrib_topn_spin.value()),
            return_full_table=self.return_full_table_checkbox.isChecked(),
        )

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
            if not config.genome_digest_dirs:
                raise ValueError("Please add at least one genome digest directory.")
            for genome_dir in config.genome_digest_dirs:
                _require_existing_directory(genome_dir, "Genome digest directory")
            if config.matched_peptides_cache_path:
                _require_output_parent_directory(config.matched_peptides_cache_path, "matched peptide cache")
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
        self.peptide_error_col_edit.setText(config.peptide_error_col)
        self.peptide_error_cutoff_edit.setText(str(config.peptide_error_cutoff))
        self.peptide_decoy_flag_col_edit.setText(config.peptide_decoy_flag_col)
        self.decoy_flag_value_edit.setText(config.decoy_flag_value)
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
        self.use_peptide_error_checkbox.setChecked(config.use_peptide_error_for_unique_pvalue)
        self.compute_coverage_checkbox.setChecked(config.compute_coverage)
        self.export_temp_checkbox.setChecked(config.export_temp)
        self.return_full_table_checkbox.setChecked(config.return_full_table)
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
            else:
                result = run_scoring_workflow(self.config, self.log_message.emit)
            self.finished.emit(self.task_name, result)
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.failed.emit(self.task_name, error_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TaxaSeeker GUI")
        self.resize(1200, 900)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self.worker_thread: QThread | None = None
        self.worker: WorkflowWorker | None = None
        self._last_config_dir = ""
        self._build_ui()
        self._apply_styles()
        self._set_status_state("idle", "Ready")

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.load_button = QPushButton("Load Config")
        self.save_button = QPushButton("Save Config")
        self.clear_log_button = QPushButton("Clear Log")
        self.state_badge = QLabel("Idle")
        self.state_badge.setObjectName("StatusBadge")
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusDetail")
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setVisible(False)
        self.busy_bar.setFixedWidth(180)
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.clear_log_button)
        button_row.addStretch(1)
        button_row.addWidget(self.busy_bar)
        button_row.addWidget(self.state_badge)
        button_row.addWidget(self.status_label)
        root_layout.addLayout(button_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        self.digest_tab = DigestTab()
        self.scoring_tab = ScoringTab()
        self.tabs.addTab(self.digest_tab, "Digest FASTA")
        self.tabs.addTab(self.scoring_tab, "Genome Presence Scoring")

        log_panel = QGroupBox("Run Log")
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
        self.clear_log_button.clicked.connect(self.log_output.clear)
        self.digest_tab.run_button.clicked.connect(self._run_digest_tab)
        self.scoring_tab.run_button.clicked.connect(self._run_scoring_tab)

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f6f8fb;
                color: #1f2a37;
            }
            QTabWidget::pane {
                border: 1px solid #d8e0ea;
                border-radius: 10px;
                background: #fbfcfe;
                top: -1px;
            }
            QTabBar::tab {
                background: transparent;
                color: #5b6776;
                border: 1px solid transparent;
                border-bottom: none;
                padding: 9px 16px;
                margin-right: 6px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #fbfcfe;
                color: #1f2f3f;
                font-weight: 600;
                border-color: #d8e0ea;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #dde4ec;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox[elevated="true"] {
                border-color: #d4dce6;
                background: #ffffff;
            }
            QGroupBox[subtle="true"] {
                background: #fafbfd;
                border-color: #e3e8ef;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #2d3b4b;
            }
            QWidget#FormCanvas {
                background: transparent;
            }
            QLabel#StatusBadge {
                padding: 5px 11px;
                border-radius: 10px;
                font-weight: 700;
                background: #eef2f7;
                color: #425466;
            }
            QLabel#StatusBadge[statusState="running"] {
                background: #e9f2ff;
                color: #1f63c1;
            }
            QLabel#StatusBadge[statusState="done"] {
                background: #e8f6ec;
                color: #1c7a43;
            }
            QLabel#StatusBadge[statusState="failed"] {
                background: #fdeced;
                color: #b42318;
            }
            QLabel#StatusDetail {
                color: #516274;
                padding-left: 4px;
            }
            QLabel#ActionHint {
                color: #637386;
            }
            QWidget#ActionBar {
                border: 1px solid #dbe2ea;
                background: #fbfcfe;
                border-radius: 10px;
            }
            QLineEdit, QComboBox, QSpinBox, QTextEdit, QPlainTextEdit, QListWidget {
                background: #ffffff;
                border: 1px solid #ccd6e0;
                border-radius: 8px;
                padding: 6px 10px;
                selection-background-color: #d9e7ff;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus, QListWidget:focus {
                border-color: #2a6fdb;
            }
            QComboBox::drop-down {
                border: none;
                width: 26px;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
            QPlainTextEdit#RunLog {
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
                line-height: 1.3;
                background: #f8fafc;
                color: #334155;
                border-color: #d7e0ea;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cad3de;
                border-radius: 8px;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background: #f6f9fc;
            }
            QPushButton[accent="true"] {
                background: #1f6feb;
                color: white;
                border: 1px solid #1f6feb;
                font-weight: 700;
            }
            QPushButton[accent="true"]:hover {
                background: #185ec7;
            }
            QPushButton:disabled {
                background: #eef2f6;
                color: #90a0b0;
                border-color: #d8e0e8;
            }
            QCheckBox#SectionToggle {
                font-weight: 700;
                color: #314456;
                spacing: 10px;
                padding: 0 0 0 2px;
            }
            QCheckBox#SectionToggle::indicator {
                width: 16px;
                height: 16px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QSplitter::handle {
                background: #e2e7ee;
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
        self.tabs.setEnabled(not is_busy)
        self._set_run_buttons_enabled(not is_busy)

    @staticmethod
    def _summarize_error_text(error_text: str) -> str:
        lines = [line.strip() for line in error_text.splitlines() if line.strip()]
        if not lines:
            return "Unknown error."
        for line in reversed(lines):
            if line != "Traceback (most recent call last):":
                return line
        return lines[-1]

    def _run_current_tab(self) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Task Running", "A task is already running. Please wait for it to finish.")
            return

        try:
            if self.tabs.currentIndex() == 0:
                task_name = "digest"
                config = self.digest_tab.build_config(require_required_fields=True)
                busy_text = "Digest workflow in progress"
            else:
                task_name = "scoring"
                config = self.scoring_tab.build_config(require_required_fields=True)
                busy_text = "Genome presence scoring in progress"
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return

        self._append_log("")
        self._append_log(f"=== Starting {task_name} task ===")
        self._set_busy_state(True, busy_text)

        self.worker_thread = QThread(self)
        self.worker = WorkflowWorker(task_name, config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self._append_log)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.failed.connect(self._on_task_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    @Slot(str, dict)
    def _on_task_finished(self, task_name: str, payload: dict) -> None:
        summary = self._format_summary(task_name, payload)
        self.busy_bar.setVisible(False)
        self._set_status_state("done", summary)
        self._append_log(summary)
        QMessageBox.information(self, "Task Complete", summary)
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.tabs.setEnabled(True)
        self._set_run_buttons_enabled(True)

    @Slot(str, str)
    def _on_task_failed(self, task_name: str, error_text: str) -> None:
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
        self.tabs.setEnabled(True)
        self._set_run_buttons_enabled(True)

    @Slot()
    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def _set_run_buttons_enabled(self, enabled: bool) -> None:
        self.digest_tab.run_button.setEnabled(enabled)
        self.scoring_tab.run_button.setEnabled(enabled)

    def _run_digest_tab(self) -> None:
        self.tabs.setCurrentIndex(0)
        self._run_current_tab()

    def _run_scoring_tab(self) -> None:
        self.tabs.setCurrentIndex(1)
        self._run_current_tab()

    def _format_summary(self, task_name: str, payload: dict) -> str:
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
                "selected_tab": int(self.tabs.currentIndex()),
                "digest": asdict(self.digest_tab.build_config(require_required_fields=False)),
                "scoring": asdict(self.scoring_tab.build_config(require_required_fields=False)),
            }
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Save Config", str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GUI config",
            _initial_dialog_path("", self._last_config_dir, "taxaseeker_gui_config.json"),
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
        self.scoring_tab.load_config(ScoringConfig(**payload.get("scoring", {})))
        selected_tab = int(payload.get("selected_tab", 0))
        if selected_tab in (0, 1):
            self.tabs.setCurrentIndex(selected_tab)
        self._last_config_dir = _remember_dialog_directory(path)
        self._set_status_state("idle", f"Loaded config: {path}")

    def closeEvent(self, event) -> None:
        if self.worker_thread is None:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Task Running",
            "A task is still running. Close anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


def main() -> None:
    app = QApplication([])
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    main()
