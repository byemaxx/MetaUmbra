from __future__ import annotations

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


def _make_path_row(browse_text: str, browse_handler) -> tuple[QWidget, QLineEdit]:
    wrapper = QWidget()
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    line_edit = QLineEdit()
    button = QPushButton(browse_text)
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


def _choose_directory(parent: QWidget, title: str, initial_path: str = "") -> str:
    return QFileDialog.getExistingDirectory(parent, title, initial_path or "")


class CollapsibleOptions(QWidget):
    def __init__(self, title: str = "More Options"):
        super().__init__()
        layout = QVBoxLayout(self)
        self.toggle = QCheckBox(title)
        self.body = QGroupBox()
        self.body.setVisible(False)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self.body.setVisible)


class DigestTab(QWidget):
    def __init__(self):
        super().__init__()
        outer_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        required_box = QGroupBox("Required")
        required_form = QFormLayout(required_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Digest a directory of FASTA files", "directory")
        self.mode_combo.addItem("Digest one FASTA file", "file")
        required_form.addRow("Mode", self.mode_combo)

        self.file_inputs = QGroupBox("Single File Input")
        file_form = QFormLayout(self.file_inputs)
        file_input_row, self.input_file_edit = _make_path_row("Browse", self._browse_input_file)
        file_output_row, self.output_file_edit = _make_path_row("Browse", self._browse_output_file)
        file_form.addRow("Input FASTA file", file_input_row)
        file_form.addRow("Output TSV file", file_output_row)

        self.dir_inputs = QGroupBox("Directory Input")
        dir_form = QFormLayout(self.dir_inputs)
        dir_input_row, self.input_dir_edit = _make_path_row("Browse", self._browse_input_dir)
        dir_output_row, self.output_dir_edit = _make_path_row("Browse", self._browse_output_dir)
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
        layout.addWidget(self.more_options)
        self.run_button = QPushButton("Run Digest")
        self.run_button.setMinimumHeight(44)
        layout.addWidget(self.run_button)
        layout.addStretch(1)

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
            "",
            "FASTA files (*.fa *.faa *.fasta *.fna);;All files (*.*)",
        )
        if path:
            self.input_file_edit.setText(path)
            if not self.output_file_edit.text().strip():
                self.output_file_edit.setText(str(Path(path).with_suffix(".tsv")))

    def _browse_output_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output TSV file",
            "",
            "TSV files (*.tsv);;All files (*.*)",
        )
        if path:
            self.output_file_edit.setText(path)

    def _browse_input_dir(self) -> None:
        path = _choose_directory(self, "Select FASTA directory", self.input_dir_edit.text().strip())
        if path:
            self.input_dir_edit.setText(path)

    def _browse_output_dir(self) -> None:
        path = _choose_directory(self, "Select output directory", self.output_dir_edit.text().strip())
        if path:
            self.output_dir_edit.setText(path)

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
            else:
                if not config.input_dir:
                    raise ValueError("Please choose an input FASTA directory.")
                if not config.output_dir:
                    raise ValueError("Please choose an output TSV directory.")
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
        self._sync_mode_visibility()


class ScoringTab(QWidget):
    def __init__(self):
        super().__init__()
        outer_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        required_box = QGroupBox("Required")
        required_form = QFormLayout(required_box)
        peptide_row, self.peptide_table_edit = _make_path_row("Browse", self._browse_peptide_table)
        required_form.addRow("Observed peptide table", peptide_row)
        output_row, self.output_tsv_edit = _make_path_row("Browse", self._browse_output_tsv)
        required_form.addRow("Output result TSV", output_row)

        self.peptide_seq_col_edit = QLineEdit("Sequence")
        self.peptide_score_col_edit = QLineEdit("score")
        required_form.addRow("Sequence column", self.peptide_seq_col_edit)
        required_form.addRow("Score column", self.peptide_score_col_edit)

        genome_box = QGroupBox("Genome Digest Directories")
        genome_layout = QVBoxLayout(genome_box)
        self.genome_dir_list = QListWidget()
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
        layout.addWidget(required_box)

        self.more_options = CollapsibleOptions()
        options_layout = QVBoxLayout(self.more_options.body)

        columns_box = QGroupBox("Peptide Table Columns")
        columns_form = QFormLayout(columns_box)
        self.peptide_error_col_edit = QLineEdit("Q.Value")
        self.peptide_decoy_flag_col_edit = QLineEdit("Reverse")
        self.decoy_flag_value_edit = QLineEdit("+")
        columns_form.addRow("Error column", self.peptide_error_col_edit)
        columns_form.addRow("Decoy flag column", self.peptide_decoy_flag_col_edit)
        columns_form.addRow("Decoy flag value", self.decoy_flag_value_edit)
        options_layout.addWidget(columns_box)

        runtime_box = QGroupBox("Runtime And Knockoff Settings")
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
        cache_row, self.cache_path_edit = _make_path_row("Browse", self._browse_cache_path)
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
        options_layout.addWidget(runtime_box)

        exclude_box = QGroupBox("Excluded Genome IDs")
        exclude_layout = QVBoxLayout(exclude_box)
        self.exclude_text = QTextEdit()
        self.exclude_text.setPlaceholderText("One genome ID per line, or comma-separated.")
        exclude_layout.addWidget(self.exclude_text)
        options_layout.addWidget(exclude_box)

        flags_box = QGroupBox("Flags")
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
        self.run_button = QPushButton("Run Genome Presence Scoring")
        self.run_button.setMinimumHeight(44)
        layout.addWidget(self.run_button)
        layout.addStretch(1)

    def _browse_peptide_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select peptide table",
            "",
            "TSV files (*.tsv *.txt);;All files (*.*)",
        )
        if path:
            self.peptide_table_edit.setText(path)
            if not self.output_tsv_edit.text().strip():
                self.output_tsv_edit.setText(str(Path(path).with_name("genome_presence.tsv")))
            if not self.cache_path_edit.text().strip():
                self.cache_path_edit.setText(str(Path(path).with_name("matched_peptides.pkl")))

    def _browse_output_tsv(self) -> None:
        current_value = self.output_tsv_edit.text().strip()
        initial_path = current_value or "genome_presence.tsv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output result TSV",
            initial_path,
            "TSV files (*.tsv);;All files (*.*)",
        )
        if path:
            self.output_tsv_edit.setText(path)

    def _browse_cache_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select matched peptide cache",
            "",
            "Pickle files (*.pkl);;All files (*.*)",
        )
        if path:
            self.cache_path_edit.setText(path)

    def _add_genome_dir(self) -> None:
        path = _choose_directory(self, "Select digested genome directory")
        if path:
            existing = {self.genome_dir_list.item(i).text() for i in range(self.genome_dir_list.count())}
            if path not in existing:
                self.genome_dir_list.addItem(path)

    def _remove_genome_dir(self) -> None:
        for item in self.genome_dir_list.selectedItems():
            row = self.genome_dir_list.row(item)
            self.genome_dir_list.takeItem(row)

    def _clear_genome_dirs(self) -> None:
        self.genome_dir_list.clear()

    def build_config(self, require_required_fields: bool = True) -> ScoringConfig:
        config = ScoringConfig(
            peptide_table_path=self.peptide_table_edit.text().strip(),
            genome_digest_dirs=[self.genome_dir_list.item(i).text() for i in range(self.genome_dir_list.count())],
            output_tsv_path=self.output_tsv_edit.text().strip(),
            peptide_seq_col=self.peptide_seq_col_edit.text().strip(),
            peptide_score_col=self.peptide_score_col_edit.text().strip(),
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
            if not config.output_tsv_path:
                raise ValueError("Please choose an output result TSV file.")
            if not config.peptide_seq_col:
                raise ValueError("Please provide the sequence column name.")
            if not config.peptide_score_col:
                raise ValueError("Please provide the score column name.")
            if not config.genome_digest_dirs:
                raise ValueError("Please add at least one genome digest directory.")
        return config

    def load_config(self, config: ScoringConfig) -> None:
        self.peptide_table_edit.setText(config.peptide_table_path)
        self.output_tsv_edit.setText(config.output_tsv_path)
        self.peptide_seq_col_edit.setText(config.peptide_seq_col)
        self.peptide_score_col_edit.setText(config.peptide_score_col)
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
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load Config")
        self.save_button = QPushButton("Save Config")
        self.clear_log_button = QPushButton("Clear Log")
        self.status_label = QLabel("Status: Idle")
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.clear_log_button)
        button_row.addStretch(1)
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
        log_layout.addWidget(self.log_output)

        splitter.addWidget(self.tabs)
        splitter.addWidget(log_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self.load_button.clicked.connect(self._load_config)
        self.save_button.clicked.connect(self._save_config)
        self.clear_log_button.clicked.connect(self.log_output.clear)
        self.digest_tab.run_button.clicked.connect(self._run_digest_tab)
        self.scoring_tab.run_button.clicked.connect(self._run_scoring_tab)

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def _run_current_tab(self) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "Task Running", "A task is already running. Please wait for it to finish.")
            return

        try:
            if self.tabs.currentIndex() == 0:
                task_name = "digest"
                config = self.digest_tab.build_config(require_required_fields=True)
                self.status_label.setText("Status: Running digest workflow...")
            else:
                task_name = "scoring"
                config = self.scoring_tab.build_config(require_required_fields=True)
                self.status_label.setText("Status: Running scoring workflow...")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return

        self._append_log("")
        self._append_log(f"=== Starting {task_name} task ===")
        self._set_run_buttons_enabled(False)

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
        self.status_label.setText("Status: Idle")
        summary = self._format_summary(task_name, payload)
        self._append_log(summary)
        QMessageBox.information(self, "Task Complete", summary)

    @Slot(str, str)
    def _on_task_failed(self, task_name: str, error_text: str) -> None:
        self.status_label.setText("Status: Failed")
        self._append_log(error_text)
        QMessageBox.critical(self, "Task Failed", f"{task_name} failed. See the log for details.")

    @Slot()
    def _cleanup_worker(self) -> None:
        self._set_run_buttons_enabled(True)
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

        path, _ = QFileDialog.getSaveFileName(self, "Save GUI config", "", "JSON files (*.json);;All files (*.*)")
        if not path:
            return

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.status_label.setText(f"Status: Saved config: {path}")

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load GUI config", "", "JSON files (*.json);;All files (*.*)")
        if not path:
            return

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.digest_tab.load_config(DigestConfig(**payload.get("digest", {})))
        self.scoring_tab.load_config(ScoringConfig(**payload.get("scoring", {})))
        selected_tab = int(payload.get("selected_tab", 0))
        if selected_tab in (0, 1):
            self.tabs.setCurrentIndex(selected_tab)
        self.status_label.setText(f"Status: Loaded config: {path}")

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
