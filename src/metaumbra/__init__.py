"""MetaUmbra package metadata."""

from .__version__ import __version__
from .scoring_unit_outputs import apply_patch as _apply_unit_output_patch

_apply_unit_output_patch()

del _apply_unit_output_patch

__all__ = ["__version__"]
