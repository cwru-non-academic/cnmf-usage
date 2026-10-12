from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import anndata as ad
import numpy as np
from scipy import sparse

INTEGER_ATOL = 1e-4
FIXED_TOTAL_CV = 1e-3
LOG_BACKTRANSFORM_INTEGER_FRAC = 0.98
MAX_CELLS = 2000
MAX_VALUES = 200_000

TransformSignature = Literal[
    "count_like",
    "normalized",
    "log1p_normalized",
    "log1p_raw",
    "log2_normalized",
]


class InputValidationError(ValueError):
    """Input is not valid for back-calculation."""


class InputValidationWarning(UserWarning):
    """Input may be unsuitable, but the run can continue."""


@dataclass(frozen=True)
class ValidationResult:
    """Summary of checks needed later in the run."""

    transform_signature: TransformSignature
    exact_gene_overlap: int
    case_insensitive_gene_overlap: int


def _coefficient_of_variation(values: np.ndarray) -> float | None:
    """Return CV using only cells with a positive total."""

    values = np.asarray(values, dtype=np.float64)
    values = values[values > 0]

    if values.size == 0:
        return None

    mean = float(values.mean())

    if mean == 0:
        return None

    return float(values.std() / mean)


def _sample_rows(matrix):
    """Use a fixed cell sample for large inputs."""

    if matrix.shape[0] <= MAX_CELLS:
        return matrix

    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(matrix.shape[0], MAX_CELLS, replace=False))

    return matrix[indices]


def _diagnostic_matrix(matrix) -> sparse.csr_matrix:
    """Create a sparse float64 matrix for transform checks."""

    sampled = _sample_rows(matrix)

    if sparse.issparse(sampled):
        result = sampled.tocsr(copy=True).astype(np.float64)
    else:
        result = sparse.csr_matrix(np.asarray(sampled, dtype=np.float64))

    # Stored zeros should not affect integer diagnostics.
    result.eliminate_zeros()

    return result


def _sample_nonzero_values(
    matrix: sparse.csr_matrix,
) -> np.ndarray:
    """Return a fixed sample of stored non-zero values."""

    values = np.asarray(matrix.data, dtype=np.float64)

    if values.size <= MAX_VALUES:
        return values

    rng = np.random.default_rng(0)
    indices = rng.choice(
        values.size,
        MAX_VALUES,
        replace=False,
    )

    return values[indices]


def _near_integer_fraction(
    values: np.ndarray,
) -> float | None:
    """Return the fraction of values close to integers."""

    if values.size == 0:
        return None

    return float(
        np.mean(
            np.isclose(
                values,
                np.rint(values),
                atol=INTEGER_ATOL,
                rtol=0.0,
            )
        )
    )


def _backtransform(
    matrix: sparse.csr_matrix,
    base: Literal["e", "2"],
) -> sparse.csr_matrix:
    """Reverse log1p or log2(value + 1) for diagnostics."""

    result = matrix.copy().astype(np.float64)

    if base == "e":
        result.data = np.expm1(result.data)
    else:
        result.data = np.exp2(result.data) - 1.0

    result.eliminate_zeros()

    return result


def _inspect_transform(matrix) -> dict[str, float | None]:
    """Measure signatures used to detect transformed expression."""

    matrix = _diagnostic_matrix(matrix)
    values = _sample_nonzero_values(matrix)

    integer_fraction = _near_integer_fraction(values)

    noninteger_fraction = None if integer_fraction is None else 1.0 - integer_fraction

    totals = np.asarray(matrix.sum(axis=1)).ravel()

    metrics: dict[str, float | None] = {
        "noninteger_fraction": noninteger_fraction,
        "raw_total_cv": _coefficient_of_variation(totals),
        "expm1_near_integer_fraction": None,
        "expm1_total_cv": None,
        "exp2_total_cv": None,
    }

    if values.size == 0 or float(values.max()) > 30:
        return metrics

    expm1_matrix = _backtransform(matrix, "e")
    expm1_values = _sample_nonzero_values(expm1_matrix)
    expm1_totals = np.asarray(expm1_matrix.sum(axis=1)).ravel()

    metrics["expm1_near_integer_fraction"] = _near_integer_fraction(expm1_values)
    metrics["expm1_total_cv"] = _coefficient_of_variation(expm1_totals)

    exp2_matrix = _backtransform(matrix, "2")
    exp2_totals = np.asarray(exp2_matrix.sum(axis=1)).ravel()

    metrics["exp2_total_cv"] = _coefficient_of_variation(exp2_totals)

    return metrics


def _classify_transform(
    metrics: dict[str, float | None],
) -> TransformSignature:
    """Classify only signatures already covered by validation tests."""

    noninteger = metrics["noninteger_fraction"]
    raw_cv = metrics["raw_total_cv"]
    expm1_integer = metrics["expm1_near_integer_fraction"]
    expm1_cv = metrics["expm1_total_cv"]
    exp2_cv = metrics["exp2_total_cv"]

    if (
        noninteger is not None
        and noninteger > 0.1
        and raw_cv is not None
        and raw_cv <= FIXED_TOTAL_CV
    ):
        return "normalized"

    if (
        noninteger is not None
        and noninteger > 0.1
        and expm1_cv is not None
        and expm1_cv <= FIXED_TOTAL_CV
    ):
        return "log1p_normalized"

    if (
        expm1_integer is not None
        and expm1_integer >= LOG_BACKTRANSFORM_INTEGER_FRAC
        and expm1_cv is not None
        and expm1_cv > FIXED_TOTAL_CV
    ):
        return "log1p_raw"

    if (
        noninteger is not None
        and noninteger > 0.1
        and exp2_cv is not None
        and exp2_cv <= FIXED_TOTAL_CV
    ):
        return "log2_normalized"

    return "count_like"


def _validate_values(matrix, counts_layer: str) -> None:
    """Require finite, non-negative counts."""

    values = np.asarray(matrix.data) if sparse.issparse(matrix) else np.asarray(matrix)

    if not np.isfinite(values).all():
        raise InputValidationError(
            f"Counts layer '{counts_layer}' contains NaN or Inf. "
            "Back-calculation requires finite raw or corrected counts. "
            "The analysis stopped."
        )

    if np.any(values < 0):
        raise InputValidationError(
            f"Counts layer '{counts_layer}' contains negative values. "
            "Back-calculation requires non-negative raw or corrected "
            "counts. The analysis stopped."
        )


def _gene_overlap(
    input_genes: Sequence[str],
    reference_genes: Sequence[str],
) -> tuple[int, int]:
    """Count exact and case-insensitive reference-gene matches."""

    input_exact = set(input_genes)
    input_folded = {gene.casefold() for gene in input_genes}

    exact = sum(gene in input_exact for gene in reference_genes)

    case_insensitive = sum(gene.casefold() in input_folded for gene in reference_genes)

    return int(exact), int(case_insensitive)


def validate_input(
    adata: ad.AnnData,
    reference_genes: Sequence[str],
    *,
    counts_layer: str,
) -> ValidationResult:
    """Validate the requested counts layer for back-calculation."""

    if counts_layer not in adata.layers:
        available = (
            ", ".join(str(key) for key in adata.layers if key is not None) or "none"
        )

        raise InputValidationError(
            f"Counts layer '{counts_layer}' was not found. "
            "Back-calculation requires the explicitly named layer "
            "to contain raw or corrected counts. "
            f"Available layers: {available}. "
            "The analysis stopped."
        )

    if adata.obs_names.has_duplicates:
        raise InputValidationError(
            "Input contains duplicate cell IDs. "
            "Back-calculation requires unique cell IDs. "
            "The analysis stopped."
        )

    if adata.var_names.has_duplicates:
        raise InputValidationError(
            "Input contains duplicate gene IDs. "
            "Back-calculation requires unique gene IDs for exact "
            "reference alignment. The analysis stopped."
        )

    # Only this explicitly requested layer is inspected.
    matrix = adata.layers[counts_layer]

    _validate_values(matrix, counts_layer)

    signature = _classify_transform(_inspect_transform(matrix))

    if signature in {
        "normalized",
        "log1p_normalized",
        "log1p_raw",
    }:
        raise InputValidationError(
            f"Counts layer '{counts_layer}' has a "
            f"'{signature}' expression signature. "
            "Back-calculation requires raw or corrected counts, "
            "not normalized or log-transformed expression. "
            "The analysis stopped."
        )

    if signature == "log2_normalized":
        warnings.warn(
            f"Counts layer '{counts_layer}' has a "
            "log2-normalized expression signature. "
            "Back-calculation requires raw or corrected counts. "
            "This signature is warning-only in v1; "
            "the analysis will continue.",
            InputValidationWarning,
            stacklevel=2,
        )

    input_genes = tuple(str(gene) for gene in adata.var_names)
    reference_genes = tuple(str(gene) for gene in reference_genes)

    exact_overlap, case_insensitive_overlap = _gene_overlap(
        input_genes,
        reference_genes,
    )

    if exact_overlap == 0:
        raise InputValidationError(
            "Exact gene overlap with the reference is 0. "
            f"Case-insensitive overlap is "
            f"{case_insensitive_overlap}. "
            "Back-calculation requires at least one exact "
            "case-sensitive gene match. "
            "The analysis stopped."
        )

    if case_insensitive_overlap > exact_overlap:
        warnings.warn(
            f"Exact gene overlap: {exact_overlap} of "
            f"{len(reference_genes)} reference genes. "
            f"Case-insensitive overlap: "
            f"{case_insensitive_overlap}. "
            "The input may be a different species or use a "
            "different gene-symbol style than the reference.",
            InputValidationWarning,
            stacklevel=2,
        )

    return ValidationResult(
        transform_signature=signature,
        exact_gene_overlap=exact_overlap,
        case_insensitive_gene_overlap=(case_insensitive_overlap),
    )
