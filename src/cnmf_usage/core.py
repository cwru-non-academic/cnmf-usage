from __future__ import annotations

import hashlib
import platform
import warnings
from importlib.metadata import PackageNotFoundError, version

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import non_negative_factorization
from sklearn.exceptions import ConvergenceWarning

from .models import UsageResult
from .references import Reference, load_reference
from .validation import validate_input

NMF_MAX_ITER = 1000

_NMF_PARAMETERS = {
    "init": "random",
    "update_H": False,
    "solver": "cd",
    "beta_loss": "frobenius",
    "tol": 1e-4,
    "max_iter": NMF_MAX_ITER,
    "alpha_W": 0.0,
    "alpha_H": "same",
    "l1_ratio": 0.0,
    "verbose": 0,
    "shuffle": False,
}


def _package_version() -> str:
    """Return the installed package version."""

    try:
        return version("cnmf-usage")
    except PackageNotFoundError:
        return "unknown"


def _ordered_sha256(values) -> str:
    """Hash an ordered list without losing item boundaries."""

    digest = hashlib.sha256()

    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    return digest.hexdigest()


def _shared_genes(
    adata: ad.AnnData,
    reference: Reference,
) -> list[str]:
    """Return shared genes in input gene order."""

    reference_genes = set(reference.spectra.index)

    return [str(gene) for gene in adata.var_names if str(gene) in reference_genes]


def _to_csr_copy(matrix) -> sparse.csr_matrix:
    """Return an independent CSR matrix."""

    if sparse.issparse(matrix):
        return matrix.tocsr(copy=True)

    return sparse.csr_matrix(np.asarray(matrix))


def _prepare_nmf_inputs(
    adata: ad.AnnData,
    reference: Reference,
    counts_layer: str,
    shared_genes: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Prepare raw shared counts, scaled X, and fixed H."""

    positions = adata.var_names.get_indexer(shared_genes)

    x_counts = _to_csr_copy(adata.layers[counts_layer][:, positions])

    # Scale factors use all supplied cells; do not scale in chunks.
    # Keep the input dtype until scaling is complete.
    x_scaled = sc.pp.scale(
        x_counts.copy(),
        zero_center=False,
    )

    if not sparse.issparse(x_scaled):
        x_scaled = sparse.csr_matrix(x_scaled)

    x_scaled = x_scaled.tocsr().astype(
        np.float64,
        copy=False,
    )

    # H follows the same input-gene order as X.
    h = reference.spectra.loc[shared_genes].T.to_numpy(dtype=np.float64)

    return x_counts, x_scaled, h


def _fit_fixed_h(
    x: sparse.csr_matrix,
    h: np.ndarray,
) -> tuple[np.ndarray, int, list[str]]:
    """Fit W while keeping the reference programs H fixed."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        # Fit all cells in one call. W initialization and the stopping rule
        # use the whole matrix, so chunking changes the result.
        w, _, n_iter = non_negative_factorization(
            x,
            W=None,
            H=h,
            n_components=h.shape[0],
            random_state=None,
            **_NMF_PARAMETERS,
        )

    warning_text = [
        str(item.message)
        for item in caught
        if issubclass(
            item.category,
            ConvergenceWarning,
        )
    ]

    return w, int(n_iter), warning_text


def _normalize_usage(w: np.ndarray) -> np.ndarray:
    """Convert raw W to per-cell percentages."""

    totals = w.sum(
        axis=1,
        keepdims=True,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        return w / totals * 100.0


def _relative_fit_error(
    x: sparse.csr_matrix,
    w: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Calculate per-cell relative reconstruction error."""

    x_norm_sq = np.asarray(x.multiply(x).sum(axis=1)).ravel()

    x_h = np.asarray(x @ h.T)

    h_h = h @ h.T

    model_norm_sq = np.einsum(
        "ij,jk,ik->i",
        w,
        h_h,
        w,
        optimize=True,
    )

    cross_term = np.einsum(
        "ij,ij->i",
        w,
        x_h,
        optimize=True,
    )

    residual_sq = x_norm_sq + model_norm_sq - 2.0 * cross_term

    # Negative values here can only come from floating-point error.
    residual_sq = np.maximum(
        residual_sq,
        0.0,
    )

    denominator = np.sqrt(x_norm_sq)

    result = np.full(
        x.shape[0],
        np.nan,
        dtype=np.float64,
    )

    valid = denominator > 0

    result[valid] = np.sqrt(residual_sq[valid]) / denominator[valid]

    return result


def _cell_qc(
    cell_ids,
    x_counts: sparse.csr_matrix,
    x_scaled: sparse.csr_matrix,
    w: np.ndarray,
    h: np.ndarray,
) -> pd.DataFrame:
    """Build per-cell fit and reference-coverage metrics."""

    genes_detected = np.asarray((x_counts > 0).sum(axis=1)).ravel().astype(np.int64)

    counts_sum = np.asarray(x_counts.sum(axis=1)).ravel()

    raw_w_sum = w.sum(axis=1)

    return pd.DataFrame(
        {
            "reference_genes_detected": genes_detected,
            "reference_counts_sum": counts_sum,
            "raw_w_sum": raw_w_sum,
            "relative_fit_error": _relative_fit_error(
                x_scaled,
                w,
                h,
            ),
            "zero_usage_cell": raw_w_sum == 0,
        },
        index=pd.Index(
            cell_ids,
            name="cell_id",
        ),
    )


def _fit_error_summary(
    values: np.ndarray,
) -> dict[str, float | None]:
    """Summarize finite per-cell fit errors."""

    finite = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = finite[np.isfinite(finite)]

    keys = [
        "min",
        "p25",
        "p50",
        "p75",
        "p95",
        "max",
    ]

    if finite.size == 0:
        return {key: None for key in keys}

    percentiles = np.percentile(
        finite,
        [0, 25, 50, 75, 95, 100],
    )

    return {
        key: float(value)
        for key, value in zip(
            keys,
            percentiles,
        )
    }


def _run_qc(
    adata: ad.AnnData,
    reference: Reference,
    counts_layer: str,
    shared_genes: list[str],
    n_iter: int,
    warning_text: list[str],
    cell_qc: pd.DataFrame,
) -> dict:
    """Build run-level provenance and QC."""

    reference_gene_set = set(shared_genes)

    missing_genes = [
        str(gene)
        for gene in reference.spectra.index
        if str(gene) not in reference_gene_set
    ]

    full_mass = reference.spectra.sum(axis=0)

    retained_mass = reference.spectra.loc[shared_genes].sum(axis=0)

    mass_fraction = retained_mass / full_mass

    parameters = {
        **_NMF_PARAMETERS,
        "n_components": int(reference.spectra.shape[1]),
        "random_state": None,
    }

    return {
        "package_version": _package_version(),
        "python_version": platform.python_version(),
        "dependency_versions": {
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "scikit_learn": version("scikit-learn"),
            "scanpy": version("scanpy"),
            "anndata": version("anndata"),
            "pandas": version("pandas"),
        },
        "reference_id": reference.id,
        "reference_filename": reference.filename,
        "reference_sha256": reference.sha256,
        "reference_version": reference.reference_version,
        "counts_layer": counts_layer,
        "input_dtype": str(adata.layers[counts_layer].dtype),
        "n_cells": int(adata.n_obs),
        "n_input_genes": int(adata.n_vars),
        "n_reference_genes": int(reference.spectra.shape[0]),
        "n_shared_genes": len(shared_genes),
        "gene_coverage_fraction": (len(shared_genes) / reference.spectra.shape[0]),
        "missing_reference_genes": missing_genes,
        "program_mass_retained": {
            program: float(mass_fraction[program])
            for program in reference.spectra.columns
        },
        "nmf_n_iter": n_iter,
        "nmf_max_iter": NMF_MAX_ITER,
        "nmf_converged": (n_iter < NMF_MAX_ITER and not warning_text),
        "nmf_convergence_warning": bool(warning_text),
        "nmf_warning_text": warning_text,
        "zero_usage_cells": int(cell_qc["zero_usage_cell"].sum()),
        "relative_fit_error": (
            _fit_error_summary(cell_qc["relative_fit_error"].to_numpy())
        ),
        "ordered_cell_ids_sha256": (_ordered_sha256(adata.obs_names)),
        "ordered_shared_genes_sha256": (_ordered_sha256(shared_genes)),
        "nmf_parameters": parameters,
    }


def calculate_usage(
    adata: ad.AnnData,
    *,
    reference: str,
    counts_layer: str,
) -> UsageResult:
    """Back-calculate per-cell usage of a fixed cNMF reference."""

    ref = load_reference(reference)

    validate_input(
        adata,
        ref.spectra.index,
        counts_layer=counts_layer,
    )

    shared_genes = _shared_genes(
        adata,
        ref,
    )

    x_counts, x_scaled, h = _prepare_nmf_inputs(
        adata,
        ref,
        counts_layer,
        shared_genes,
    )

    w, n_iter, warning_text = _fit_fixed_h(x_scaled, h)

    usage_values = _normalize_usage(w)

    columns = list(ref.spectra.columns)

    raw_w = pd.DataFrame(
        w,
        index=adata.obs_names.copy(),
        columns=columns,
    )

    usage = pd.DataFrame(
        usage_values,
        index=adata.obs_names.copy(),
        columns=columns,
    )

    cell_qc = _cell_qc(
        adata.obs_names,
        x_counts,
        x_scaled,
        w,
        h,
    )

    run_qc = _run_qc(
        adata,
        ref,
        counts_layer,
        shared_genes,
        n_iter,
        warning_text,
        cell_qc,
    )

    return UsageResult(
        usage=usage,
        raw_w=raw_w,
        cell_qc=cell_qc,
        run_qc=run_qc,
    )


def add_usage_to_obs(
    adata: ad.AnnData,
    result: UsageResult,
    *,
    column_prefix: str = "",
    overwrite: bool = False,
) -> None:
    """Add calculated program usages to adata.obs."""

    if not adata.obs_names.equals(result.usage.index):
        raise ValueError(
            "Usage cell IDs or order do not match the AnnData object. "
            "No columns were added."
        )

    output_columns = [f"{column_prefix}{program}" for program in result.usage.columns]

    collisions = [column for column in output_columns if column in adata.obs.columns]

    if collisions and not overwrite:
        names = ", ".join(collisions)

        raise ValueError(
            f"Usage columns already exist in adata.obs: {names}. "
            "Use overwrite=True to replace them."
        )

    for program, output_column in zip(
        result.usage.columns,
        output_columns,
    ):
        adata.obs[output_column] = result.usage[program].to_numpy()
