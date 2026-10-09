from __future__ import annotations

import anndata as ad
import numpy as np
from scipy import sparse

from cnmf_usage import calculate_usage
from cnmf_usage.references import load_reference

REFERENCE_ID = "human_glioma_myeloid_k14"


def make_adata(
    *,
    n_cells: int = 40,
    n_genes: int = 250,
    dtype=np.float64,
    dense_counts: bool = False,
) -> ad.AnnData:
    """Create reproducible synthetic counts using reference gene names."""

    reference = load_reference(REFERENCE_ID)
    genes = list(reference.spectra.index[:n_genes])

    rng = np.random.default_rng(42)

    # Vary library sizes so the matrix looks like raw counts.
    cell_scale = np.linspace(
        1.0,
        6.0,
        n_cells,
    )[:, None]

    gene_scale = np.linspace(
        0.5,
        1.5,
        n_genes,
    )[None, :]

    counts = rng.poisson(cell_scale * gene_scale).astype(dtype)

    if dense_counts:
        counts_layer = counts
    else:
        counts_layer = sparse.csr_matrix(counts)

    # .X is intentionally different and must be ignored.
    obj = ad.AnnData(X=sparse.csr_matrix(np.log1p(counts.astype(np.float64))))

    obj.obs_names = [f"cell_{i}" for i in range(n_cells)]

    obj.var_names = genes
    obj.layers["counts"] = counts_layer

    return obj


def assert_matrix_equal(left, right):
    """Compare dense or sparse matrices exactly."""

    if sparse.issparse(left):
        left = left.toarray()

    if sparse.issparse(right):
        right = right.toarray()

    np.testing.assert_array_equal(
        np.asarray(left),
        np.asarray(right),
    )


def test_calculation_does_not_modify_input():
    obj = make_adata()

    x_before = obj.X.copy()
    counts_before = obj.layers["counts"].copy()

    obs_names_before = obj.obs_names.copy()

    var_names_before = obj.var_names.copy()

    calculate_usage(
        obj,
        reference=REFERENCE_ID,
        counts_layer="counts",
    )

    assert_matrix_equal(
        obj.X,
        x_before,
    )

    assert_matrix_equal(
        obj.layers["counts"],
        counts_before,
    )

    assert obj.obs_names.equals(obs_names_before)

    assert obj.var_names.equals(var_names_before)


def test_dense_and_sparse_counts_match():
    sparse_obj = make_adata(
        n_cells=30,
        n_genes=200,
    )

    dense_obj = make_adata(
        n_cells=30,
        n_genes=200,
        dense_counts=True,
    )

    sparse_result = calculate_usage(
        sparse_obj,
        reference=REFERENCE_ID,
        counts_layer="counts",
    )

    dense_result = calculate_usage(
        dense_obj,
        reference=REFERENCE_ID,
        counts_layer="counts",
    )

    np.testing.assert_allclose(
        dense_result.raw_w,
        sparse_result.raw_w,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )

    np.testing.assert_allclose(
        dense_result.usage,
        sparse_result.usage,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )


def test_integer_counts_are_supported():
    obj = make_adata(
        dtype=np.int32,
    )

    result = calculate_usage(
        obj,
        reference=REFERENCE_ID,
        counts_layer="counts",
    )

    assert result.run_qc["input_dtype"] == "int32"

    valid = ~result.usage.isna().all(axis=1)

    np.testing.assert_allclose(
        result.usage.loc[valid].sum(axis=1),
        100.0,
        atol=1e-10,
    )


def test_zero_signal_cell_returns_nan_usage():
    reference = load_reference(REFERENCE_ID)

    genes = list(reference.spectra.index[:50])

    counts = sparse.csr_matrix(
        np.vstack(
            [
                np.ones(50),
                np.zeros(50),
            ]
        )
    )

    obj = ad.AnnData(X=counts.copy())

    obj.obs_names = [
        "signal_cell",
        "zero_cell",
    ]

    obj.var_names = genes
    obj.layers["counts"] = counts

    result = calculate_usage(
        obj,
        reference=REFERENCE_ID,
        counts_layer="counts",
    )

    assert result.usage.loc["zero_cell"].isna().all()

    assert bool(
        result.cell_qc.loc[
            "zero_cell",
            "zero_usage_cell",
        ]
    )

    np.testing.assert_allclose(
        result.usage.loc["signal_cell"].sum(),
        100.0,
        atol=1e-10,
    )


def test_repeated_runs_are_deterministic():
    obj = make_adata(
        n_cells=30,
        n_genes=200,
    )

    first = calculate_usage(
        obj,
        reference=REFERENCE_ID,
        counts_layer="counts",
    )

    second = calculate_usage(
        obj,
        reference=REFERENCE_ID,
        counts_layer="counts",
    )

    np.testing.assert_allclose(
        first.raw_w,
        second.raw_w,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )

    np.testing.assert_allclose(
        first.usage,
        second.usage,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )

    assert first.run_qc["nmf_n_iter"] == second.run_qc["nmf_n_iter"]
