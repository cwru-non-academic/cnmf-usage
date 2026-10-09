import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from cnmf_usage.validation import (
    InputValidationError,
    InputValidationWarning,
    validate_input,
)

GENES = ["G1", "G2", "G3"]

RAW = np.array(
    [
        [1, 2, 3],
        [10, 0, 1],
        [5, 5, 5],
        [50, 1, 0],
    ],
    dtype=np.float64,
)


def make_adata(
    counts,
    *,
    genes=None,
    x=None,
):
    """Create a small AnnData with counts in layers['counts']."""

    counts = sparse.csr_matrix(counts)

    if x is None:
        x = counts.copy()

    obj = ad.AnnData(X=sparse.csr_matrix(x))

    obj.obs_names = [f"cell_{i}" for i in range(obj.n_obs)]

    obj.var_names = GENES[: obj.n_vars] if genes is None else genes

    obj.layers["counts"] = counts

    return obj


def validate(obj, reference_genes=None):
    """Run validation using the test counts layer."""

    return validate_input(
        obj,
        GENES if reference_genes is None else reference_genes,
        counts_layer="counts",
    )


def test_counts_layer_argument_is_required():
    obj = make_adata(RAW)

    with pytest.raises(TypeError):
        validate_input(obj, GENES)


def test_only_requested_counts_layer_is_inspected():
    # Bad data elsewhere in AnnData must not affect valid counts.
    transformed = np.log1p(RAW)

    obj = make_adata(
        RAW,
        x=transformed,
    )

    obj.layers["normalized"] = sparse.csr_matrix(transformed)

    obj.layers["irrelevant_negative_layer"] = sparse.csr_matrix(-np.ones_like(RAW))

    result = validate(obj)

    assert result.transform_signature == "count_like"


def test_raw_attribute_is_ignored():
    # .raw may contain transformed data; only counts_layer matters.
    obj = make_adata(RAW)

    raw_obj = ad.AnnData(X=sparse.csr_matrix(np.log1p(RAW)))
    raw_obj.obs_names = obj.obs_names.copy()
    raw_obj.var_names = obj.var_names.copy()

    obj.raw = raw_obj

    result = validate(obj)

    assert result.transform_signature == "count_like"


def test_missing_counts_layer_errors():
    obj = make_adata(RAW)
    del obj.layers["counts"]

    with pytest.raises(
        InputValidationError,
        match="was not found",
    ):
        validate(obj)


def test_duplicate_cell_ids_error():
    obj = make_adata(RAW)

    obj.obs_names = [
        "cell",
        "cell",
        "cell_2",
        "cell_3",
    ]

    with pytest.raises(
        InputValidationError,
        match="duplicate cell IDs",
    ):
        validate(obj)


def test_duplicate_gene_ids_error():
    obj = make_adata(
        RAW,
        genes=["G1", "G1", "G3"],
    )

    with pytest.raises(
        InputValidationError,
        match="duplicate gene IDs",
    ):
        validate(obj)


def test_negative_values_error():
    bad = RAW.copy()
    bad[0, 0] = -1

    with pytest.raises(
        InputValidationError,
        match="negative values",
    ):
        validate(make_adata(bad))


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf],
)
def test_nonfinite_values_error(bad_value):
    bad = RAW.copy()
    bad[0, 0] = bad_value

    with pytest.raises(
        InputValidationError,
        match="NaN or Inf",
    ):
        validate(make_adata(bad))


def test_zero_exact_overlap_errors():
    obj = make_adata(RAW)

    with pytest.raises(
        InputValidationError,
        match="Exact gene overlap.*0",
    ):
        validate(
            obj,
            ["OTHER1", "OTHER2"],
        )


def normalized_counts():
    """Return fixed-library-size values, including one zero cell."""

    return np.array(
        [
            [2500.5, 7499.5],
            [5000.25, 4999.75],
            [1000.5, 8999.5],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    ("matrix", "signature"),
    [
        (
            normalized_counts(),
            "normalized",
        ),
        (
            np.log1p(normalized_counts()),
            "log1p_normalized",
        ),
        (
            np.log1p(RAW[:, :2]),
            "log1p_raw",
        ),
    ],
)
def test_known_transformed_inputs_error(
    matrix,
    signature,
):
    genes = ["G1", "G2"]

    obj = make_adata(
        matrix,
        genes=genes,
    )

    with pytest.raises(
        InputValidationError,
        match=signature,
    ):
        validate(
            obj,
            genes,
        )


def test_log2_normalized_warns_and_continues():
    genes = ["G1", "G2"]

    matrix = np.log2(normalized_counts() + 1.0)

    obj = make_adata(
        matrix,
        genes=genes,
    )

    with pytest.warns(
        InputValidationWarning,
        match="log2-normalized",
    ):
        result = validate(
            obj,
            genes,
        )

    assert result.transform_signature == "log2_normalized"


def test_fractional_corrected_counts_pass():
    corrected = RAW * 0.87
    obj = make_adata(corrected)

    result = validate(obj)

    assert result.transform_signature == "count_like"


def test_case_insensitive_overlap_warns_and_continues():
    genes = ["CD74", "LYZ", "G3"]

    obj = make_adata(
        RAW,
        genes=genes,
    )

    reference = [
        "CD74",
        "LYZ",
        "g3",
    ]

    with pytest.warns(
        InputValidationWarning,
        match="Case-insensitive overlap",
    ):
        result = validate(
            obj,
            reference,
        )

    assert result.exact_gene_overlap == 2
    assert result.case_insensitive_gene_overlap == 3


def test_zero_total_cells_are_ignored_for_cv():
    genes = ["G1", "G2"]

    obj = make_adata(
        normalized_counts(),
        genes=genes,
    )

    with pytest.raises(
        InputValidationError,
        match="normalized",
    ):
        validate(
            obj,
            genes,
        )


def test_stored_sparse_zeros_do_not_affect_detection():
    rows = 3
    cols = 20

    data = []
    indices = []
    indptr = [0]

    pairs = [
        (2500.5, 7499.5),
        (5000.25, 4999.75),
        (1000.5, 8999.5),
    ]

    for first, second in pairs:
        data.extend([first, second] + [0.0] * (cols - 2))
        indices.extend(range(cols))
        indptr.append(len(data))

    matrix = sparse.csr_matrix(
        (
            np.asarray(data),
            np.asarray(indices),
            np.asarray(indptr),
        ),
        shape=(rows, cols),
    )

    genes = [f"G{i}" for i in range(cols)]

    obj = ad.AnnData(X=matrix.copy())

    obj.obs_names = [f"cell_{i}" for i in range(rows)]

    obj.var_names = genes
    obj.layers["counts"] = matrix

    with pytest.raises(
        InputValidationError,
        match="normalized",
    ):
        validate(
            obj,
            genes,
        )
