import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cnmf_usage import (
    UsageResult,
    add_usage_to_obs,
)


def make_result() -> UsageResult:
    """Create a small usage result for API tests."""

    usage = pd.DataFrame(
        {
            "Program_A": [70.0, 20.0],
            "Program_B": [30.0, 80.0],
        },
        index=pd.Index(["cell_1", "cell_2"]),
    )

    return UsageResult(
        usage=usage,
        raw_w=usage.copy(),
        cell_qc=pd.DataFrame(index=usage.index),
        run_qc={},
    )


def make_adata() -> ad.AnnData:
    """Create matching cells for API tests."""

    obj = ad.AnnData(X=np.zeros((2, 1)))

    obj.obs_names = [
        "cell_1",
        "cell_2",
    ]

    obj.var_names = ["G1"]

    return obj


def test_add_usage_to_obs():
    obj = make_adata()
    result = make_result()

    add_usage_to_obs(
        obj,
        result,
    )

    assert list(obj.obs.columns) == [
        "Program_A",
        "Program_B",
    ]

    np.testing.assert_array_equal(
        obj.obs["Program_A"],
        [70.0, 20.0],
    )


def test_add_usage_to_obs_with_prefix():
    obj = make_adata()
    result = make_result()

    add_usage_to_obs(
        obj,
        result,
        column_prefix="GLIOMA_CONSENSUS_",
    )

    assert "GLIOMA_CONSENSUS_Program_A" in obj.obs.columns


def test_add_usage_collision_errors():
    obj = make_adata()
    result = make_result()

    obj.obs["Program_A"] = 0.0

    with pytest.raises(
        ValueError,
        match="already exist",
    ):
        add_usage_to_obs(
            obj,
            result,
        )


def test_add_usage_overwrite():
    obj = make_adata()
    result = make_result()

    obj.obs["Program_A"] = 0.0

    add_usage_to_obs(
        obj,
        result,
        overwrite=True,
    )

    np.testing.assert_array_equal(
        obj.obs["Program_A"],
        [70.0, 20.0],
    )


def test_add_usage_requires_matching_cells():
    obj = make_adata()
    result = make_result()

    obj.obs_names = [
        "cell_2",
        "cell_1",
    ]

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        add_usage_to_obs(
            obj,
            result,
        )
