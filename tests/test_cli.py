from __future__ import annotations

import json

import anndata as ad
import numpy as np
from scipy import sparse

from cnmf_usage.cli import main
from cnmf_usage.references import load_reference


def test_cli_writes_expected_outputs(tmp_path):
    """Run the CLI on a small local H5AD."""

    reference = load_reference("human_glioma_myeloid_k14")

    genes = list(reference.spectra.index[:400])

    rng = np.random.default_rng(42)

    counts = rng.poisson(
        lam=5,
        size=(8, len(genes)),
    ).astype(np.int32)

    obj = ad.AnnData(X=sparse.csr_matrix(np.log1p(counts)))

    obj.obs_names = [f"cell_{i}" for i in range(obj.n_obs)]

    obj.var_names = genes

    # Back-calculation must use only this layer.
    obj.layers["counts"] = sparse.csr_matrix(counts)

    input_path = tmp_path / "input.h5ad"

    output_dir = tmp_path / "results"

    obj.write_h5ad(input_path)

    status = main(
        [
            "run",
            "--input",
            str(input_path),
            "--reference",
            "human_glioma_myeloid_k14",
            "--counts-layer",
            "counts",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0

    expected = {
        "usage_percent.tsv",
        "raw_w.tsv",
        "cell_qc.tsv",
        "run_qc.json",
    }

    assert {path.name for path in output_dir.iterdir()} == expected

    usage_header = (output_dir / "usage_percent.tsv").read_text().splitlines()[0]

    assert usage_header.startswith("cell_id\t")

    run_qc = json.loads((output_dir / "run_qc.json").read_text())

    assert run_qc["n_cells"] == 8
    assert run_qc["counts_layer"] == "counts"
    assert run_qc["reference_id"] == "human_glioma_myeloid_k14"


def test_run_help_lists_references_and_outputs(capsys):
    """CLI help shows valid references and main outputs."""

    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])

    assert exc.value.code == 0

    output = capsys.readouterr().out

    assert "human_glioma_myeloid_k14" in output
    assert "mouse_glioma_myeloid_k14_ortholog" in output
    assert "human_adult_glioma_allcell_k18" in output
    assert "usage_percent.tsv" in output
    assert "raw_w.tsv" in output
    assert "cell_qc.tsv" in output
    assert "run_qc.json" in output
