import pytest

import cnmf_usage.references as reference_module
from cnmf_usage.references import (
    available_references,
    load_reference,
)

EXPECTED = {
    "human_glioma_myeloid_k14": {
        "shape": (2278, 14),
        "programs": [
            "Microglia",
            "Complement_Immunosuppressive",
            "Neutrophil",
            "Scavenger_Immunosuppressive",
            "G2_M",
            "IFN_Response",
            "IL1B_CD83_TNF_Inflammatory",
            "cDC",
            "G1_S",
            "Hypoxia",
            "Inflammatory_microglia",
            "HS_UPR",
            "Macrophage",
            "Monocyte",
        ],
    },
    "mouse_glioma_myeloid_k14_ortholog": {
        "shape": (1897, 14),
        "programs": [
            "Microglia",
            "Complement_Immunosuppressive",
            "Neutrophil",
            "Scavenger_Immunosuppressive",
            "G2_M",
            "IFN",
            "Systemic_Inflammatory",
            "cDC",
            "G1_S",
            "Hypoxia",
            "Microglia_Inflammatory",
            "HS_UPR",
            "Macrophage",
            "Monocyte",
        ],
    },
    "human_adult_glioma_allcell_k18": {
        "shape": (2999, 18),
        "programs": [
            "T-cell",
            "Cancer-1",
            "Cancer-2",
            "Myeloid-1",
            "Cancer-3",
            "Cancer-4",
            "Oligo",
            "Cancer-5",
            "Myeloid-2",
            "Myeloid-3",
            "Myeloid-4",
            "Cancer-6",
            "GiantCellCancer",
            "Cycling",
            "Vasculature-2",
            "OtherImmune_1",
            "Vasculature-1",
            "OtherImmune_2",
        ],
    },
}


def test_available_references():
    assert available_references() == tuple(EXPECTED)


@pytest.mark.parametrize("reference_id", EXPECTED)
def test_reference_loads_with_exact_shape_and_program_order(reference_id):
    expected = EXPECTED[reference_id]
    reference = load_reference(reference_id)

    assert reference.id == reference_id
    assert reference.orientation == "genes_x_programs"
    assert reference.reference_version == "1.0"
    assert reference.spectra.shape == expected["shape"]
    assert list(reference.spectra.columns) == expected["programs"]


def test_unknown_reference_fails():
    with pytest.raises(ValueError, match="Unknown reference"):
        load_reference("not_a_reference")


def test_reference_hash_mismatch_stops(monkeypatch):
    monkeypatch.setattr(
        reference_module,
        "_read_reference_bytes",
        lambda filename: b"corrupted reference",
    )

    with pytest.raises(RuntimeError, match="SHA-256"):
        load_reference("human_glioma_myeloid_k14")
