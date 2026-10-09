from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import pandas as pd

_PACKAGE = "cnmf_usage"
_MANIFEST = "data/reference_manifest.json"
_REFERENCE_DIR = "data/references"


@dataclass(frozen=True)
class Reference:
    id: str
    species: str
    cell_scope: str
    intended_input: str
    filename: str
    orientation: str
    reference_version: str
    sha256: str
    citation: str | None
    notes: str
    spectra: pd.DataFrame


def _resource(path: str):
    resource = files(_PACKAGE)

    for part in path.split("/"):
        resource = resource.joinpath(part)

    return resource


def _read_manifest() -> dict[str, Any]:
    with _resource(_MANIFEST).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_reference_bytes(filename: str) -> bytes:
    path = f"{_REFERENCE_DIR}/{filename}"

    with _resource(path).open("rb") as handle:
        return handle.read()


def available_references() -> tuple[str, ...]:
    manifest = _read_manifest()

    return tuple(reference["id"] for reference in manifest["references"])


def load_reference(reference_id: str) -> Reference:
    manifest = _read_manifest()

    entries = {reference["id"]: reference for reference in manifest["references"]}

    if reference_id not in entries:
        available = ", ".join(entries)

        raise ValueError(
            f"Unknown reference '{reference_id}'. Available references: {available}."
        )

    metadata = entries[reference_id]
    raw = _read_reference_bytes(metadata["filename"])

    actual_sha256 = hashlib.sha256(raw).hexdigest()
    expected_sha256 = metadata["sha256"]

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Bundled reference '{reference_id}' failed its SHA-256 "
            "integrity check. "
            f"Expected {expected_sha256}, found {actual_sha256}. "
            "The package reference has changed or is corrupted; "
            "the analysis stopped."
        )

    spectra = pd.read_table(
        io.BytesIO(raw),
        sep="\t",
        header=0,
        index_col=0,
    )

    return Reference(
        id=metadata["id"],
        species=metadata["species"],
        cell_scope=metadata["cell_scope"],
        intended_input=metadata["intended_input"],
        filename=metadata["filename"],
        orientation=metadata["orientation"],
        reference_version=metadata["reference_version"],
        sha256=metadata["sha256"],
        citation=metadata["citation"],
        notes=metadata["notes"],
        spectra=spectra,
    )
