from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import pandas as pd

from .core import calculate_usage
from .references import available_references, load_reference
from .validation import InputValidationError

DESCRIPTION = """\
Back-calculate per-cell cNMF program usage against a bundled reference.
Input must contain raw or corrected counts, not normalized or log-transformed data.
"""


def _package_version() -> str:
    """Return the installed package version."""

    try:
        return version("cnmf-usage")
    except PackageNotFoundError:
        return "unknown"


def _reference_help() -> str:
    """Describe bundled references using packaged metadata."""

    lines = []

    for reference_id in available_references():
        reference = load_reference(reference_id)

        label = reference.intended_input

        lines.append(f"  {reference_id:<40} {label}")

    return "\n".join(lines)


def _run_epilog() -> str:
    """Return output and example help text."""

    return """\
Notes:
  All supplied cells are scaled and fitted together.
  Do not split cells that you want to compare into separate runs.

Output files:
  usage_percent.tsv   Per-cell program usage (%).
                      Non-zero rows sum to 100; zero-usage cells are NaN.
  raw_w.tsv           NMF program weights before percent normalization.
  cell_qc.tsv         Per-cell fit and reference-gene coverage QC.
  run_qc.json         Run settings, versions and summary QC.

Example:
  cnmf-usage run --input cells.h5ad --reference human_glioma_myeloid_k14 \\
    --counts-layer counts --output-dir results/

Reference selection and result interpretation: see README.
"""


def _write_table(
    table: pd.DataFrame,
    path: Path,
) -> None:
    """Write a TSV with full float precision."""

    table.to_csv(
        path,
        sep="\t",
        index=True,
        index_label="cell_id",
        na_rep="NaN",
        float_format="%.17g",
    )


def _run(args: argparse.Namespace) -> int:
    """Run back-calculation and write the result files."""

    output_files = (
        "usage_percent.tsv",
        "raw_w.tsv",
        "cell_qc.tsv",
        "run_qc.json",
    )

    existing = [
        args.output_dir / name
        for name in output_files
        if (args.output_dir / name).exists()
    ]

    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output file(s) already exist: {names}. Use --overwrite to replace them."
        )

    adata = ad.read_h5ad(args.input)

    result = calculate_usage(
        adata,
        reference=args.reference,
        counts_layer=args.counts_layer,
    )

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_table(
        result.usage,
        output_dir / "usage_percent.tsv",
    )

    _write_table(
        result.raw_w,
        output_dir / "raw_w.tsv",
    )

    _write_table(
        result.cell_qc,
        output_dir / "cell_qc.tsv",
    )

    with (output_dir / "run_qc.json").open("w") as handle:
        json.dump(
            result.run_qc,
            handle,
            indent=2,
            allow_nan=False,
        )

    print(f"Cells: {result.run_qc['n_cells']}")
    print(f"Shared genes: {result.run_qc['n_shared_genes']}")
    print(f"NMF iterations: {result.run_qc['nmf_n_iter']}")
    print(f"Output: {output_dir}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    formatter = argparse.RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        prog="cnmf-usage",
        description=DESCRIPTION,
        formatter_class=formatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
        help="show version and exit",
    )

    commands = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    run_parser = commands.add_parser(
        "run",
        help="Run fixed-reference back-calculation.",
        description=DESCRIPTION,
        epilog=_run_epilog(),
        formatter_class=formatter,
    )

    input_group = run_parser.add_argument_group("Input")

    input_group.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="INPUT",
        help="Input AnnData file (.h5ad).",
    )

    input_group.add_argument(
        "--counts-layer",
        required=True,
        metavar="COUNTS_LAYER",
        help=(
            "AnnData layer containing raw or corrected counts.\n"
            ".X, .raw and other layers are ignored."
        ),
    )

    reference_group = run_parser.add_argument_group("Reference")

    reference_group.add_argument(
        "--reference",
        required=True,
        choices=available_references(),
        help=(
            "Fixed gene-program spectra used to estimate how much\n"
            "each program is used in each cell.\n\n"
            "Available references:\n"
            f"{_reference_help()}"
        ),
    )

    output_group = run_parser.add_argument_group("Output")

    output_group.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="OUTPUT_DIR",
        help="Directory for result files.",
    )

    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cnmf-usage result files.",
    )

    run_parser.set_defaults(handler=_run)

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    """Run the cnmf-usage command."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except (
        InputValidationError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:
        parser.exit(
            2,
            f"Error: {exc}\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
