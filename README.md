<h1 align="center">cnmf-usage</h1>

`cnmf-usage` back-calculates per-cell cNMF program usage from fixed reference spectra and an AnnData count matrix.

## Scope

The bundled human reference programs were derived from single-cell RNA-seq of human adult diffuse gliomas. The source study included IDH-mutant and IDH-wild-type tumors, primary and recurrent tumors, and both low- and high-grade gliomas.
The references should be used when the species, cell population, and biological context of the input data are appropriate for the selected reference. They are not intended as general-purpose programs for unrelated tumor types.
The mouse K14 reference is an ortholog-mapped version of the human glioma myeloid K14 reference for use with mouse glioma myeloid data.

## Available references

| Reference | Species | Programs | Intended input |
| --- | --- | ---: | --- |
| `human_glioma_myeloid_k14` | Human | 14 | Adult diffuse glioma myeloid cells |
| `mouse_glioma_myeloid_k14_ortholog` | Mouse | 14 | Mouse glioma myeloid cells |
| `human_adult_glioma_allcell_k18` | Human | 18 | Adult diffuse glioma all-cell data |

Reference choice should match the species, cell population, and biological context of the dataset.
Program names are defined by each reference and are not necessarily identical between the human and mouse K14 references.

## Installation

Requires Python 3.11. Python 3.12 and later are not supported yet.

Clone a release:

```bash
git clone --branch v0.1.0 https://github.com/cwru-non-academic/cnmf-usage.git
cd cnmf-usage
```

### Option 1: uv (recommended)

Installs the exact validated environment from `uv.lock`. uv downloads Python 3.11 if needed.
```bash
uv sync --no-dev
uv run cnmf-usage --version
```
> `uv run` works inside the repository folder. To use `cnmf-usage` from any folder, activate the environment first: `source .venv/bin/activate`.

### Option 2: pip
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
cnmf-usage --version
```
> With pip, the main dependency versions match the validated environment, but other dependency versions can differ.
> Conda users: Create an environment with `python=3.11`, activate it, then run `python -m pip install .`.

## Input

Input is an AnnData `.h5ad` file with raw or corrected counts stored in a named layer.
For example:
```python
adata.layers["counts"]
```

Only the layer specified by `counts_layer` is used for back-calculation. `.X`, `.raw`, and other layers are not used as substitutes.
Normalized or log-transformed expression must not be supplied as the count layer. Input validation stops the run if the selected layer appears normalized or log-transformed, or if fewer than two cells are supplied.
`adata.var_names` must contain gene symbols that match the selected reference. Matching is exact and case-sensitive. Human references require human gene symbols, and the mouse reference requires mouse gene symbols.
Gene identifiers are not converted automatically. For example, Ensembl IDs are not converted to gene symbols.
A run with no shared genes fails validation.
All supplied cells are scaled and fitted together. Do not split cells that you want to compare into separate runs.

## Python usage

```python
import scanpy as sc

from cnmf_usage import add_usage_to_obs, calculate_usage

adata = sc.read_h5ad("sample.h5ad")

result = calculate_usage(
    adata,
    reference="human_glioma_myeloid_k14",
    counts_layer="counts",
)

print(result.usage.head())
print(result.cell_qc.head())

print("Shared genes:", result.run_qc["n_shared_genes"])
print("NMF converged:", result.run_qc["nmf_converged"])

add_usage_to_obs(adata, result.usage)
```

`result.usage` contains per-cell program usage percentages.
For cells with non-zero fitted usage, each row sums to 100%. Cells with zero total fitted usage remain `NaN` and are flagged by `zero_usage_cell` in the per-cell QC output.
`add_usage_to_obs` explicitly adds the usage percentages to `adata.obs`.

## Command line

```bash
cnmf-usage run \
  --input sample.h5ad \
  --reference human_glioma_myeloid_k14 \
  --counts-layer counts \
  --output-dir cnmf_usage_output
```

Run:

```bash
cnmf-usage run --help
```

to see all available options.
The command stops if its result files already exist in the output directory. Use `--overwrite` to replace them.

The command writes:

- `usage_percent.tsv` — per-cell program usage percentages.
- `raw_w.tsv` — fitted non-negative program weights before percentage normalization.
- `cell_qc.tsv` — per-cell reference-gene and fit QC.
- `run_qc.json` — reference information, software versions, gene coverage, NMF convergence information, and run-level QC.

## Method

The package projects new expression data onto fixed cNMF programs:

```text
X ≈ W × H
```

where:

- `X` is the input expression matrix after fixed preprocessing.
- `H` is the fixed reference program matrix.
- `W` is the estimated program usage matrix.

The package does not learn new cNMF programs.
The workflow:

1. Matches shared genes using exact, case-sensitive gene symbols and preserves input gene order.
2. Subsets the count matrix to shared genes.
3. Scales each shared gene to unit variance across all supplied cells without centering.
4. Fits non-negative matrix factorization with the reference matrix `H` fixed.
5. Converts each cell's fitted program weights in `W` to percentages.

The package does not perform library-size normalization, log transformation, highly variable gene selection, batch correction, gene-name conversion, or missing-gene imputation.
Gene matching, scaling, NMF settings, and usage normalization are fixed by the package to preserve reproducibility. Bundled reference files are verified against stored SHA-256 hashes when they are loaded to detect accidental modification of the reference spectra.

## Interpretation and QC

Program usage is relative within each cell.
Because scaling and fitting use all supplied cells together, usage estimates can change when the set of cells in a run changes. Cells that should be compared should therefore be analyzed in the same run.

Useful QC fields include:

- `n_shared_genes` — number of genes shared between the input and reference.
- `program_mass_retained` — fraction of each reference program's total weight represented by the shared genes. A low value indicates that a larger part of that reference program is represented by genes absent from the input.
- `nmf_converged` — whether the fixed-H NMF fit converged within the allowed iterations.
- `relative_fit_error` — per-cell reconstruction error; larger values indicate a poorer fit to the reference programs.
- `zero_usage_cell` — flags cells for which the fitted program weights sum to zero.

## Tutorial

See the [single-sample tutorial](examples/tutorial.ipynb) for an end-to-end example.

## License

The software source code is licensed under the [MIT License](LICENSE).
The bundled reference spectra are distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See the [reference spectra license](src/cnmf_usage/data/references/LICENSE) for attribution and modification information.

## Citation

If you use the bundled glioma reference programs, please cite:

Miller TE, El Farran CA, Couturier CP, et al.  
**Programs, origins and immunomodulatory functions of myeloid cells in glioma.**  
*Nature*. 2025;640:1072-1082.  
doi:10.1038/s41586-025-08633-8
