# Capacity-weighted conservative thermal transfer

Minimal reproducibility code for **Capacity-weighted moment projection for
conservative thermal transfer between nonmatching discretizations**.

The method corrects a local load approximation to preserve source-discrete
total heat and independent first spatial moments. The inverse-capacity metric
minimizes the correction to that approximation. The manufactured benchmark
uses cell-integrated target loads to assess accuracy; vehicle wall fields
demonstrate transfer to a reduced thermal graph.

This initial code release is a snapshot of the existing method. It does not
claim that reviewer-requested additional analyses have already been completed.

## Contents

| Location | Purpose |
| --- | --- |
| `code/conservative_transfer.py` | Local transfer, independent moment rows and the explicit reference operator |
| `code/run_manufactured_benchmark.py` | Manufactured fields, quadrature, RBF and common-refinement comparators, thermal response and matrix-free scaling |
| `code/analyze_transfer.py` | Matrix-free vehicle wall-field transfer and thermal graph response |
| `code/analyze_gci_transfer.py` | Three-grid assessment of source and transferred quantities |
| `code/run_transfer_sweep.py` | Neighborhood and projection-metric sensitivity |
| `code/vtp_polydata.py` | ASCII VTP wall-field reader |
| `scripts/` | Statistical analysis and numerical table generation |
| `tests/test_transfer.py` | Conservation, planar rank adaptation, overlap coverage and capacity-metric optimality |
| `data/vehicle/` | Indispensable small graph and alignment inputs |
| `data/gci/grid_convergence.json` | Recorded scalar source-grid convergence inputs for the evidence analysis |
| `SOURCE_MANIFEST.json` | Source hashes and the limited portability adaptations |

The explicit operator is a reference implementation. The benchmark scaling
and vehicle application apply the projection without constructing a dense
source-to-target transfer matrix. A small constraint Gram matrix is still used.

## Installation and tests

Use Python 3.10 or newer, preferably in a separate virtual environment.
Run these commands from the repository root:

```text
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

The only Python dependencies are NumPy, SciPy and pandas. No GPU, OpenFOAM,
HPC account or TeX installation is needed to run the supplied transfer code
and manufactured benchmark.

## Manufactured benchmark

```text
python -B code/run_manufactured_benchmark.py
python -B scripts/analyze_manufactured_benchmark.py
```

The first command generates the analytical fields and executes 48 mesh-field
scenarios, seven principal methods and three graph-prior variants. It also
runs the source-size study through 1,048,576 source cells and the quadrature
audit. The second command computes paired comparisons with Holm adjustment
and writes numerical table rows. Outputs are written to
`results/manufactured_benchmark/` and `tables/`, which are not tracked in Git.
Recorded execution times depend on the machine and are not portable reference
values.

The conserved first moments use cell centers and integrated source loads.
The exact target cell integrals do not automatically satisfy the same source
discrete first moments. The projection therefore does not provide an
unconditional guarantee of smaller error against those target integrals.

## Vehicle application using the submitted data archive

Obtain `supplementary_vehicle_surface_fields.zip` from the article's
supplementary material. Extract it in the repository root so that
`vehicle_surface_fields/wall_fields/` contains the case directories. Those
large VTP files are intentionally distributed separately from this small
code repository. The original vehicle geometry is based on the public
[NHTSA crash-simulation vehicle-model collection](https://www.nhtsa.gov/crash-simulation-vehicle-models).
The thermal graph includes documented engineering property assumptions;
the wall fields are simulation outputs, not production-line measurements.

```text
python -B code/analyze_transfer.py --surface-root vehicle_surface_fields/wall_fields --neighbors 8 --workers 1 --projection-metric capacity
python -B code/analyze_gci_transfer.py
python -B code/run_transfer_sweep.py --surface-root vehicle_surface_fields/wall_fields --parallel-runs 1 --total-workers 1
python -B scripts/analyze_evidence.py
python -B scripts/write_latex_tables.py
```

The archive contains nine application cases and two additional coarse and
medium GCI fields. `analyze_transfer.py` excludes the two GCI-only fields from
the application; `analyze_gci_transfer.py` reads the three GCI levels separately.
The envelope input in this repository is required for the same coordinate
alignment used by the recorded vehicle transfer. The CSV inputs retain the
original graph, capacities and conductances.

The evidence and table scripts require the preceding vehicle, sensitivity
and GCI outputs, or the corresponding retained results from
`supplementary_reproducibility.zip`. They do not create CFD wall fields.

## Scope of this repository

This is a small code release with four indispensable input files. The full
supplementary reproducibility archive supplies retained numerical outputs
and figure-generation code. Manuscripts, artwork, submission documents,
communication histories, scheduler configurations, large CFD files and
generated results are not part of this repository.

Numerical routines and original tests are retained. The source manifest
identifies the changes to portable field locations, the nine-case selection
and removal of historical output-directory labels from scalar GCI metadata.
