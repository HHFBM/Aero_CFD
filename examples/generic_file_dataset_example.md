# Generic File Dataset Example

This example shows a minimal **file-backed generic 2D airfoil dataset** that can be loaded with:

```yaml
data:
  dataset_type: file
  dataset_path: examples/generic_file_dataset_example.csv
  branch_feature_mode: points
  num_surface_points: 4
```

The paired CSV is:

- [generic_file_dataset_example.csv](/Users/jason/Documents/CFD/examples/generic_file_dataset_example.csv)

## What This Example Demonstrates

- `geometry_mode=generic_surface_points`
- geometry is provided as discrete 2D airfoil points, not NACA parameters
- no `branch_*` columns are included
- the dataloader derives `branch_inputs` automatically from `geometry_x/geometry_y`
- the same sample contains:
  - geometry points
  - query points
  - surface rows
  - global scalar targets

This is the intended path for **generic 2D airfoil training data** when you want the current project to keep its fixed-size branch network while no longer requiring NACA parameters.

## Column Meanings

- `sample_id`
  - groups rows into one CFD sample
- `geometry_mode`
  - use `generic_surface_points` for generic 2D airfoil contours
- `geometry_x`, `geometry_y`
  - 2D airfoil geometry points
  - the loader collects unique points inside each `sample_id`
  - they are used to build canonical geometry and then derive `branch_inputs`
- `mach`, `aoa`
  - sample-level flow conditions
- `x`, `y`
  - query point coordinates for field supervision
- `u`, `v`, `p`
  - pointwise supervised field targets
- `surface_flag`
  - `1` for rows that lie on the airfoil surface
  - `0` for off-surface query points
- `cp`
  - surface Cp target for rows with `surface_flag=1`
- `cl`, `cd`
  - sample-level scalar targets
- `fidelity_level`, `source`, `convergence_flag`
  - project metadata fields kept for compatibility

## Minimum Required Columns

For a generic file dataset, the current loader expects these columns:

- `sample_id`
- `mach`
- `aoa`
- `x`
- `y`
- `u`
- `v`
- `p`

For geometry input, provide at least one of:

- `geometry_x`
- `geometry_y`

Recommended but optional columns:

- `geometry_mode`
  - defaults to `generic_surface_points` when omitted
- `surface_flag`
  - needed only if you want explicit surface supervision rows from the query table
- `cp`
  - needed only if you want `surface_cp` capability
- `cl`
- `cd`
  - scalar supervision stays available when present; otherwise scalar metrics/losses should be disabled or capability-gated
- `fidelity_level`
- `source`
- `convergence_flag`
  - default compatibility values are filled when omitted

## How Automatic Branch Encoding Works

If you **do not** provide precomputed `branch_*` columns, the loader uses `DataConfig` to derive `branch_inputs`:

- `branch_feature_mode: params`
  - derive a compact geometry summary from surface points
  - branch encoding becomes `derived_geometry_summary_plus_flow`
- `branch_feature_mode: points`
  - derive a canonical surface signature from geometry points
  - branch encoding becomes `derived_surface_signature_plus_flow`

This keeps the current fixed-width branch network intact while allowing training on non-NACA airfoil geometry.

## When To Provide `branch_*` Yourself

Provide explicit `branch_0`, `branch_1`, ... columns if:

- you already have a custom geometry encoder
- you want exact control over branch input contents
- you need a branch schema different from the built-in summary/signature adapters

If `branch_*` exists, the loader uses those values directly.

## Important Limits

- This example upgrades the **data/geometry interface**, not the model into a variable-length native geometry encoder
- training still uses fixed-width `branch_inputs`
- generic geometry is therefore supported through a clean adapter path, not a full branch-network redesign
- this example is 2D only

## Recommended Next Step

To try it quickly, point a file-backed experiment at this CSV and keep the model dimensions inferred from the datamodule:

```yaml
data:
  dataset_type: file
  dataset_path: examples/generic_file_dataset_example.csv
  branch_feature_mode: points
  num_surface_points: 4
  strict_quality_checks: true
```

Then run training through the normal entrypoint:

```bash
python scripts/train.py --config <your_config>.yaml
```
