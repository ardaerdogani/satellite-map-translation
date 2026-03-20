# i2i-translation

Notebook-first image-to-image translation project for the `maps/` paired dataset.

The workflow now lives in the notebooks themselves. The command-line `scripts/` layer has been removed, so the notebooks are the main surface you present and run.

## Open These Notebooks In Order

1. `notebooks/01_setup_and_data_prep.ipynb`
2. `notebooks/02_pix2pix.ipynb`
3. `notebooks/03_cyclegan.ipynb`
4. `notebooks/04_evaluation_and_report.ipynb`

## What Stays As Python Files

Only the core engine is still kept in `src/i2i/`:

- dataset loaders
- model definitions
- training utilities
- evaluation metric helpers

Everything else that used to be a separate runnable script is now done directly inside notebook cells.

## Dataset Format

Expected source layout:

- `maps/train/*.jpg`
- `maps/val/*.jpg`

Each source image is `1200x600`:

- left half: satellite
- right half: map

Prepared dataset layout:

```text
data/pix2pix-801010/
  train/
    sat/
    map/
  val/
    sat/
    map/
  test/
    sat/
    map/
  manifest.csv
```

Training normalization:

```python
x_norm = (x / 127.5) - 1.0
```

## Outputs

The notebooks write results into:

- `outputs/pix2pix_sat2map/`
- `outputs/pix2pix_map2sat/`
- `outputs/cyclegan_sat_map/`
- `outputs/comparison_report.md`

For presentation, stay inside the notebooks and show the generated outputs from there.
