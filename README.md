# Spiral Tremor Analysis Pipeline

Python implementation of an image-based spiral drawing analysis pipeline for essential tremor assessment after MR-guided focused ultrasound (MRgFUS) thalamotomy.

The pipeline reads scanned spiral drawings and accompanying metadata, extracts the patient-drawn trace, computes interpretable image-derived tremor metrics, and optionally fits an ordinal logistic model for clinician-rated CRST spiral scores.

## What this project does

Given:

- a folder of scanned spiral drawing images, and
- a CSV file containing the image paths and clinical metadata,

this pipeline computes three scalar spiral metrics:

1. **Total line length**  
   Skeletonizes the extracted patient trace and estimates its physical length in millimeters using 8-connectivity. Longer traces may reflect oscillatory stroke production, retracing, or inefficient drawing caused by tremor.

2. **Sobel orientation irregularity**  
   Applies Sobel gradients to the isolated trace, compares local edge orientation with the radial azimuth from the spiral center, and summarizes the dispersion of relative orientations.

3. **Optimal-solution deviation**  
   Resamples the patient trace and a reference spiral trajectory to the same number of points, then computes the aggregate Manhattan distance between corresponding points.

The script can also fit an ordered logistic regression using:

```text
CRST spiral score ~ Sobel score + optimal-solution deviation + line length
                    + post-treatment status + treated-hand status
                    + post-treatment × treated-hand interaction
```

## Repository contents

```text
spiral_tremor_pipeline.py       Main analysis pipeline
requirements_spiral_tremor.txt  Python dependencies
README_spiral_tremor.md         This README
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements_spiral_tremor.txt
```

Optional Bayesian mixed-model support requires additional packages:

```bash
pip install pymc arviz
```

## Expected input structure

A typical project folder may look like this:

```text
project/
├── data/
│   ├── images/
│   │   ├── patient001_pre_right.png
│   │   ├── patient001_pre_left.png
│   │   ├── patient001_post_right.png
│   │   └── patient001_post_left.png
│   ├── empty_spiral_A.png
│   └── metadata.csv
├── spiral_tremor_pipeline.py
└── requirements_spiral_tremor.txt
```

The image paths in the metadata CSV can be either absolute paths or paths relative to `--data_dir`.

## Metadata CSV

By default, the script expects the following columns:

| Column | Required for | Description |
|---|---:|---|
| `image_path` | Metrics | Path to the scanned spiral image. |
| `subject_id` | Modeling | Subject or patient identifier. |
| `crst_score` | Modeling | Clinician-rated CRST spiral score, usually integer 0–4. |
| `hand` | Modeling | Drawing hand, for example `right` or `left`. |
| `treated_hand` | Modeling | Hand treated by MRgFUS, for example `right` or `left`. |
| `time` | Modeling | Pre/post indicator, for example `pre`, `post`, `baseline`, or `follow-up`. |

You can override all default column names using command-line arguments.

Example metadata:

```csv
image_path,subject_id,crst_score,hand,treated_hand,time
patient001_pre_right.png,patient001,3,right,right,pre
patient001_pre_left.png,patient001,2,left,right,pre
patient001_post_right.png,patient001,0,right,right,post
patient001_post_left.png,patient001,2,left,right,post
```

If you do not have a `time` column, the script can infer pre/post status from dates when both date columns are provided:

```text
--drawing_date_col drawing_date --treatment_date_col treatment_date
```

## Quick start

Compute the image metrics only:

```bash
python spiral_tremor_pipeline.py \
  --data_dir data/images \
  --metadata_csv data/metadata.csv \
  --output_dir outputs \
  --empty_template data/empty_spiral_A.png
```

Compute metrics and fit the ordinal model:

```bash
python spiral_tremor_pipeline.py \
  --data_dir data/images \
  --metadata_csv data/metadata.csv \
  --output_dir outputs \
  --empty_template data/empty_spiral_A.png \
  --fit_model
```

Use an explicitly prepared ideal reference trajectory:

```bash
python spiral_tremor_pipeline.py \
  --data_dir data/images \
  --metadata_csv data/metadata.csv \
  --output_dir outputs \
  --empty_template data/empty_spiral_A.png \
  --ideal_reference data/ideal_spiral_A_midline.png \
  --fit_model
```

Run the optional Bayesian ordered-logit random-intercept model:

```bash
python spiral_tremor_pipeline.py \
  --data_dir data/images \
  --metadata_csv data/metadata.csv \
  --output_dir outputs \
  --empty_template data/empty_spiral_A.png \
  --fit_model \
  --run_pymc_mixed
```

## Important command-line options

| Argument | Default | Description |
|---|---:|---|
| `--data_dir` | required | Folder containing spiral images. |
| `--metadata_csv` | required | CSV file with image paths and metadata. |
| `--output_dir` | required | Folder where outputs are written. |
| `--empty_template` | optional | Empty Spiral A template used for template subtraction and approximate reference generation. |
| `--ideal_reference` | optional | Ideal midline/reference image for optimal-solution deviation. Recommended when available. |
| `--dpi` | `300` | Scan resolution used to convert pixels to millimeters. |
| `--image_size` | `500` | Analysis image size after preprocessing. |
| `--threshold_method` | `otsu` | Thresholding method: `otsu`, `adaptive`, or `percentile`. |
| `--roi` | optional | Fixed crop region as `x,y,w,h`. Useful when scans share the same layout. |
| `--no_auto_crop` | off | Disable automatic foreground cropping. |
| `--fit_model` | off | Fit the ordered logistic model after computing metrics. |
| `--run_pymc_mixed` | off | Run the optional Bayesian random-intercept ordinal model. |

## Outputs

The pipeline writes the following files to `--output_dir`:

| Output file | Created when | Description |
|---|---|---|
| `spiral_metrics.csv` | Always | Original metadata plus computed image metrics and processing errors, if any. |
| `ordinal_cluster_model_odds_ratios.csv` | `--fit_model` | Odds ratios, confidence intervals, z statistics, and p-values for the ordinal model predictors. |
| `ordinal_cluster_model_all_parameters.csv` | `--fit_model` | Full ordered-logit parameter table, including cutpoints. |
| `ordinal_cluster_model_statsmodels_summary.txt` | `--fit_model` | Raw `statsmodels` model summary. |
| `bayesian_ordinal_mixed_model.nc` | `--run_pymc_mixed` | ArviZ/PyMC model trace. |
| `bayesian_ordinal_mixed_model_summary.csv` | `--run_pymc_mixed` | Posterior summary for the Bayesian mixed model. |

The main metrics columns are:

```text
line_length_mm
sobel_orientation_irregularity_deg
optimal_solution_deviation_mm
trace_pixels
processing_error
```

Rows that fail processing are retained in the output CSV with the error message stored in `processing_error`.

## Method overview

The implementation follows a simple, reproducible image-processing workflow:

1. Read each scan as grayscale.
2. Crop the spiral region automatically or with a fixed ROI.
3. Resize the image to a square analysis canvas, by default 500 × 500 pixels.
4. Threshold dark pixels to identify foreground ink/template marks.
5. If an empty template is provided, subtract the template to isolate the patient-drawn trace.
6. Keep the largest connected components to reduce dust, labels, and small artifacts.
7. Skeletonize the extracted trace for length and trajectory-based measurements.
8. Compute the three spiral metrics.
9. Optionally fit an ordinal model for clinician-rated CRST spiral scores.

## Design choices

The implementation prioritizes reproducibility, interpretability, and robustness over highly tuned template-specific assumptions. Each metric is computed as a separate function, the metadata schema is configurable through command-line arguments, and the script preserves failed rows instead of stopping the entire run. The default statistical model is a Python-friendly ordered logistic regression with subject-cluster robust standard errors, while an optional Bayesian model is provided for users who want a closer random-intercept analogue to the mixed-effects formulation.

## Notes on the optimal-solution score

The manuscript describes the optimal-solution deviation score at a conceptual level. The original private implementation may include template-specific choices that are not fully recoverable from the text alone.

This pipeline therefore supports two reproducible modes:

- **Preferred:** provide an explicit `--ideal_reference` image containing the ideal spiral midline.
- **Fallback:** provide only `--empty_template`; the script skeletonizes the template foreground and uses it as an approximate reference.

For exact replication of a prior study, validate the generated reference trajectory against the original reference-generation code or against manually inspected reference images.

## Data quality recommendations

For best results:

- use scans with a known and consistent DPI, preferably 300 DPI;
- keep the same template layout across all observations;
- avoid scans with severe rotation, cropping, blur, or low contrast;
- avoid overlapping handwritten notes or clinical markings in the spiral region;
- prefer a fixed ROI when all scans come from the same template and scanner;
- visually inspect a sample of extracted masks before running the full analysis.

Severe tremor, retracing, or poor task performance should not automatically be excluded, because these may reflect true clinical severity. Exclusion is most appropriate when the trace cannot be reliably separated from the template or artifacts.

## Interpreting the model

The ordered logistic model predicts the odds of a higher, worse CRST spiral score. Odds ratios greater than 1 indicate higher odds of worse clinician-rated tremor severity, while odds ratios below 1 indicate lower odds of worse severity.

The `post_x_treated` interaction tests whether pre-to-post change differs between the treated and non-treated hands. This is useful for checking whether the model captures the expected MRgFUS treatment signature: improvement primarily in the treated hand.

## Limitations

This implementation is intended as a research pipeline, not a clinical device. Important limitations include:

- scanned paper drawings do not capture drawing speed, pauses, pressure, or tremor frequency;
- image-derived metrics can be affected by scanner settings, pen thickness, contrast, rotation, and template alignment;
- automatic trace extraction may need tuning for a specific scanner/template setup;
- the default Python model is a cluster-robust sensitivity model, not a frequentist ordered-logit mixed-effects model with random intercepts;
- the optimal-solution score is an approximation unless an explicit ideal reference trajectory is supplied.

## Troubleshooting

### Many rows have empty or tiny traces

Try:

```bash
--threshold_method adaptive
```

or provide a fixed crop:

```bash
--roi x,y,w,h
```

### Template subtraction removes too much of the patient trace

Try running without `--empty_template` to inspect whether the raw thresholded trace is more stable, or tune the preprocessing for your template.

### The optimal-solution deviation is `NaN`

Provide either:

```bash
--ideal_reference path/to/reference.png
```

or:

```bash
--empty_template path/to/empty_template.png
```

### The model does not fit

Check that the following columns are present and complete:

```text
subject_id, crst_score, hand, treated_hand, time
```

Also confirm that `crst_score` contains valid ordered categories, typically integers from 0 to 4.

## Minimal Python API example

The script is designed primarily for command-line use, but its functions can also be imported:

```python
from pathlib import Path
import pandas as pd
from spiral_tremor_pipeline import PipelineConfig, compute_all_metrics

metadata = pd.read_csv("data/metadata.csv")
cfg = PipelineConfig(dpi=300, image_size=500)

metrics = compute_all_metrics(
    metadata=metadata,
    data_dir=Path("data/images"),
    output_dir=Path("outputs"),
    cfg=cfg,
    image_col="image_path",
    empty_template_path=Path("data/empty_spiral_A.png"),
)

print(metrics.head())
```

## Suggested citation text

If this code is used in a manuscript, report that the pipeline computes image-derived spiral metrics from preprocessed Spiral A scans, including skeletonized total line length, Sobel-based orientation irregularity, and reference-trajectory Manhattan deviation, followed by ordinal modeling of clinician-rated CRST spiral scores.

## License

Add the appropriate project license before public release.
