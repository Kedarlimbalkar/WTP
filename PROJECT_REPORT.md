# Wind Turbine Spotter — Project Report

## 1. Executive summary

Wind Turbine Spotter is a browser-based object-detection application for locating
wind turbines in individual images or image collections. Its backend is a
FastAPI application that loads a fine-tuned Ultralytics YOLOv8-small model,
returns turbine bounding boxes and confidence scores, and can package batch
results into a ZIP archive.

The model is trained in a Kaggle notebook using a two-class dataset
(`wind_turbine` and `pylon`). At inference time, pylon detections are discarded:
the product reports turbines only. The web application is packaged in a Docker
image and configured to download its model weights from Azure Blob Storage at
container startup.

This report is based on the checked-in application, notebook, tests, Dockerfile,
workflow files, and architecture diagrams. In particular, the notebook contains
the v10 training configuration but does not preserve the final validation or
test metric summary. No final accuracy claim is made here.

## 2. Problem and project objectives

Manually finding and counting turbines in aerial, ground, or other imagery can
be slow, particularly when a user has a large collection of images. The project
provides a visual tool to:

1. Upload one image or a ZIP of images.
2. Run object detection and display the detected turbine locations.
3. Review confidence scores and per-image counts.
4. Download annotated images and CSV summaries for batch jobs.

The intended output is decision support, not an authoritative inventory.
Confidence values describe model output and should be interpreted in the
context of the image and the model's validation limitations.

## 3. Repository and component overview

| Path | Responsibility |
|---|---|
| [`src/app.py`](src/app.py) | FastAPI routes, YOLO inference, result formatting, ZIP processing, CSV generation, and the embedded browser UI. |
| [`src/download_model.py`](src/download_model.py) | Downloads the model weights from Azure Blob Storage when the container starts, unless the file already exists locally. |
| [`notebooks/v10-9.ipynb`](notebooks/v10-9.ipynb) | Kaggle data preparation, v9/v10 training experiments, validation code, example inference, and model publication steps. |
| [`scripts/upload_model.py`](scripts/upload_model.py) | Uploads a local `.pt` model file to the configured Azure Blob Storage container and blob path. |
| [`tests/test_app.py`](tests/test_app.py) | Unit/API tests intended to cover detection logic and HTTP routes. Some tests currently refer to an older API and should be reconciled with the app. |
| [`requirements.txt`](requirements.txt) | Runtime Python dependencies. |
| [`requirements-dev.txt`](requirements-dev.txt) | Test, lint, coverage, and type-checking tools. |
| [`pyproject.toml`](pyproject.toml) | Ruff, pytest, and coverage configuration. |
| [`Dockerfile`](Dockerfile) | Builds the Python 3.11 application image and starts model download followed by Uvicorn. |
| [CI workflow](.github/workflows/ci.yml) | Runs Ruff, mypy (non-blocking), pytest with coverage, and a coverage threshold. |
| [Azure deployment workflow](.github/workflows/wind-mill-app-AutoDeployTrigger-99602067-9e11-424e-86f7-57f479e6b001.yml) | Builds and deploys the image to Azure Container Apps on pushes to `main` or by manual dispatch. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md), [`diagram1_system_architecture.mmd`](diagram1_system_architecture.mmd), [`diagram2_bulk_pipeline.mmd`](diagram2_bulk_pipeline.mmd), [`diagram3_data_model.mmd`](diagram3_data_model.mmd) | Existing architecture and batch-processing diagrams. |

The model weights and training dataset are intentionally not part of the source
tree. The `.gitignore` excludes `.pt` files and dataset folders; production
weights are expected in Azure Blob Storage.

## 4. Machine-learning data and training

### 4.1 Dataset source and preparation

The notebook identifies the source as Kyle Graupe's Kaggle **Object Detection
Dataset - Wind Turbines**, in YOLO format. The notebook searches the attached
Kaggle input for image/label split directories and copies the data into a
writable normalized dataset tree.

The canonical class IDs used by the notebook are:

| Class ID | Canonical name | Meaning |
|---:|---|---|
| 0 | `wind_turbine` | Target object counted and shown by the application. |
| 1 | `pylon` | Secondary class retained during training to help distinguish pylons/towers from turbines; filtered out by the deployed application. |

The source labels shown in the notebook are `cable tower` and `turbine`.
Preparation maps those names to the canonical classes above, copies images and
rewrites YOLO-format label files. Images with no labels are retained as
background examples. Unknown source classes are dropped by the mapping logic.
The notebook checks for at least `train` and `valid` splits and uses a `test`
split when it is available.

The prepared dataset counts recorded in the notebook are:

| Split | Images | Turbine boxes | Pylon boxes | Empty-label images |
|---|---:|---:|---:|---:|
| Train | 2,643 | 15,534 | 441 | 15 |
| Validation | 247 | 1,538 | 24 | 1 |
| Test | 130 | 726 | 26 | 2 |

These are object/annotation counts, not counts of unique turbines in the real
world. The source data and annotation quality should be checked before treating
any count as a certified dataset statistic.

### 4.2 Model and training procedure

The v10 notebook cell initializes `YOLO('yolov8s.pt')`, which is the small
YOLOv8 detection model with pretrained starting weights, and fine-tunes it for
the two project classes. The recorded training setup is:

| Setting | Configured value |
|---|---|
| Framework | Ultralytics YOLO; notebook output records Ultralytics 8.4.171 |
| Initial checkpoint | `yolov8s.pt` |
| Task | Object detection |
| Classes | 2 (`wind_turbine`, `pylon`) |
| Input image size | 960 pixels |
| Batch size | 16 |
| Maximum epochs | 100 |
| Early stopping patience | 20 epochs |
| Optimizer | AdamW |
| Initial learning rate | 0.005 |
| Final learning-rate factor | 0.01 |
| Learning-rate schedule | Cosine |
| Warmup | 3 epochs |
| Weight decay | 0.0005 |
| Random seed | 42 |
| Mosaic | 1.0, closed for the final 15 epochs |
| MixUp / copy-paste | 0.0 / 0.0 |
| Geometric augmentation | Rotation 3°, translation 0.1, scale 0.5; shear and perspective 0 |
| Flip | Horizontal 0.5; vertical 0 |
| HSV augmentation | Hue 0.015, saturation 0.5, value 0.4 |
| Recorded training environment | Kaggle, Tesla T4 GPU; Python 3.12.13 and PyTorch 2.10.0+cu128 in notebook output |

The maximum epoch count is a training setting; the saved notebook output does
not establish how many epochs completed or whether early stopping occurred.
The notebook also contains an earlier v9 experiment (80 epochs, patience 15,
with stronger augmentation), but v10 is the later recipe associated with the
`turbine_v10` model path used for deployment.

### 4.3 Evaluation and model artifact

The notebook contains code to evaluate the selected `best.pt` checkpoint on
validation data and, if present, the held-out test split. The requested
evaluation measures are precision, recall, mAP@0.5, and mAP@0.5:0.95, with a
per-class AP@0.5 breakdown. It also includes code for checking false positives
on pylon-only and background images, and code for visually comparing predictions
to labels.

The final metric values and the completed false-positive-check output are not
available in the saved notebook output included in this repository. Therefore:

* Model performance cannot be independently quantified from the checked-in
  artifacts.
* Training logs visible in the notebook are partial; they do not establish a
  final checkpoint score or held-out test result.
* A successful training run or a successful deployment does not, on its own,
  demonstrate detection accuracy.

The notebook copies its best checkpoint to
`/kaggle/working/turbine_v10_final.pt`. The deployment downloader expects the
Azure blob `models/turbine_v10/best.pt` in the `wind-turbine-data` container by
default. The upload helper writes to that blob path, but its checked-in local
source path is specific to one Windows machine
(`E:\Atman-project\V10\best.pt`). Confirm that this local file is the intended
v10 `best.pt` checkpoint before publishing it; the repository does not contain
the weight file or a model checksum/version manifest.

## 5. Application implementation

### 5.1 Startup and model loading

The Docker image uses Python 3.11 slim, installs operating-system libraries
needed by OpenCV/Pillow, installs packages from `requirements.txt`, and copies
the `src` package. On container startup, its command runs
`src/download_model.py` and then starts Uvicorn on port 8000.

The downloader reads:

* `AZURE_STORAGE_CONNECTION_STRING` (required),
* `AZURE_CONTAINER_NAME` (defaults to `wind-turbine-data`),
* `MODEL_BLOB_PATH` (defaults to `models/turbine_v10/best.pt`), and
* `MODEL_PATH` (defaults to `model/best.pt`).

If the local weights file does not exist, it downloads the blob to disk. The
FastAPI lifespan handler checks the resulting local model path, creates the
Ultralytics `YOLO` model once, and stores it in application state. Startup fails
with an explicit error if the file is missing. The `model_lock` serializes
calls into model prediction.

### 5.2 Single-image detection

`POST /predict` accepts a multipart image and optional form fields:

| Field | Default | Purpose |
|---|---:|---|
| `conf` | 0.25 | Minimum YOLO confidence threshold. |
| `iou` | 0.7 | Non-maximum-suppression overlap threshold. |
| `imgsz` | 640 | Inference image size. |

The image is decoded with Pillow and converted to RGB. Invalid or unreadable
image data returns HTTP 400. The inference helper calls `model.predict`,
measures inference time, converts boxes to pixel-coordinate `[x1, y1, x2, y2]`
values, rounds confidence and coordinates, removes any class whose name
contains `pylon` (case-insensitive), and sorts the remaining detections by
confidence.

The response includes image width and height, inference time in milliseconds,
the turbine count, and a list of detections containing label, confidence, and
bounding box. The model's class metadata is exposed through `GET /info`, which
also reports the ZIP limits.

### 5.3 Batch ZIP processing

`POST /predict_zip` accepts a ZIP plus the same inference settings. It:

1. Removes expired completed jobs from the in-memory registry.
2. Streams the upload to a per-job temporary directory while enforcing the ZIP
   size limit (default 200 MB).
3. Rejects invalid ZIP files, archives without supported images, and archives
   containing more than 300 images.
4. Ignores directories, hidden entries, and macOS metadata entries; supported
   extensions include JPEG, PNG, BMP, WebP, and TIFF variants.
5. Creates an in-memory job record and starts a daemon worker thread, then
   returns a job ID and image total to the browser.

The background worker processes each archive image in sequence. Files over the
per-image limit (default 50 MB) and unreadable images become per-image error
rows; they do not prevent valid images in the same archive from being processed.
Each valid image is detected, annotated with Pillow, stored as a JPEG in the
output archive, and also rendered as an 800-by-800 maximum thumbnail for the
browser gallery.

The downloadable `results.zip` contains an `annotated/` folder plus:

* `results.csv` — one row per image with the filename, turbine count, maximum
  confidence, inference time, and status.
* `detections.csv` — one row per turbine box with filename, label, confidence,
  and pixel coordinates.

`GET /jobs/{job_id}` reports job status, progress, total turbines, and processed
rows. `GET /jobs/{job_id}/image/{idx}` serves a thumbnail. Once the job reaches
`done`, `GET /jobs/{job_id}/download` returns the result archive. Completed jobs
are retained in temporary storage and the in-memory registry for up to one
hour, after which cleanup occurs on a later ZIP-job request.

### 5.4 Browser interface

The UI is embedded in `src/app.py` as a single HTML/CSS/JavaScript document and
is served at `/`. It provides single-image and ZIP modes, drag-and-drop and file
selection, controls for confidence/NMS/image size, a single-image canvas
overlay, and batch progress/gallery/download controls. The batch UI polls the
job endpoint while processing continues.

The two principal request paths are:

```text
Single image: Browser -> POST /predict -> YOLO -> JSON detections -> canvas
ZIP batch:    Browser -> POST /predict_zip -> background worker
              -> GET /jobs/{id} + thumbnails -> download results ZIP
```

## 6. Deployment and operations

The CI workflow targets Python 3.11 on GitHub-hosted Ubuntu runners. It installs
runtime and development dependencies, runs Ruff, runs mypy with
`continue-on-error: true`, runs pytest with coverage, and enforces a 70% coverage
threshold.

The Azure deployment workflow is configured to build and push the Docker image
to Azure Container Registry and deploy it to the Azure Container App named
`wind-mill-app`. It uses GitHub Actions secrets for Azure workload identity and
registry credentials. Model-storage credentials must also be supplied to the
running container as environment configuration/secrets; they are not provided
by the Dockerfile.

Batch job state, locks, worker threads, and temporary files are local to one
application process/container. They are not persisted in a database or shared
between replicas. A production deployment that routes subsequent polling or
download requests to a different replica may not find the original job. The
current design is consequently best suited to a single instance or a deployment
with deliberate session affinity; durable/distributed job storage would be
needed for robust multi-replica batch processing.

## 7. Testing status and known gaps

The checked-in tests describe the intended use of `TestClient` and mock model
inference so CI does not require model weights or a GPU. However, the current
tests are not aligned with the application implementation:

* They import `merge_stacked_boxes`, which is not defined in the current
  `src/app.py`.
* They call `POST /detect`, while the current endpoint is `POST /predict`.
* They call `GET /detect-bulk/export-csv`, which is not an implemented route;
  CSV files are currently included in the ZIP download.
* They expect a `needs_review` field that the current prediction response does
  not return.

Accordingly, the test file should not be treated as evidence that the current
API passes CI. Update the tests to match the current API (or restore the
intended legacy behavior), then run the configured CI checks and record the
actual results. This report does not claim tests passed.

Other reproducibility and evaluation gaps:

* Runtime dependencies are mostly unpinned, so a rebuild may resolve different
  package versions.
* The final training metrics and test-set results are not preserved in the
  checked-in notebook outputs.
* The training dataset and final model weights are external artifacts; their
  exact versions/checksums are not recorded here.
* The model class mapping and deployed pylon filter must remain consistent when
  replacing model weights.
* Batch jobs and their generated files are ephemeral and process-local.

## 8. Suggested next steps

1. Re-run validation and held-out test evaluation for the exact checkpoint
   uploaded to Azure; save the complete metrics and evaluation environment.
2. Align the automated tests with the active `/predict` and `/predict_zip`
   API, then verify lint, type-check, tests, and the coverage gate.
3. Make the model upload source path configurable and publish a checksum,
   model version, class-name mapping, and training-run identifier with each
   model release.
4. Pin the production dependencies (including the PyTorch wheel source/version)
   and rebuild the container to check reproducibility.
5. If scaling beyond one application instance, move job state/results to
   shared durable storage and use a managed background queue/worker pattern.
6. Document the exact Azure Container App settings for model-download secrets,
   resource sizing, health checks, and scale limits.

## 9. Local development

For local development, install the dependencies from `requirements.txt`, make
the model file available at `MODEL_PATH` (default `model/best.pt`), and start
the API with:

```powershell
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

The browser interface is served at `http://127.0.0.1:8000/`. For a local run
without an Azure download, place the intended model checkpoint at the configured
model path before startup. Keep storage connection strings and other credentials
in environment configuration or a secret manager; do not add them to source
control.
