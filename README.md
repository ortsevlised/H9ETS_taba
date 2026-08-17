# Sustainable deepfake-image detection

This project compares MobileNetV3-Large, EfficientNet-B0 and XceptionNet on fake-face detection using DFFD. The experiment records detection quality, training cost, inference cost and peak GPU memory. It also tests each model on StyleGAN-FFHQ after excluding that generator from training and validation.

Run the commands from the project root in Windows PowerShell.

## Environment

The original runs used Windows 11, Python 3.13.7 and an NVIDIA RTX 5090. Exact package and driver versions are recorded in [environment.md](environment.md).

Create and activate a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the pinned packages. The requirements file uses the CUDA 12.8 PyTorch build used for the original runs:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

The first run of each architecture downloads its ImageNet weights.

## DFFD data

Request DFFD from the [official dataset page](https://cvlab.cse.msu.edu/dffd-dataset.html) and follow its licence conditions. The dataset is not stored in this repository. Download the seven image archives listed below and `scripts.zip`.

The official split lists are inside `scripts.zip` under `data_lists/`. Copy them into the `_lists` directory expected by `build_manifest.py`:

```powershell
Expand-Archive -Path dffd_dataset\scripts.zip -DestinationPath dffd_dataset\official_scripts
New-Item -ItemType Directory -Force -Path dffd_dataset\_lists | Out-Null
Copy-Item dffd_dataset\official_scripts\data_lists\*.txt dffd_dataset\_lists\
```

Place the archives and official split lists in this structure:

```text
dffd_dataset/
|-- _lists/
|   |-- train_real.txt
|   |-- train_fake.txt
|   |-- validation_real.txt
|   |-- validation_fake.txt
|   |-- test_real.txt
|   `-- test_fake.txt
|-- faceapp.zip
|-- ffhq.zip
|-- pggan_v1.zip
|-- pggan_v2.zip
|-- scripts.zip
|-- stargan.zip
|-- stylegan_celeba.zip
`-- stylegan_ffhq.zip
```

The archive names and their internal directory names must match this layout because the manifest builder refers to them directly.

## Prepare the sampled data

Build the seeded manifests, then extract only the images referenced by them:

```powershell
python scripts\build_manifest.py
python scripts\extract_images.py
Get-FileHash data\manifests\*.csv -Algorithm SHA256
```

This creates:

- `data/manifests/main_manifest.csv` for the main train, validation and test splits;
- `data/manifests/holdout_train_val_manifest.csv`, which excludes StyleGAN-FFHQ;
- `data/manifests/generalisation_manifest.csv` for the held-out test;
- `data/images/` containing the selected images.

The main sample contains 18,000 training images, 1,800 validation images and 9,000 test images. The manifest seed is fixed at 42.

Compare the manifest hashes with `results/reference_metrics.json` before training. Matching hashes confirm that the same image records and splits were selected.

## Train the main models

Train each architecture with seeds 0, 1 and 2. The script uses ImageNet weights, 224 by 224 inputs, AdamW, a batch size of 64 and early stopping with patience 3.

```powershell
$models = @("mobilenet_v3_large", "efficientnet_b0", "xception")
$seeds = @(0, 1, 2)

foreach ($model in $models) {
    foreach ($seed in $seeds) {
        $run = "${model}_seed${seed}"
        python scripts\train.py --model $model --run-name $run --seed $seed
    }
}
```

Each run writes its checkpoint, training history and CodeCarbon record to `runs/<run-name>/`.

## Evaluate the main models

```powershell
$models = @("mobilenet_v3_large", "efficientnet_b0", "xception")
$seeds = @(0, 1, 2)

foreach ($model in $models) {
    foreach ($seed in $seeds) {
        $run = "${model}_seed${seed}"
        python scripts\evaluate.py --model $model --run-name $run --split test --output-suffix test
    }
}
```

Evaluation uses a threshold of 0.5. It records precision, recall, F1, ROC-AUC, false-positive and false-negative rates, the confusion matrix and recall for each fake source.

The inference benchmark loads a 1,000-image buffer onto the GPU and repeatedly evaluates it for at least 30 seconds. It measures the model forward pass. Image decoding, preprocessing and host-to-device transfer are excluded.

## Run the held-out-generator test

The generalisation check trains one run per model with StyleGAN-FFHQ removed from training and validation. It then evaluates the checkpoint on 1,000 StyleGAN-FFHQ images and 1,000 FFHQ real images.

```powershell
$models = @("mobilenet_v3_large", "efficientnet_b0", "xception")
$holdoutTrainManifest = "data\manifests\holdout_train_val_manifest.csv"
$holdoutTestManifest = "data\manifests\generalisation_manifest.csv"

foreach ($model in $models) {
    $run = "${model}_holdout_seed0"
    python scripts\train.py --model $model --run-name $run --seed 0 --manifest $holdoutTrainManifest
    python scripts\evaluate.py --model $model --run-name $run --manifest $holdoutTestManifest --split generalisation --output-suffix generalisation
}
```

Only seed 0 is used for this test.

## Combine the results

Measure the idle system draw if it is needed for comparison, then build the summary:

```powershell
python scripts\measure_idle_baseline.py
python scripts\consolidate.py
```

`scripts/consolidate.py` expects the run names used above. It writes `runs/summary.json` with the per-seed results, means, sample standard deviations, decision-rule outcome and held-out-generator results.

The original metrics and manifest hashes are stored in `results/reference_metrics.json`. Detection results should be close when the same data and package versions are used. Energy, emissions and latency will change with the hardware and background system load.

## Run the explainability check

```powershell
python scripts\gradcam.py
```

This creates Grad-CAM overlays and metadata under `runs/gradcam/`. The command runs after the main models have been trained because it loads their checkpoints.

## Reproducibility notes

- CodeCarbon uses `IRL` as the country code for its grid-intensity estimate.
- Energy, emissions, latency and peak memory depend on the GPU, drivers and background system load.
- Training uses fixed random seeds, but exact GPU results can still vary across hardware and software versions.
- The dataset, extracted images, checkpoints and generated results are excluded from Git because of their size and licensing conditions.
