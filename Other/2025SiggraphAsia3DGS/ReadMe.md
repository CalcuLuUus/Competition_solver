# Dependencies and Environment Configuration Report

## Environment Requirements

- CUDA: `12.4`
- PyTorch: `2.5.1+cu124`
- The rest of the environment configuration is consistent with the official 3D Gaussian Splatting repository:
  https://github.com/graphdeco-inria/gaussian-splatting

---

## 1. Get the Code

```bash
# Assuming the code has been downloaded, named gaussian_splatting/
cd gaussian_splatting
```

---

## 2. Create and Activate Conda Environment

```bash
conda create -n gaussian_env python=3.12
conda activate gaussian_env
```

---

## 3. Install Dependencies

### 3.1 Install PyTorch and CUDA

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

```

### 3.2 Install Other Conda Dependencies

```bash
pip install  plyfile 
pip install  tqdm 
```

### 3.3 Install Pip Dependencies and Submodules

```bash
pip install -v  --no-build-isolation submodules/diff-gaussian-rasterization 
pip install -v  --no-build-isolation submodules/simple-knn 
pip install -v  --no-build-isolation submodules/fused-ssim 
pip install  opencv-python 
pip install  joblib
```

---

## 4. Dataset Format Requirements

Assume the data root directory is `DATA_ROOT`, with the following directory structure:

```text
|-- DATA_ROOT
    |-- <scene_name>/
        |-- images_gt_downsampled/   # Downsampled GT image sequence
        |-- sparse/                  # Official camera parameters
        |-- train_test_split.json    # Train/test view split
```

---

## 5. Modify Evaluation Script Parameters

In `full_eval_our.sh`, modify the command to:

```bash
python full_eval_our.py --ourdata path/to/your/data --output_path ./results --fast
python calmetrics.py
```

Replace `path/to/your/data` with the `DATA_ROOT` path mentioned above.

---

## 6. Result Output Locations

- Evaluation result images and intermediate results: `gaussian_splatting/results/`
- Metrics file: `gaussian_splatting/metrics.json`


Our solution win the 3rd Place in @SIGGRAPH Asia 2025 3D Gaussian Splatting Challenge (PSNR 27.25dB, Time 34s)