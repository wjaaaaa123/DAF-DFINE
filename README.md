# DAF-DFINE

Official implementation of **DAF-DFINE: A Defect-Aware and Frequency-Adaptive Network for Aluminum Surface Defect Detection**.

DAF-DFINE is an end-to-end detector built on **D-FINE-S** for industrial aluminum surface defect detection. The framework introduces three complementary modules:

- **DHEC**: Directional High-order Edge-enhanced Convolution for shallow directional and weak-texture feature enhancement.
- **DFAF**: Directional Frequency-Adaptive Fusion for adaptive cross-scale feature coordination.
- **DD-LKA**: Defect-aware Deformable Large Kernel Attention for irregular-defect spatial modeling.
- **DAF-DFINE-Distill**: a lightweight deployment-oriented student model obtained by channel-width compression and Channel-wise Knowledge Distillation (CWD).

The detector preserves the end-to-end, NMS-free detection paradigm of D-FINE.

---

## 1. Repository Structure

```text
DAF-DFINE/
├── configs/
│   ├── base/
│   ├── dataset/
│   ├── dfine/
│   ├── ultralytics-yaml/
│   └── runtime.yml
│
├── dataset/
│   ├── README.md
│   └── yolo2coco.py
│
├── engine/
│   ├── backbone/
│   ├── core/
│   ├── data/
│   ├── deim/
│   ├── extre_module/
│   ├── logger_module/
│   ├── misc/
│   ├── optim/
│   ├── solver/
│   └── __init__.py
│
├── tools/
│   ├── benchmark/
│   ├── dataset/
│   ├── deployment/
│   ├── inference/
│   ├── utils/
│   └── visualization/
│
├── train.py
├── distill.py
├── train_auto.py
├── requirements.txt
├── ModuleList.md
├── UserGuide-Easy.md
├── ServerInfo.md
├── UpdateLog.md
├── LICENSE
└── README.md
```

---

## 2. Main Modules

The core DAF-DFINE modules are located under `engine/extre_module/custom_nn/`.

Typical locations are:

```text
engine/extre_module/custom_nn/conv_module/DHEC.py
engine/extre_module/custom_nn/featurefusion/DFAF.py
engine/extre_module/custom_nn/attention/DD_LKA.py
```

The distillation utilities are located in:

```text
engine/extre_module/distill_utils.py
```

---

## 3. Environment

Recommended environment:

- Python 3.9
- PyTorch 2.7.1
- CUDA 12.8
- NVIDIA GPU

The experiments in the paper were conducted on a Windows 11 workstation with an NVIDIA GeForce RTX 5060 Ti GPU.

### 3.1 Create Environment

Using Conda:

```bash
conda create -n deim python=3.9 -y
conda activate deim
```

### 3.2 Install PyTorch

Install the PyTorch build that matches your CUDA environment from the official PyTorch installation guide.

Example environment used in our experiments:

```text
PyTorch 2.7.1
CUDA 12.8
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

> PyTorch and torchvision are intentionally not strictly pinned in `requirements.txt` because CUDA-compatible wheels depend on the local environment.

---

## 4. Dataset Preparation

The primary aluminum surface defect dataset contains **2,776 original images** and **10 defect categories**:

1. Nonconductive
2. Scratch
3. Corner Leak
4. Orange Peel
5. Leakage
6. Jet
7. Paint Bubble
8. Crater
9. Parti-color
10. Dirty Point

Dataset files are not distributed with this repository.

Please refer to:

```text
dataset/README.md
```

for the recommended directory structure, split protocol, annotation format, and conversion instructions.

The dataset configuration is located at:

```text
configs/dataset/aluminium_detection.yml
```

Before training, update the dataset path according to your local environment.

---

## 5. Training

### 5.1 D-FINE-S Baseline

Example:

```bash
python -X utf8 train.py -c configs/dfine/dfine_hgnetv2_s_aluminium.yml --seed=0
```

### 5.2 DAF-DFINE

Use the final DAF-DFINE configuration in `configs/ultralytics-yaml/`.

Example:

```bash
python -X utf8 train.py -c configs/ultralytics-yaml/<daf_dfine_config>.yml --seed=0
```

Replace `<daf_dfine_config>.yml` with the final configuration file included in this repository.

---

## 6. Lightweight DAF-DFINE-Distill

The full-width DAF-DFINE model is used as the **Teacher**, while the compressed approximately `0.50×` model is used as the **Student**.

The distillation entry point is:

```text
distill.py
```

General usage:

```bash
python -X utf8 distill.py \
  --student-config configs/ultralytics-yaml/<student_config>.yml \
  --teacher-config configs/ultralytics-yaml/<teacher_config>.yml \
  --teacher-weights <teacher_checkpoint>
```

On Windows CMD, use a single line:

```cmd
python -X utf8 distill.py --student-config configs/ultralytics-yaml/<student_config>.yml --teacher-config configs/ultralytics-yaml/<teacher_config>.yml --teacher-weights <teacher_checkpoint>
```

During inference, only the student model is retained.

---

## 7. Evaluation

The paper reports:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Parameters
- FLOPs
- FPS

Related scripts are under:

```text
tools/benchmark/
tools/inference/
tools/visualization/
```

---

## 8. Main Results

### 8.1 Aluminum Surface Defect Dataset

| Model | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|---:|
| D-FINE-S | 81.5 | 80.9 | 84.3 | 65.5 | 10.18 | 25.37 |
| DAF-DFINE | **86.1** | **84.8** | **87.6** | **69.2** | 12.13 | 29.85 |
| DAF-DFINE-Distill | 84.9 | 84.0 | 86.5 | 67.9 | **5.06** | **12.28** |

Compared with D-FINE-S, DAF-DFINE improves mAP@0.5 and mAP@0.5:0.95 by **3.3** and **3.7** percentage points, respectively.

DAF-DFINE-Distill improves the same metrics by **2.2** and **2.4** percentage points while reducing parameters and FLOPs by **50.3%** and **51.6%**, respectively.

### 8.2 Jetson Xavier NX

| Model | mAP@0.5 (%) | Params (M) | FLOPs (G) | FPS |
|---|---:|---:|---:|---:|
| D-FINE-S | 84.1 | 10.18 | 25.37 | 31.4 |
| DAF-DFINE | 87.2 | 12.13 | 29.85 | 29.2 |
| DAF-DFINE-Distill | **85.4** | **5.06** | **12.28** | **56.7** |

---

## 9. Deployment

The deployment workflow used in the paper follows:

```text
PyTorch
  ↓
ONNX
  ↓
ONNX Simplifier
  ↓
TensorRT
  ↓
Jetson Xavier NX
```

Related scripts can be placed under:

```text
tools/deployment/
```

---

## 10. Notes

- Datasets, checkpoints, ONNX files, TensorRT engines, training outputs, and TensorBoard logs are intentionally excluded.
- Use the same dataset split and evaluation protocol when comparing models.
- Validation and test sets should use deterministic preprocessing.
- Avoid placing augmented variants derived from the same original image in different subsets.
- Public YAML files should use relative paths whenever possible.

---

## 11. Documentation

- [Dataset Preparation](dataset/README.md)
- [Module List](ModuleList.md)
- [Quick Start / User Guide](UserGuide-Easy.md)
- [Environment Information](ServerInfo.md)
- [Update Log](UpdateLog.md)

---

## 12. Citation

If this repository is useful for your research, please cite the corresponding paper after publication:

```bibtex
@article{DAF-DFINE,
  title   = {DAF-DFINE: A Defect-Aware and Frequency-Adaptive Network for Aluminum Surface Defect Detection},
  author  = {Zhang, Chen and Wang, Jie and Shan, Wentao and Han, Zhenhua},
  journal = {IEEE Access},
  year    = {2026}
}
```

The final volume, pages, and DOI will be updated after publication.

---

## 13. Acknowledgment

This project is built upon D-FINE and related open-source object detection codebases. We thank the corresponding authors and open-source community for their contributions.

---

## 14. License

Please refer to [LICENSE](LICENSE) for licensing information.
