

| ID | Class |
|---:|---|
| 0 | Nonconductive |
| 1 | Scratch |
| 2 | Corner Leak |
| 3 | Orange Peel |
| 4 | Leakage |
| 5 | Jet |
| 6 | Paint Bubble |
| 7 | Crater |
| 8 | Parti-color |
| 9 | Dirty Point |

The dataset contains defects with different scales, textures, orientations, contrasts, and morphologies, including weak-texture, elongated, small-scale, and irregular defects.

---

## 2. Recommended Directory Structure

After downloading and preparing the dataset, organize the files as follows:

```text
dataset/
├── train/
│   ├── images/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── labels/
│       ├── 000001.txt
│       ├── 000002.txt
│       └── ...
│
├── val/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
├── yolo2coco.py
└── README.md