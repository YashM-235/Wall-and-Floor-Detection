
# HybridSeg3D: Hybrid Deep Learning and Geometric Reasoning for Wall and Floor Detection

![PyTorch](https://img.shields.io/badge/framework-pytorch-red)
![Unity](https://img.shields.io/badge/synthetic--data-Unity-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A hybrid deep learning pipeline for accurate **wall and floor detection** in indoor environments using semantic segmentation and geometric reasoning. Combines **DeepLabv3+ with ResNet-101** and a custom **geometric module** for refined segmentation and **3D point cloud** generation. Trained on **real**, **augmented**, and **synthetic** data.

---

## 🚀 Key Features

- 🧠 **DeepLabv3+** with **ResNet-101** backbone for semantic segmentation
- 📐 **Geometric Reasoning Module** using primitive fitting & ray-casting
- 🌐 Trained on **NYU Depth V2**, **SUN RGB-D**, and Unity-generated synthetic data
- 🔁 Augmented with **Albumentations** for robustness
- 🔍 Outputs **refined segmentation masks** with improved boundary accuracy
- 🧊 Generates **semantically labeled 3D point clouds**
- 🖼️ Includes an **interactive demo UI with Streamlit**

---

## 📷 Example Results

| Input Image | Baseline (DeepLabv3+) | Hybrid Output | Point Cloud |
|-------------|------------------------|----------------|-------------|
| ![input](examples/input.jpg) | ![baseline](examples/baseline.jpg) | ![refined](examples/refined.jpg) | ![pc](examples/pointcloud.jpg) |

> 📈 Achieved **84.9% Wall IoU**, **90.8% Floor IoU**, and **79.4% mIoU** on the test set – outperforming HRNet and Swin-T (UPerNet).

---

## 🧩 Architecture Overview

![pipeline](examples/pipeline.png)

---

## 📁 Directory Structure

```
project-root/
├── dataset/                 # Real, synthetic, and augmented data
├── models/                  # DeepLabv3 and geometric refinement scripts
├── app.py                   # Streamlit demo
├── config.yaml              # Central config
├── train_deeplab.py         # Model training
├── livedemo.py              # Live inference
├── metrics_utils.py         # Evaluation metrics
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/yourusername/HybridSeg3D-WallFloor.git
cd HybridSeg3D-WallFloor

# Create environment and install dependencies
pip install -r requirements.txt
```

---

## 🧪 Run Demo

```bash
# Run the Streamlit app
streamlit run app.py
```

Or use:

```bash
# For real-time inference
python livedemo.py
```

---

## 📊 Training

```bash
python train_deeplab.py --config config.yaml
```

---

## 📦 Dependencies

- PyTorch & TorchVision
- OpenCV
- Albumentations
- NumPy, Matplotlib, Seaborn
- Streamlit
- (Optional) Unity (for synthetic data)
- (Optional) Open3D (for point cloud visualization)

---

## 📚 Citation

If you use this project in your research, please cite:

```
@inproceedings{HybridSeg3D2025,
  title={Hybrid Deep Semantic Segmentation and Geometric Reasoning for Enhanced Wall and Floor Detection},
  author={Karmakar, Ritish and Mehta, Yash and Kumar, Prabhat},
  year={2025},
  institution={Bennett University}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [NYU Depth V2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [SUN RGB-D Dataset](http://rgbd.cs.princeton.edu/)
- [Albumentations](https://albumentations.ai/)
- Unity Technologies for synthetic data generation
