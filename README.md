# Land-Water Segmentation Using Deep Learning

## Overview

Land-Water Segmentation is a semantic segmentation project designed to automatically identify and separate land and water regions from satellite imagery. The project leverages a U-Net architecture with a ResNet34 encoder backbone to perform pixel-level classification and generate accurate segmentation masks.

The system is capable of extracting water bodies from remote sensing data, enabling applications in environmental monitoring, flood assessment, water resource management, agricultural planning, and geospatial analysis. By utilizing deep learning techniques, the model provides a scalable and efficient alternative to traditional image processing approaches.

---

## Features

- Pixel-level land and water classification
- U-Net architecture with ResNet34 backbone
- Transfer learning for improved feature extraction
- Data augmentation for better generalization
- Automated training and validation pipeline
- Segmentation mask generation and visualization
- Support for custom satellite imagery datasets
- Performance evaluation using standard segmentation metrics

---

## Problem Statement

Accurate identification of water bodies from satellite imagery is a critical task for environmental and geospatial applications. Traditional methods often struggle with varying terrain, lighting conditions, seasonal changes, and image noise.

This project aims to develop a deep learning-based semantic segmentation system capable of automatically distinguishing water regions from land regions in satellite images with high accuracy and robustness.

---

## Dataset

The dataset consists of paired satellite images and corresponding segmentation masks.

### Input

- RGB Satellite Images
- Remote Sensing Imagery

### Output

- Binary Segmentation Masks
  - Water = 1
  - Land = 0

### Data Preprocessing

The following preprocessing steps are applied before training:

- Image resizing
- Pixel normalization
- Mask encoding
- Tensor conversion
- Dataset splitting

---

## Data Augmentation

To improve model generalization and reduce overfitting, the following augmentation techniques are used:

- Horizontal Flip
- Vertical Flip
- Random Rotation
- Random Crop
- Brightness Adjustment
- Contrast Enhancement
- Image Normalization

These transformations help the model learn robust representations from diverse satellite imagery conditions.

---

## Model Architecture

### U-Net

The segmentation model is based on the U-Net architecture, which consists of:

- Encoder Path
- Bottleneck Layer
- Decoder Path
- Skip Connections

The encoder extracts hierarchical image features, while the decoder reconstructs pixel-level segmentation masks.

### ResNet34 Encoder

A pre-trained ResNet34 backbone is used as the encoder to improve feature extraction and accelerate convergence.

Benefits include:

- Better feature representation
- Faster training
- Improved segmentation accuracy
- Enhanced generalization

---

## System Workflow

```text
Satellite Image
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
ResNet34 Encoder
        │
        ▼
U-Net Decoder
        │
        ▼
Segmentation Prediction
        │
        ▼
Binary Land-Water Mask
```

---

## Technology Stack

### Programming Language

- Python

### Deep Learning Framework

- PyTorch

### Libraries

- segmentation-models-pytorch
- TorchVision
- OpenCV
- NumPy
- Pandas
- Albumentations
- Matplotlib
- Pillow (PIL)

### Development Environment

- Google Colab
- Jupyter Notebook

---

## Project Structure

```text
Land-Water-Segmentation/
│
├── dataset/
│   ├── images/
│   └── masks/
│
├── notebooks/
│   └── land_water_segmentation.ipynb
│
├── models/
│   └── best_model.pth
│
├── outputs/
│   ├── predictions/
│   └── visualizations/
│
├── requirements.txt
├── train.py
├── inference.py
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/land-water-segmentation.git
cd land-water-segmentation
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

To train the model:

```bash
python train.py
```

Or run the Jupyter Notebook:

```bash
jupyter notebook land_water_segmentation.ipynb
```

The training pipeline includes:

1. Dataset loading
2. Data preprocessing
3. Data augmentation
4. Model initialization
5. Forward propagation
6. Loss computation
7. Backpropagation
8. Validation
9. Model checkpointing

---

## Evaluation Metrics

The model is evaluated using the following metrics:

### Dice Coefficient

Measures overlap between predicted and ground truth masks.

```text
Dice = (2 × TP) / (2 × TP + FP + FN)
```

### Intersection over Union (IoU)

Measures segmentation quality based on overlap.

```text
IoU = TP / (TP + FP + FN)
```

### Pixel Accuracy

Measures the percentage of correctly classified pixels.

```text
Pixel Accuracy = Correct Pixels / Total Pixels
```

### Validation Loss

Used to monitor model convergence and select the best-performing checkpoint.

---

## Inference

After training, the model can be used to generate segmentation masks for unseen satellite images.

Example:

```python
image = load_image("sample.jpg")

prediction = model(image)

mask = prediction > 0.5
```

Output:

- Predicted segmentation mask
- Water body extraction
- Land-water boundary visualization
- Overlay comparison with original image

---

## Applications

### Environmental Monitoring

Monitor lakes, rivers, reservoirs, and wetlands over time.

### Flood Detection

Identify flooded regions and support disaster response planning.

### Water Resource Management

Estimate water coverage and monitor resource availability.

### Agricultural Analysis

Assess irrigation infrastructure and nearby water sources.

### Urban Planning

Support land-use classification and infrastructure development.

### Coastal Monitoring

Track shoreline changes and coastal erosion.

### Geospatial Intelligence

Provide accurate segmentation data for GIS and remote sensing applications.

---

## Future Enhancements

Potential improvements include:

- Multi-class segmentation
- Transformer-based segmentation architectures
- Real-time inference pipelines
- GIS platform integration
- Cloud deployment on AWS or Azure
- Temporal change detection
- Web-based visualization dashboard
- Large-scale geospatial analytics

---

## Results

The model successfully learns spatial and contextual features from satellite imagery and generates accurate binary segmentation masks for land and water classification. The combination of U-Net architecture, transfer learning, and data augmentation enables effective segmentation performance while maintaining computational efficiency.

---

## Conclusion

This project demonstrates the effectiveness of deep learning-based semantic segmentation for automated land-water classification from satellite imagery. The developed framework provides a scalable and reliable solution for remote sensing applications and can serve as a foundation for more advanced geospatial intelligence systems.

---

## Author

**Jatin Rajani**

B.Tech Computer Science and Engineering
GLA University, Mathura

### Areas of Interest

- Deep Learning
- Computer Vision
- Remote Sensing
- Semantic Segmentation
- Geospatial Artificial Intelligence
- PyTorch Development
