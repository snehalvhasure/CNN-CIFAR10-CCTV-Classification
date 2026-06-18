# CNN Based CCTV Object Classification using CIFAR-10
## Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset and integrates the trained model with a webcam/CCTV feed for real-time object classification.

The model learns visual features such as edges, shapes, textures, and patterns through convolution operations and predicts one of the ten CIFAR-10 object classes from live camera input.

---

## Motivation
This project was developed as an improvement over my previous ANN-based CCTV Digit Recognition project.

While the ANN model achieved good accuracy on the MNIST dataset, it had several limitations when dealing with image data:

- Loss of spatial information after flattening images.
- Inability to effectively learn edges and visual patterns.
- Limited feature extraction capability.
- Less suitable for real-world computer vision tasks.

To overcome these challenges, I implemented a Convolutional Neural Network (CNN), which is specifically designed for image processing and computer vision applications.

---

## Problem with ANN Approach
In the previous ANN implementation:

- Images were flattened before being passed into the neural network.
- Pixel relationships were lost.
- Edge and texture information was not preserved.
- Real-world image understanding was limited.
- Performance could degrade when images contained noise or background variations.

These limitations motivated the transition from ANN to CNN.

---

## Why CNN?
Unlike ANN, CNN preserves image structure and automatically extracts important visual features.

Advantages of CNN:

- Preserves spatial information
- Learns edges automatically
- Learns textures and patterns
- Better feature extraction
- Improved image understanding
- Better suited for computer vision tasks
- More efficient for image classification problems

---

## Technologies Used
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

---

## Dataset

### CIFAR-10 Dataset
The model is trained using the CIFAR-10 dataset.

Dataset Information:

- 50,000 Training Images
- 10,000 Testing Images
- 10 Object Categories
- Image Size: 32 × 32 × 3 (RGB)

### Classes
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

---

## CNN Architecture

### Input Layer
- Shape: 32 × 32 × 3

### Convolution Block 1
- Conv2D (32 Filters)
- Kernel Size: 3 × 3
- Activation: ReLU
- MaxPooling2D

### Convolution Block 2
- Conv2D (64 Filters)
- Kernel Size: 3 × 3
- Activation: ReLU
- MaxPooling2D

### Convolution Block 3
- Conv2D (32 Filters)
- Kernel Size: 3 × 3
- Activation: ReLU
- MaxPooling2D

### Fully Connected Layers
- Dense (128)
- Dense (256)
- Dense (512)
- Dense (1024)

### Output Layer
- Dense (10)
- Activation: Softmax

---

## Project Workflow

### Step 1: Load Dataset
The CIFAR-10 dataset is loaded from TensorFlow.

### Step 2: Image Normalization
Pixel values are scaled from:

0 - 255

to

0 - 1

This helps the model train faster and more efficiently.

### Step 3: CNN Training
The CNN learns visual features from training images through convolution and pooling operations.

### Step 4: Model Evaluation
The trained model is evaluated on unseen test data.

### Step 5: Accuracy Visualization
Training and validation accuracy graphs are generated using Matplotlib.

### Step 6: CCTV / Webcam Integration
The webcam continuously captures frames.

Each frame is:

- Resized to 32 × 32
- Normalized
- Passed to the trained CNN
- Classified into one of the CIFAR-10 classes

### Step 7: Display Prediction
The predicted class label and confidence score are displayed directly on the live video feed.

---

## CCTV Integration
The trained CNN model is connected to a webcam feed using OpenCV.

Workflow:

Camera Frame
    ↓
Resize (32 × 32)
    ↓
Normalize
    ↓
CNN Prediction
    ↓
Class Label
    ↓
Display on Screen

Example Output:
Cat (0.92)

or

Automobile (0.85)

---

## Model Performance

The model is evaluated using:

- Accuracy
- Loss

Typical Results:

- Training Accuracy: 80%+
- Validation Accuracy: 70%+
- Test Accuracy: 70%–80%

Results may vary depending on hardware, TensorFlow version, and training configuration.

---

## ANN vs CNN Comparison

| Feature               | ANN     | CNN       |
|-----------------------|---------|-----------|
| Spatial Information   | Lost    | Preserved |
| Edge Detection        | No      | Yes       |
| Texture Learning      | No      | Yes       |
| Feature Extraction    | Manual  | Automatic |
| Image Understanding   | Limited | Better    |
| Computer Vision Tasks | Weak    | Strong    |
| CCTV Integration      | Basic   | Better    |

CNN successfully addresses several limitations observed in the ANN implementation.

---

## Project Structure
```text
CNN-CIFAR10-CCTV-Classification/
│
├── cifar10_cnn_cctv_project.py
├── README.md
├── requirements.txt
├── screenshots/
│   ├── training_accuracy.png
│   ├── model_summary.png
│   └── webcam_prediction.png
│
└── outputs/
```

## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/CNN-CIFAR10-CCTV-Classification.git
```

Move into project directory:

```bash
cd CNN-CIFAR10-CCTV-Classification
```

Install required libraries:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

## Run the Project
```bash
python cifar10_cnn_cctv_project.py
```

---

## Limitations
This project uses the CIFAR-10 dataset, which contains only ten object categories and very small images (32 × 32).

Therefore:

- Real-world webcam predictions may not always be accurate.
- Objects outside the CIFAR-10 classes may be misclassified.
- This implementation is intended for learning CNN concepts and real-time inference.

---

## Future Improvements
The project can be extended using advanced object detection models such as:

- YOLO
- SSD
- Faster R-CNN

Possible enhancements:

- Real-time object detection
- CCTV surveillance systems
- Person detection
- Vehicle detection
- Security monitoring
- Smart city applications

---

## Learning Outcomes
Through this project, I learned:

- Convolutional Neural Networks (CNN)
- Image preprocessing techniques
- Feature extraction using convolution layers
- Pooling operations
- TensorFlow and Keras
- OpenCV integration
- Real-time image classification
- CCTV/Webcam inference systems
- Deep Learning model evaluation

---

## Author
Snehal Vhasure | Full Stack Developer | AI Enthusiast | GenAI and AgenticAI Learner

This project was built as part of my Deep Learning and Computer Vision learning journey to understand how CNNs improve upon traditional ANN architectures for image-based applications.
