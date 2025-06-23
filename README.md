# ML-for-vegetation-classification

![Dylan_Love_SYMPOSIUM_PRES_page-0001 (1)](https://github.com/user-attachments/assets/846d9708-9d3e-4934-8243-8b65af8d8594)

---

# 🌿 Crown Feature Classification with CNN

This project performs classification of tree crowns using spatially aware 2D tensor representations of remote sensing features, followed by training a Convolutional Neural Network (CNN) for predicting functional types (`pft`) of tree crowns.

---

## 📄 Overview

* Loads and filters a dataset of crown-level features
* Preprocesses data using PCA and scaling
* Groups data by individual crown (`shapeID`)
* Reshapes each crown’s features into 2D tensors based on spatial (`x`, `y`) coordinates
* Trains a CNN on these image-like tensors to classify plant functional types
* Visualizes training metrics and CNN filters

---

## 📁 Input Data

`ref_labelled_crowns-extracted_features_inv.csv`

Each row represents a pixel tied to a crown and contains:

* Spectral and structural features (columns starting with `X`)
* Positional coordinates: `x`, `y`
* Metadata: `shapeID`, `indvdID`, `pft`, etc.

---

## 🔧 Preprocessing

1. **Filter Rare Crowns**:

   * Only keep crowns (`shapeID`) that appear more than 9 times.

2. **Scale & Reduce Dimensionality**:

   * Use `StandardScaler` to normalize spectral features.
   * Apply `PCA` to retain 99% variance.

3. **Merge PCA results back into the main DataFrame**:

   * Original features are dropped.
   * Replaces high-dimensional input with compressed PCs.

---

## 🧱 Tensor Construction

The data is grouped by `shapeID` and reshaped into spatial grids:

* Unique `x`, `y` coordinates form a meshgrid.
* A 3D tensor is built for each crown: `(height, width, num_features)`
* These tensors are used as CNN inputs.

Each tensor maps pixel-level features across the spatial crown layout.

---

## 📦 Custom Dataset Class

A custom PyTorch `Dataset`:

* Loads 3D tensors
* Extracts labels (`pft`)
* Handles padding or resizing if needed (e.g., into 10×10)

---

## 🧠 CNN Model

A simple PyTorch CNN model:

* Two `Conv2d` layers with ReLU and MaxPool
* Flattened and passed through fully connected layers
* Outputs logits for multiclass `pft` classification

---

## 📊 Training & Evaluation

* Model is trained using Adam optimizer and CrossEntropy loss.
* Tracks accuracy and loss per epoch.
* Includes `plot_training_history()` to visualize metrics.
* `visualize_filters()` displays learned convolutional kernels.

---

## 📈 Example Outputs

* Bar chart showing `pft` distribution
* Training curves for accuracy and loss
* Sample predictions and CNN filters

---

## 🧪 Requirements

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

---

## 🚀 How to Run

1. Place the CSV in the correct path or update the notebook path accordingly.
2. Run all cells in the Jupyter notebook.
3. Tensors will be created and the CNN will be trained and evaluated.

---

## 📌 Notes

* The CNN input comes from spatially structured tensors, not raw CSV rows.
* Padding or resizing is applied for consistency across samples.
* This notebook merges tabular preprocessing and image-based modeling.

---
