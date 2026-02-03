# Multimodal Product Price Prediction (Text + Image)

This repository implements a deep learning-based multimodal approach to predict e-commerce product prices using both **textual descriptions** and **product images**.

The project was developed for the *Smart Product Pricing Challenge (ML Challenge 2025)*, where the goal is to predict product prices using only the provided dataset — **without any external price lookup**.

---

## 🚀 Problem Overview

Given:
- `catalog_content`: Product title + description + item pack quantity (text)
- `image_link`: URL of the product image  
- `sample_id`: Unique identifier for each product  

Task:
- Train a model to predict `price` for 75,000 test products.
- Output predictions in `test_out.csv` with:
  - `sample_id`
  - `predicted_price` (positive float)

Evaluation Metric:
- **SMAPE (Symmetric Mean Absolute Percentage Error)**

---

## 🧠 Model Approach

We use a **multimodal deep learning model** combining:

### 🔹 Text Encoder (NLP)
- **BERT (`bert-base-uncased`)**
- Fine-tuned on `catalog_content`
- Uses CLS token representation (768-dimensional embedding)

### 🔹 Image Encoder (Computer Vision)
- **EfficientNet-B0 (pretrained on ImageNet)**
- Last classification layer removed
- Produces 1280-dimensional image embeddings

### 🔹 Fusion & Regression Head
- Concatenate text + image embeddings  
- Fully connected layers:
  - 1024 → 256 → 1  
- Output: Predicted **log(price+1)**  
- Converted back to price using `exp` during inference

### 🔹 Training Setup
- Loss: **MSE on log(price)**
- Optimizer: **AdamW (lr = 2e-5)**
- Mixed Precision Training (AMP) enabled for faster GPU training
- Batch size: 8  
- Training: 3–5 epochs (tunable)
- Image caching enabled to speed up training

---


