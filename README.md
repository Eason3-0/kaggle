# Kaggle Projects

A collection of my Kaggle competition solutions.

## Projects

### [Store Sales — Time Series Forecasting](store-sales/)

Predict daily sales for 54 stores × 33 product families across Ecuador.

| Metric | Value |
|--------|-------|
| **Score** | RMSLE 0.49477 |
| **Model** | XGBoost with 155 engineered features |
| **Key Techniques** | LinearRegression trend, Fourier seasonality, lag/rolling features, holiday matching |

[→ Project Details](store-sales/)

---

### [Petals to the Metal — Flower Classification](petals-to-the-metal/)

Classify 104 flower species from TFRecord images.

| Metric | Value |
|--------|-------|
| **Score** | val acc 75.24% |
| **Model** | GoogLeNet (Inception v1) from scratch, 7.1M params |
| **Key Techniques** | BatchNorm, RandomRotation, AdamW + CosineAnnealing, checkpoint resume |

[→ Project Details](petals-to-the-metal/)

---

## Structure

Each project lives in its own subdirectory with a standard 4-step pipeline:

```
project-name/
├── code/           # Python scripts + Jupyter notebooks
├── data/           # raw/ → processed/ → features/
├── models/         # Trained model files
├── submissions/    # Kaggle submission files
└── README.md       # Project-specific documentation
```
