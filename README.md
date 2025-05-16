[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/twilJ7f4)
# 🧠 Real Estate Price Prediction with XGBoost

This project aims to build a machine learning pipeline to **train and evaluate a model using XGBoost** on a real estate dataset provided in CSV format. It is designed with modular components to handle data preprocessing, encoding, training, evaluation, and prediction.

🌐 **Live Demo on render**: [https://immo-predict.onrender.com](https://immo-predict.onrender.com)

🌐 **Live Demo on railway**: [https://immo-eliza-predict.up.railway.app](https://immo-eliza-predict.up.railway.app/)

🌐 **Live Demo on HuggingFace**: [https://huggingface.co/spaces/Fillinger66/immo-eliza-demo](https://huggingface.co/spaces/Fillinger66/immo-eliza-demo)


---

## 📁 Project Structure


├── data/                         # Raw input data (CSV files)\
│   └── *.csv\
│\
├── lib/                          # Core library code\
│   ├── encoders/\
│   │   └── TopKEncoder.py        # Custom encoder for top-K categories\
│   │\
│   ├── model/\
│   │   └── XGBoostModel.py       # Wrapper for XGBoost model\
│   │\
│   ├── DataCleaner.py            # Optional: cleaning/preprocessing logic\
│   ├── DataClustering.py         # Optional: clustering operations (e.g., KMeans)\
│   ├── DataManager.py            # File I/O and data manipulation\
│   ├── DataMetrics.py            # Model evaluation metrics (R², MAE, RMSE)\
│   ├── DataPipeline.py           # ML preprocessing pipeline (scikit-learn style)\
│\
├── model/                        # Saved model files\
│   └── *.model\
│\
├── pipeline/                     # Optional: pipeline configurations or artifacts\
│   └── *?pipeline                # (Clarify what's inside)\
│\
├── run.py                        # Main execution script\
├── requirements.txt              # Python dependencies\
└── README.md                     # Project documentation\

## Features

- **DataPipeline**: Used to create the pipeline.
- **DataManager**: Used to interact with files (load CSV, merge DataFrame columns, etc.).
- **DataMetric**: Used to get metrics like R², MAE, RMSE, etc.
- **XGBoostModel**: Used to create, train, and predict using an XGBoost model.
- **TopKEncoder**: Used as a pipeline encoder to get the top K categories to reduce the number of columns.
- **Run script**: Used to train, predict, etc., and uses DataPipeline, DataManager, and XGBoostModel.
---

## 🎯 Purpose

The goal is to create a reproducible pipeline to:

1. **Preprocess and encode** real estate dataset features
2. **Train** an XGBoost model
3. **Predict** on unseen data
4. **Evaluate** using regression metrics like R², MAE, and RMSE

---

## 🧩 Components

### `DataPipeline`

- Builds a preprocessing pipeline using scikit-learn and custom encoders.
- Handles:
  - Missing value imputation
  - Label encoding
  - Boolean transformation
  - Train/test split
- Utilizes `TopKEncoder` for categorical column compression.

---

### `DateManager`

- Responsible for file operations:
  - Loading CSV files
  - Merging column values into new derived features
- Acts as a utility class to manage I/O operations.

---

### `DataMetric`

- Calculates evaluation metrics:
  - R² (R-squared)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
- Used during validation and model performance analysis.

---

### `XGBoostModel`

- Wraps XGBoost regressor for training and inference.
- Supports:
  - Custom hyperparameters
  - Model saving/loading
  - Feature importance extraction

---

### `TopKEncoder`

- Custom categorical encoder to reduce one-hot encoding dimensionality.
- Retains only top-K frequent categories for each column.
- Reduces feature space and risk of overfitting.

---

### `run.py`

- Main script to execute the end-to-end ML pipeline:
  - Loads data using `DateManager`
  - Preprocesses via `DataPipeline`
  - Trains model using `XGBoostModel`
  - Evaluates performance with `DataMetric`

---


## 🛠 Requirements


```txt
pgeocode
pandas
numpy
scikit-learn
xgboost
tensorflow
geopy
matplotlib
plotly
```

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/run.py
```

## Author

Alexandre Kavadias

