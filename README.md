# Credit Score Classification

This repository contains a machine learning project for **credit score classification**.  
The goal is to develop accurate models to classify credit scores using a dataset that includes demographic and financial data.

## 📌 Project Overview
This project involves:
- Data preprocessing (handling missing values, encoding, and feature selection).
- Exploratory data analysis (EDA) with visualizations.
- Training and optimizing machine learning models.
- Evaluating model performance using classification metrics.

## 📂 Dataset Information
- **File Name:** `credit_score_data.csv` 
- **Source:** [Kaggle - Credit Score Classification Dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- **License:** CC0: Public Domain
- **Description:** Contains demographic and financial data such as annual income, customer age, and payment history.
- **Format:** CSV file

## 📐 Methodology

The project follows a structured machine learning pipeline:

1. **Data Cleaning:** Removal of duplicates, non-numeric values, and formatting inconsistencies.
2. **Missing Value Imputation:** Custom techniques (e.g., domain-based replacement, KNN, mean-group filling) were applied based on column context.
3. **Feature Engineering:** Extraction of binary features (e.g., loan types), credit history normalization, and handling categorical variables.
4. **Encoding:** Combination of One-Hot and Ordinal Encoding using `ColumnTransformer` from scikit-learn.
5. **Model Training:** Gradient boosting models (CatBoost, LightGBM ) and Random Forest were trained and evaluated.
6. **Hyperparameter Optimization:** Optuna was used to fine-tune model parameters for optimal performance.
7. **Visualization & Interpretation:** Correlation heatmaps, and bar charts for feature importance and model behavior insights.

## 📊 Evaluation Metrics

To ensure reliable performance assessment, the following metrics were used:

- **F1-Score:** Main metric due to class imbalance. It balances Precision and Recall effectively.
- **Accuracy:** Measures overall correct predictions but may be misleading in imbalanced datasets.
- **Precision & Recall:** Precision measures how many selected items are relevant, Recall how many relevant items are selected.
- **Confusion Matrix:** Visual representation of model performance across classes.

> 📌 Note: F1-Score is especially emphasized because it provides a better reflection of real-world performance in credit risk modeling, where false positives and negatives have different costs.

## 📈 Key Results

This project evaluated multiple machine learning models for credit score classification, using **F1-score** as the primary metric. Below are the main findings:

- **Best Performing Model:**  
  Random Forest achieved the highest average F1-score of **0.790**, outperforming CatBoost (0.749) and LightGBM (0.731) with default parameters.

- **Feature Selection Impact:**  
  Models performed better using the full feature set compared to manually selected subsets, highlighting the value of comprehensive features in credit scoring tasks.

- **Top Features (by importance):**  
  - `Outstanding_Debt`  
  - `Interest_Rate`  
  - `Credit_Mix`  
  - `Delay_from_due_date`  
  - `Credit_History_Age`  

- **Class-wise Behavior:**  
  The model was especially strong at predicting the "Standard" class but showed some overlap between "Poor" and "Standard"/"Good" classifications.

These findings demonstrate that with robust preprocessing, targeted optimization, high performance can be achieved in credit score prediction. Still the dataset was consist of large amount of problematic values so data preprocessing step took lots of time. This indicates the importance of data quality.


## 📄 Report Access

You can view the full version of the report including visualizations, methodology, and outputs as a PDF:

👉 [View Full PDF Report](./Credit_Score_Classification_Pdf_File.pdf)


## 🛠️ Dependencies
The following Python libraries are required:

| Library  | Usage |
|----------|-----------------------------------------------|
| NumPy    | Numerical computations (arrays, matrices). |
| Pandas   | Data manipulation, cleaning, and organization. |
| Matplotlib | Static visualizations for data patterns. |
| Seaborn  | Statistical data visualization. |
| Plotly & Cufflinks | Interactive visualizations for deeper insights. |
| Scikit-learn | Data preprocessing, model training, and evaluation. |
| CatBoostClassifier | Gradient boosting algorithm for classification. |
| LGBMClassifier | Efficient gradient boosting model for large datasets. |
| Optuna | Hyperparameter optimization for better performance. |
| SciPy | Statistical analysis (e.g., chi-square tests). |
| Termcolor | Color-coded console outputs for readability. |

## ⚙️ Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Experimental Setup

All experiments were conducted on a local machine with the following hardware and software specifications:

- **Device:** Acer Aspire A315-44P Laptop  
- **Processor:** AMD Ryzen 5 5500U (12 threads, ~2.1GHz)  
- **RAM:** 16 GB  
- **Operating System:** Windows 11 Pro (64-bit)  
- **Graphics:** DirectX 12 support  
- **Memory Management:** 38 GB page file  
------------------------------------------------------------------------------------------------------------------------
📊 Usage Instructions

1️⃣ Download the dataset:
Go to Kaggle Dataset.
Download the dataset and place it in the same directory as the project files.
Rename it to credit_score_data.csv if necessary.

2️⃣ Run the Jupyter Notebook:
Open the notebook in Jupyter Notebook, VS Code, or another IDE.
Make sure the dataset is correctly placed in the same directory.
Execute the cells sequentially:
Load and preprocess the dataset.
Perform exploratory data analysis (EDA).
Train and evaluate the models.

🚀 Results & Model Evaluation
Models are evaluated based on metrics like F1-score, Accuracy, Precision, and Recall.
Hyperparameter tuning using Optuna improves model performance.
Visualizations and reports are included in the notebook for deeper insights.

📜 License
This project is licensed under the MIT License – you are free to use, modify, and distribute it.

👥 Contributors
This project was developed by:
Onur Tuncay – Data Science & Machine Learning Development  
Gifty Osafo – Business Support  
Christine Njadja Noumtchuet – Business Support  

