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

