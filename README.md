# Credit Card Fraud Detection: A Benchmark Study

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20LightGBM%20%7C%20Imbalanced--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
Detecting credit card fraud is challenging due to the highly imbalanced nature of the data (Fraud cases account for only **0.17%** of all transactions).

In this project, I did not just apply a single algorithm. I performed a **comprehensive benchmark comparison** between Unsupervised and Supervised Learning methods to find the optimal solution.

The project evolved from basic Anomaly Detection to advanced Data Augmentation techniques.

## Dataset
* **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Transactions:** 284,807
* **Fraud Cases:** 492
* **Key Challenge:** Extreme Class Imbalance

## Methodology & Models
I tested and compared 5 different approaches:

1.  **Isolation Forest** (Unsupervised - Anomaly Detection)
2.  **Local Outlier Factor (LOF)** (Unsupervised)
3.  **Random Forest** (Supervised - Standard)
4.  **LightGBM** (Supervised - Tuned with `is_unbalance=True`)
5.  **Random Forest + SMOTE** (Supervised + Synthetic Data Generation)

## Key Results (The Benchmark)

| Model | Type | Recall (Catch Rate) | Precision (Reliability) | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **Isolation Forest** | Unsupervised | 26% | 30% | ❌ Failed |
| **LOF** | Unsupervised | ~0% | ~0% | ❌ Failed |
| **Random Forest** | Supervised | 77% | 97% | ✅ Good Baseline |
| **LightGBM (Tuned)** | Supervised | **86%** | 2% | ⚠️ High Risk (Too many False Positives) |
| **RF + SMOTE** | Data-Centric | **85%** | **87%** | 🚀 **WINNER** |

### Visual Comparison
<img width="993" height="392" alt="image" src="https://github.com/user-attachments/assets/fcf0624b-e4f7-4ae9-8d3e-550ec7b23785" />

## Conclusion
* **Unsupervised methods** (Isolation Forest) failed to capture the complexity of fraud patterns in this dataset.
* **LightGBM** achieved the highest recall (86%) but suffered from the "Precision-Recall Trade-off", generating too many false alarms (Low Precision).
* **The Winner:** The combination of **SMOTE (Synthetic Minority Over-sampling Technique)** and **Random Forest** provided the best balance. It captured **85% of fraud cases** while maintaining **high precision**, making it the most practical model for a real-world banking system.

## Technologies
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, LightGBM, Imbalanced-Learn, Matplotlib/Seaborn.

## How to Run
1. Clone the repository.
2. Download the dataset from Kaggle and place it in the `data/` folder.
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn lightgbm imbalanced-learn matplotlib seaborn
   ```
4. Run the Jupyter Notebook

