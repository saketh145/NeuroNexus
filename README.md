# NeuroNexus# Credit Card Fraud Detection

## Problem Statement
The problem statement chosen for this project is to predict fraudulent credit card transactions using machine learning models. This is a classification problem where we need to classify whether a credit card transaction is fraudulent or not based on the given transaction data.

## Dataset
The dataset contains transactions made by credit cards, where each transaction is represented by a set of features. The dataset is highly imbalanced, with fraudulent transactions making up less than 0.2% of all transactions.

## Approach
1. **Data Preprocessing**: Normalized the transaction data using StandardScaler for 'Amount' and 'Time' features.
2. **Handling Class Imbalance**: Implemented SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance issue.
3. **Model Building**: Trained two classification algorithms:
   - Logistic Regression
   - Random Forest Classifier
4. **Evaluation**: Evaluated model performance using precision, recall, F1-score, and ROC-AUC metrics.

## Results
- The Random Forest model achieved better performance with higher precision and recall for fraudulent transactions.
- Feature importance analysis revealed which transaction attributes were most predictive of fraud.
- Confusion matrix and ROC curves were used to visualize model performance.

## Technologies Used
- Python
- Libraries: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

## Future Improvements
- Implement additional models like XGBoost or Neural Networks
- Perform hyperparameter tuning to optimize model performance
- Explore other sampling techniques for handling imbalance
- Implement anomaly detection approaches

## Installation and Usage
```
git clone https://github.com/yourusername/NeuroNexus.git
cd NeuroNexus
pip install -r requirements.txt
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

## Project Structure
- `Credit_Card_Fraud_Detection.ipynb`: Main notebook containing all code and analysis
- `requirements.txt`: List of required libraries
- `README.md`: Project documentation

## Acknowledgments
- Dataset source: Kaggle Credit Card Fraud Detection dataset
- NeuroNexus Innovations for providing this internship opportunity
