# main.py
from models.rf_model import train_random_forest
from models.xgboost_model import train_xgboost
from models.svm_model import train_svm

if __name__ == "__main__":
    print("1 Training Random Forest Model...")
    train_random_forest()

    print("\n 2 Training XGBoost Model...")
    train_xgboost()

    print("\n 3 Training Support Vector Machine (SVM) Model...")
    train_svm()