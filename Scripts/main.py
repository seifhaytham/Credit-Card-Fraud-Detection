from preprocessing import preprocess_data
from data_split import split_data
from model_training import train_models
from results import evaluate_models

if __name__ == "__main__":
    print("Starting script execution...")
    X, y = preprocess_data("creditcard_2023.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)
    log_reg, xgb = train_models(X_train, y_train)
    evaluate_models(X_test, y_test)
    print("Script execution completed.")