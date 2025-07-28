from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def evaluate_models(X_test, y_test):
    log_reg = joblib.load('model1.pkl')
    gb = joblib.load('model2.pkl')
    y_pred_log = log_reg.predict(X_test)
    y_pred_xgb = gb.predict(X_test)
    
    print("Logistic Regression Report")
    print(classification_report(y_test, y_pred_log))
    
    print("XGBoost Report")
    print(classification_report(y_test, y_pred_xgb))
    
    conf_log = confusion_matrix(y_test, y_pred_log)
    conf_gb = confusion_matrix(y_test, y_pred_xgb)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_log, annot=True, cmap="Blues", fmt='d')
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(conf_gb, annot=True, cmap="Blues", fmt='d')
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    print("Confusion matrices saved as 'confusion_matrices.png'")