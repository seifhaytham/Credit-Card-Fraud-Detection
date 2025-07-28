from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import joblib

def train_models(X_train, y_train):
    log_reg = LogisticRegression(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    log_reg.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    joblib.dump(log_reg, 'model1.pkl')
    joblib.dump(gb, 'model2.pkl')
    return log_reg, gb