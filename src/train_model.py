import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model


from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


import os

def save_model(model, path="artifacts/model.pkl"):
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, path)
    print("\n✅ Model saved at:", path)



def calculate_business_cost(model, X_test, y_test, threshold=0.3,
                            failure_cost=50000,
                            false_alarm_cost=5000):

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total_cost = (fn * failure_cost) + (fp * false_alarm_cost)

    print("\nConfusion Matrix Values:")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)

    print("\nEstimated Business Cost: ₹", total_cost)

    return total_cost

def get_feature_importance(model, X_train):
    import pandas as pd
    
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    print("\nFeature Importance:\n")
    print(importance)
    
    return importance    