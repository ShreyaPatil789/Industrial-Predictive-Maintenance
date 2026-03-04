from src.data_preprocessing import prepare_data
from src.train_model import (
    train_model,
    evaluate_model,
    save_model,
    calculate_business_cost,
    get_feature_importance
)

def main():

    print("\n🚀 Starting Predictive Maintenance Pipeline...\n")

    # -----------------------
    # 1️⃣ Data Preparation
    # -----------------------
    X_train, X_test, y_train, y_test = prepare_data("data/predictive_maintenance.csv")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # -----------------------
    # 2️⃣ Model Training
    # -----------------------
    model = train_model(X_train, y_train)

    # -----------------------
    # 3️⃣ Model Evaluation
    # -----------------------
    thresholds = [0.5, 0.4, 0.3, 0.2]

    for t in thresholds:
        print("\n==============================")
        print(f"🔎 Evaluation at Threshold: {t}")
        print("==============================")
        evaluate_model(model, X_test, y_test, threshold=t)
        calculate_business_cost(model, X_test, y_test, threshold=t)

    # -----------------------
    # 4️⃣ Feature Importance
    # -----------------------
    get_feature_importance(model, X_train)

    # -----------------------
    # 5️⃣ Save Model
    # -----------------------
    save_model(model)

    print("\n✅ Pipeline Completed Successfully!\n")


if __name__ == "__main__":
    main()