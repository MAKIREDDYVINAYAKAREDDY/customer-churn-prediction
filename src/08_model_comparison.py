import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - MODEL COMPARISON")
    print("=" * 70)

    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

    y_train = pd.read_csv(
        f"{DATA_DIR}/y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        f"{DATA_DIR}/y_test.csv"
    ).squeeze()

    # ---------------------------------------------------------
    # Load existing models
    # ---------------------------------------------------------

    models = {
        "Logistic Regression": joblib.load(
            f"{MODEL_DIR}/logistic_regression.joblib"
        ),
        "Random Forest": joblib.load(
            f"{MODEL_DIR}/random_forest.joblib"
        ),
        "Gradient Boosting": joblib.load(
            f"{MODEL_DIR}/gradient_boosting.joblib"
        ),
        "XGBoost": joblib.load(
            f"{MODEL_DIR}/xgboost.joblib"
        ),
    }

    # ---------------------------------------------------------
    # Evaluate models
    # ---------------------------------------------------------

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        results.append(
            evaluate_model(
                name,
                model,
                X_test,
                y_test
            )
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_df.to_csv(
        f"{REPORT_DIR}/model_comparison.csv",
        index=False
    )

    # ---------------------------------------------------------
    # ROC Curve
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 7))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        f"{REPORT_DIR}/roc_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # XGBoost Hyperparameter Tuning
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("=" * 70)

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.05],
        "min_child_weight": [1, 3],
    }

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print("\nRunning GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"\nBest cross-validation ROC-AUC: "
        f"{grid_search.best_score_:.4f}"
    )

    best_xgb = grid_search.best_estimator_

    tuned_results = evaluate_model(
        "Tuned XGBoost",
        best_xgb,
        X_test,
        y_test
    )

    print("\n" + "=" * 70)
    print("TUNED XGBOOST TEST RESULTS")
    print("=" * 70)

    for metric, value in tuned_results.items():
        if metric != "Model":
            print(f"{metric:10s}: {value:.4f}")

    # ---------------------------------------------------------
    # Save tuned model
    # ---------------------------------------------------------

    tuned_model_path = (
        f"{MODEL_DIR}/xgboost_tuned.joblib"
    )

    joblib.dump(
        best_xgb,
        tuned_model_path
    )

    print(
        f"\nTuned model saved to: "
        f"{tuned_model_path}"
    )

    # Save final comparison
    final_results = pd.concat(
        [
            results_df,
            pd.DataFrame([tuned_results])
        ],
        ignore_index=True
    )

    final_results.to_csv(
        f"{REPORT_DIR}/final_model_comparison.csv",
        index=False
    )

    print("\nFinal comparison:")
    print(
        final_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nModel comparison completed successfully.")


if __name__ == "__main__":
    main()
