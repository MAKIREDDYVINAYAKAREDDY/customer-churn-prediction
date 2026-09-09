import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - SHAP EXPLAINABILITY")
    print("=" * 70)

    X_train = pd.read_csv(
        f"{DATA_DIR}/X_train.csv"
    )

    X_test = pd.read_csv(
        f"{DATA_DIR}/X_test.csv"
    )

    model = joblib.load(
        f"{MODEL_DIR}/xgboost_tuned.joblib"
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    # Use a representative sample for SHAP
    sample_size = min(1000, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    print(
        f"\nCalculating SHAP values for "
        f"{sample_size} customers..."
    )

    # TreeExplainer is optimized for XGBoost
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_sample
    )

    # ---------------------------------------------------------
    # Global feature importance
    # ---------------------------------------------------------

    print("\nCalculating global feature importance...")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance_df.to_csv(
        f"{REPORT_DIR}/shap_feature_importance.csv",
        index=False
    )

    print("\nTop 20 features:")
    print(
        importance_df.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ---------------------------------------------------------
    # SHAP summary bar plot
    # ---------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=20,
        show=False
    )

    plt.title(
        "Top Features Driving Customer Churn"
    )

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/shap_summary_bar.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # SHAP beeswarm plot
    # ---------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=20,
        show=False
    )

    plt.title(
        "SHAP Feature Impact on Customer Churn"
    )

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/shap_summary.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # Save SHAP values
    # ---------------------------------------------------------

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=X_sample.columns
    )

    shap_values_df.to_csv(
        f"{REPORT_DIR}/shap_values_sample.csv",
        index=False
    )

    print("\nGenerated files:")
    print(
        "  reports/shap_feature_importance.csv"
    )
    print(
        "  reports/shap_summary_bar.png"
    )
    print(
        "  reports/shap_summary.png"
    )
    print(
        "  reports/shap_values_sample.csv"
    )

    print("\nSHAP explainability completed successfully.")


if __name__ == "__main__":
    main()
