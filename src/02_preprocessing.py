import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = "data/processed"
PREPROCESSOR_PATH = "models/preprocessor.joblib"


def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - PREPROCESSING")
    print("=" * 70)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_PATH)

    print("\nOriginal shape:", df.shape)

    # Remove customer identifier
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Convert target to binary
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify column types
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    # Numeric preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combined preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nProcessed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)

    # Save processed datasets
    pd.DataFrame(X_train_processed).to_csv(
        f"{PROCESSED_DIR}/X_train.csv",
        index=False
    )

    pd.DataFrame(X_test_processed).to_csv(
        f"{PROCESSED_DIR}/X_test.csv",
        index=False
    )

    y_train.to_csv(
        f"{PROCESSED_DIR}/y_train.csv",
        index=False
    )

    y_test.to_csv(
        f"{PROCESSED_DIR}/y_test.csv",
        index=False
    )

    # Save preprocessing pipeline
    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print("\nSaved:")
    print(f"  {PROCESSED_DIR}/X_train.csv")
    print(f"  {PROCESSED_DIR}/X_test.csv")
    print(f"  {PROCESSED_DIR}/y_train.csv")
    print(f"  {PROCESSED_DIR}/y_test.csv")
    print(f"  {PREPROCESSOR_PATH}")

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
