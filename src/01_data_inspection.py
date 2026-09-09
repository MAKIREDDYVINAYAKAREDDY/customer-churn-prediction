import pandas as pd

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - DATA INSPECTION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print("\n1. Dataset shape:")
    print(df.shape)

    print("\n2. Columns:")
    for column in df.columns:
        print(f"   - {column}")

    print("\n3. First 5 rows:")
    print(df.head())

    print("\n4. Data types:")
    print(df.dtypes)

    print("\n5. Missing values:")
    print(df.isnull().sum())

    print("\n6. Duplicate rows:")
    print(df.duplicated().sum())

    print("\n7. Churn distribution:")
    print(df["Churn"].value_counts())

    print("\n8. Churn percentage:")
    print(
        df["Churn"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n9. Numerical summary:")
    print(df.describe().T)

    print("\n10. Categorical columns:")
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    for column in categorical_columns:
        print(f"   - {column}")

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()
