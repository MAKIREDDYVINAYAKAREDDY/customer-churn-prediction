import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIGURES_DIR = "reports/figures"


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, filename),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Clean TotalCharges for analysis
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["ChurnFlag"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Overall churn
    # ---------------------------------------------------------
    print("\n1. Overall churn rate:")
    churn_rate = df["ChurnFlag"].mean() * 100
    print(f"{churn_rate:.2f}%")

    # ---------------------------------------------------------
    # 2. Churn by contract
    # ---------------------------------------------------------
    print("\n2. Churn rate by contract:")
    contract_churn = (
        df.groupby("Contract")["ChurnFlag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    print(contract_churn.round(2))

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=contract_churn.index,
        y=contract_churn.values
    )
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract")
    plt.ylabel("Churn Rate (%)")
    save_plot("churn_by_contract.png")

    # ---------------------------------------------------------
    # 3. Churn by internet service
    # ---------------------------------------------------------
    print("\n3. Churn rate by internet service:")
    internet_churn = (
        df.groupby("InternetService")["ChurnFlag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    print(internet_churn.round(2))

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=internet_churn.index,
        y=internet_churn.values
    )
    plt.title("Churn Rate by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Churn Rate (%)")
    save_plot("churn_by_internet_service.png")

    # ---------------------------------------------------------
    # 4. Churn by payment method
    # ---------------------------------------------------------
    print("\n4. Churn rate by payment method:")
    payment_churn = (
        df.groupby("PaymentMethod")["ChurnFlag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    print(payment_churn.round(2))

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=payment_churn.index,
        y=payment_churn.values
    )
    plt.title("Churn Rate by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=25, ha="right")
    save_plot("churn_by_payment_method.png")

    # ---------------------------------------------------------
    # 5. Churn by tenure
    # ---------------------------------------------------------
    print("\n5. Tenure statistics by churn:")
    print(
        df.groupby("Churn")["tenure"]
        .describe()
        .round(2)
    )

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure"
    )
    plt.title("Tenure Distribution by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (months)")
    save_plot("tenure_vs_churn.png")

    # ---------------------------------------------------------
    # 6. Monthly charges
    # ---------------------------------------------------------
    print("\n6. Monthly charges by churn:")
    print(
        df.groupby("Churn")["MonthlyCharges"]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges"
    )
    plt.title("Monthly Charges by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    save_plot("monthly_charges_vs_churn.png")

    # ---------------------------------------------------------
    # 7. Total charges
    # ---------------------------------------------------------
    print("\n7. Total charges by churn:")
    print(
        df.groupby("Churn")["TotalCharges"]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn",
        y="TotalCharges"
    )
    plt.title("Total Charges by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Total Charges")
    save_plot("total_charges_vs_churn.png")

    # ---------------------------------------------------------
    # 8. Senior citizen
    # ---------------------------------------------------------
    print("\n8. Churn rate by senior citizen status:")
    senior_churn = (
        df.groupby("SeniorCitizen")["ChurnFlag"]
        .mean()
        .mul(100)
    )
    print(senior_churn.round(2))

    # ---------------------------------------------------------
    # 9. Gender
    # ---------------------------------------------------------
    print("\n9. Churn rate by gender:")
    gender_churn = (
        df.groupby("gender")["ChurnFlag"]
        .mean()
        .mul(100)
    )
    print(gender_churn.round(2))

    # ---------------------------------------------------------
    # 10. Save EDA summary
    # ---------------------------------------------------------
    summary = pd.DataFrame({
        "Metric": [
            "Total Customers",
            "Overall Churn Rate (%)",
            "Average Tenure",
            "Average Monthly Charges",
            "Average Total Charges"
        ],
        "Value": [
            len(df),
            round(churn_rate, 2),
            round(df["tenure"].mean(), 2),
            round(df["MonthlyCharges"].mean(), 2),
            round(df["TotalCharges"].mean(), 2)
        ]
    })

    summary.to_csv(
        "reports/eda_summary.csv",
        index=False
    )

    print("\nSaved figures to:")
    print(FIGURES_DIR)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
