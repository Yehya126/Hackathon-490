import pandas as pd

df = pd.read_csv("stories_transactions.csv")

print(df.head(10))
print(df.info())
print(df.describe(include="all"))

summary = pd.DataFrame({
    "Type": df.dtypes,
    "Missing %": df.isnull().mean() * 100,
    "Unique Values": df.nunique(),
    "Example": df.iloc[0]
})

print(summary)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("stories_transactions.csv")

# Missing values
df.isnull().sum().plot(kind="bar", figsize=(10, 4))
plt.title("Missing Values by Column")
plt.tight_layout()
plt.show()

# Numerical columns
num = df.select_dtypes(include="number")

if not num.empty:
    num.hist(figsize=(12, 8), bins=30)
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm")
    plt.title("Numerical Correlations")
    plt.tight_layout()
    plt.show()

# Categorical columns
cat = df.select_dtypes(include=["object", "category"])

for col in cat.columns:
    if df[col].nunique() <= 20:
        df[col].value_counts().plot(kind="bar", figsize=(8, 4))
        plt.title(col)
        plt.tight_layout()
        plt.show()