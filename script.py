import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


# Descripción del dataset en http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

def read_dataset():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, na_values=[' ?'])
    df.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital_status", "occupation",
                  "relationship", "race", "sex", "capital_gain", "capital_loss", "hour_per_week", "native_country",
                  "income"]
    print(df.head())
    return df


def handle_nulls(df):
    print(df.isna().sum())
    # Delete workclass and occupation NaNs
    df.dropna(subset=["workclass", "occupation"], inplace=True)
    print(df.isna().sum())
    # Replace native_country NaNs by UNKNOWN_OCCUPATION
    df["native_country"] = df["native_country"].fillna("UNKNOWN_OCCUPATION")
    print(df.isna().sum())


def handle_outliers(df):
    df.boxplot()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.violinplot(ax=ax, data=df)  # iris.iloc[:, 0:4]
    plt.show()

    sns.pairplot(df, hue="income", palette="husl")
    plt.show()


def discretization(df):
    print(df.dtypes)
    print(df.describe()['age'])
    # Discretizar edad por rangos de edad
    df.age = pd.cut(df.age, bins=[16, 30, 50, 65, 90], labels=["17-29", "30-49", "50-65", "+65"])
    print(df.head())


def correlation(df):
    corr = df.corr()
    print(corr)
    sns.heatmap(corr, annot=True)
    plt.show()
