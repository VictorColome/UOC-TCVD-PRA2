import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import script

# TODO CML
def dataset_description(df):
    """
    Prints several useful statistics about the dataset
    :param df: The dataframe containing the dataset
    :return: nothing
    """
    df.describe()

# Descripción del dataset en http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names


if __name__ == '__main__':
    # Read dataset
    df = script.read_dataset()

    # VC: Nuls
    print(df.isna().sum())
    script.handle_nulls(df)
    print(df.isna().sum())

    # TODO VC: Outliers
    #script.handle_outliers(df)

    # VC: Discretization
    #script.discretization(df)

    # TODO VC: Correlation
    corr = df.corr()
    print(corr)
    #sns.heatmap(corr, annot=True)
    plt.show()

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

    script.stacked_bar(df, 'sex', 'income')
    script.stacked_bar(df, 'occupation', 'income')
    script.stacked_bar(df, 'occupation', 'sex')

    script.histogram(df, 'hour_per_week', 'Male', 'Horas trabajadas a la semana')
