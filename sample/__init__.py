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
    sns.heatmap(corr, annot=True)
    plt.show()

    #script.stacked_bar(df, 'sex', 'income', 'Income por sexo')
    #script.stacked_bar(df, 'occupation', 'income', 'Income por ocupación')
    #script.stacked_bar(df, 'occupation', 'sex', 'Sexo por ocupación')

    #script.histogram(df, 'age', 'Male', 'Edad de los hombres')
    #script.histogram(df, 'age', 'Female', 'Edad de las mujeres')

    #script.histogram(df, 'hour_per_week', 'Male', 'Horas trabajadas a la semana por hombres')
    #script.histogram(df, 'hour_per_week', 'Female', 'Horas trabajadas a la semana por mujeres')

    #script.plot_all_occupations(df)
    
    #script.plot_hours(df)

    print(df.dtypes)
    df = script.normalize(df)
    print(df.dtypes)
    script.classification(df)
