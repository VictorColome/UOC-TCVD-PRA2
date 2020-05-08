import pandas as pd
import numpy as np

# Descripción del dataset en http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

if __name__ == '__main__':
    # Read dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, na_values=[' ?'])
    df.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital_status", "occupation",
                  "relationship", "race", "sex", "capital_gain", "capital_loss", "hour_per_week", "native_country",
                  "income"]
    print(df.head())

    # TODO VC: NaN values
    print(df.isna().sum())

    # TODO VC: Outliers

    # TODO VC: Store data

    # TODO VC: Correlation
