import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Descripción del dataset en http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

# 1.- Descripción del DataSet
'''
    El juego de datos elegido es el Adult Data set (https://archive.ics.uci.edu/ml/datasets/adult), también conocido 
    como el "Censo de ingresos". Éste consta de 48842 observaciones y 14 variables, las cuales se dividen en 13 
    variables independientes de todo tipo (continuas, categóricas, etc) y una variable dependiente referida a los 
    ingresos con dos posibles valores: >50K y <=50K. El objetivo original del juego de datos fue predecir en base a las 
    variables dependientes si una persona cobraría más de 50K al año.
'''

# 	2.- Limpieza de datos
# CML		2.1.- Lectura del fichero
# CML		2.2.- Integración (en este caso No Aplica)
'''
    En nuestra práctica, no realizaremos integración de otras fuentes de datos. 
'''

#TODO CML Completar la parte de "Selección" cuando la conozcamos
# CML		2.3.- Selección (quedarnos solo con las columnas que nos sean de interés)
# CML		2.4.- Reducción
# 			2.4.a.- Reducción de la Dimensionalidad (aplicar PCA y ver si podemos resumir varias columnas en una sola)
# 			2.4.b.- Reducción de la cantidad (decir que se aplicará más adelante durante el análisis si es necesario)
# 		2.5.- Conversión
# CML			2.5.a.- Normalización o Estandarización (aplicar aquí lo que dijo el profe de Minería Avanzada)
# VC			2.5.b.- Discretización
# VC		2.6.- Elementos vacíos o nulos
# VC		2.7.- Valores extremos
# VC		2.8.- Guardar datos preprocesados
# 	3.- Análisis
# CML		3.1.- Análisis estadístico descriptivo
# CML		3.2.- Análisis de regresión
# VC		3.3.- Análisis de correlación
# 		3.4.- ¿Hacer clustering a alguna columna y ver si el análisis que hemos hecho no cambia (lo cual nos demostraría que una reducción de la cantidad funcionaría aquí)?
# 		3.5.- .... (sobre la marcha)
# los 2	4.- Conclusiones


def read_dataset():
    """
    Read dataset into a dataframe
    ----------
    :return: It returns a dataframe containing the dataset read from the url
    """
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # We describe the known N/A values to the parse so we can deal with them later 
    df = pd.read_csv(url, na_values=[' ?'])
    # Taken from http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
    df.columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                  "relationship", "race", "sex", "capital_gain", "capital_loss", "hour_per_week", "native_country",
                  "income"]
    print(df.head())
    return df


def normalize(df):
    """
    Normalize all the continuous columns of the DataFrame
    ----------
    :param df: The dataframe to be processed
    :return: Nothing
    """
    cNames = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hour_per_week']
    colsToNorm = df[cNames]
    scaler = StandardScaler().fit(colsToNorm.values)
    df[cNames] = scaler.transform(colsToNorm.values)


def reduce_dim(df):
    """
    Attempts to reduce dimension of dataset using PCA
    ----------
    :param df: The (already normalized) dataframe to be processed
    :return: Nothing
    """
    # Instead of choosing the number of components, we choose instead the amount of variance we want explained by the 
    # Principal Components, and let the method set the number of components that fits. We want 95% of the variance explained,
    # to keep the loss of information at minimum
    
    #TODO
    # Pregunta: ¿tiene sentido hacer PCA solo sobre las variables continuas?. En Internet parece haber abundante discusión sobre eso
    # Si lo aplicamos solo sobre continuas y nos quedamos luego con los PC, ¿no estaríamos perdiendo las relaciones entre las columnas
    # originales y las columnas no continuas que no forman parte del análisis PCA? O dicho de otra forma, PCA solo resume la información
    # entra las columnas que forman parte del mismo PCA, no de esas columnas con las demás, ¿correcto?
    # Respuesta: El PCA se ha de hacer con todo el dataset, no creo que puedas hacerlo solo de algunas columnas. Yo personalmente
    #  no haría PCA en este dataset.
    cNames = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hour_per_week']
    pca = PCA(n_components=0.95)
    pca.fit(df[cNames])
    reduced = pca.transform(df[cNames])
    print(reduced)


def handle_nulls(df: pd.DataFrame):
    """
    Deals with the NA appropiately
    ----------
    :param df: The dataframe to be processed
    :return: No return
    """
    #print("Number of NA found:"+df.isna().sum())
    # Delete workclass and occupation NaNs.  
    df.dropna(subset=["workclass", "occupation"], inplace=True)
    #print(df.isna().sum())
    # Replace native_country NaNs by UNKNOWN_OCCUPATION
    df["native_country"] = df["native_country"].fillna("UNKNOWN_OCCUPATION")
    #print(df.isna().sum())


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
    df.age = pd.cut(df.age, bins=[16, 30, 50, 65, 90], labels=[0, 1, 2, 3])  # 0:"17-29", 1:"30-49", 2:"50-65", 3:"+65"
    print(df.head())


def correlation(df):
    corr = df.corr()
    print(corr)
    sns.heatmap(corr, annot=True)
    plt.show()
