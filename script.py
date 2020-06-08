import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz

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


def read_dataset() -> pd.DataFrame:
    """
    Read dataset into a dataframe
    ----------
    :return: It returns a dataframe containing the dataset read from the url
    """
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # We describe the known N/A values to the parse so we can deal with them later 
    df = pd.read_csv(url, na_values=['?'], skipinitialspace=True)
    # Taken from http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
    df.columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                  "relationship", "race", "sex", "capital_gain", "capital_loss", "hour_per_week", "native_country",
                  "income"]
    print(df.head())
    return df


def normalize(df: pd.DataFrame):
    """
    Normalize all the continuous columns of the DataFrame
    ----------
    :param df: The dataframe to be processed
    :return: Nothing
    """
    # The objective variable "income" is out of these transformations
    cNamesObj = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    cNamesInt = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hour_per_week']
#    colsToNorm = df[cNames]
#    scaler = StandardScaler().fit(colsToNorm.values)
#    df[cNames] = scaler.transform(colsToNorm.values)

    # The country column has many possible values, but the most common by far is "United-States", so we will resume
    # the information transforming this variable into a True/False, and then getting the dummies. This way we keep
    # most of the information present in original dataset and prevent having too many dummy columns
    df['native_country'] = df['native_country']=='United-States'
    df[cNamesObj] = df[cNamesObj].astype('category')
    df = pd.get_dummies(df, columns=cNamesObj)

    scaler = StandardScaler().fit(df[cNamesInt].values)
    df[cNamesInt] = scaler.fit_transform(df[cNamesInt])
    return df

def reduce_dim(df: pd.DataFrame):
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
    # Delete workclass and native_country NaNs.
    df.dropna(subset=["workclass", "native_country"], inplace=True)
    #print(df.isna().sum())
    # Replace occupation NaNs by UNKNOWN_OCCUPATION
    df["occupation"] = df["occupation"].fillna("UNKNOWN_OCCUPATION")
    #print(df.isna().sum())


def handle_outliers(df: pd.DataFrame):
    fig, axes1 = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
    fig, axes2 = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
    i, j = 0, 0
    for col_name in df.columns:
        # Plot 1
        sns.boxplot(y=df[col_name], ax=axes1[i, j])
        axes1[i, j].set_title(col_name)
        # Plot 2
        sns.violinplot(data=df[col_name], ax=axes2[i, j])
        axes2[i, j].set_title(col_name)
        # Next ax
        j += 1
        if j == 3:
            i += 1
            j = 0
    plt.show()

    sns.pairplot(df, hue="income", palette="husl")
    plt.show()

    # Remove outliers
    #for col_name in outliers_columns:
    # x = df[col_name]
    # df[col_name] = x[x.between(x.quantile(.15), x.quantile(.85))] # without outliers


def discretization(df: pd.DataFrame):
    print(df.dtypes)
    print(df.describe()['age'])
    # Discretizar edad por rangos de edad
    df.age = pd.cut(df.age, bins=[16, 30, 50, 65, 90], labels=[0, 1, 2, 3])  # 0:"17-29", 1:"30-49", 2:"50-65", 3:"+65"
    print(df.head())


def correlation(df: pd.DataFrame):
    corr = df.corr()
    print(corr)
    sns.heatmap(corr, annot=True)
    plt.show()


def stacked_bar(df: pd.DataFrame, index, columns, title):
    pivot_df = df[[index, columns]].pivot_table(index=index, columns=columns, aggfunc=len)
    pivot_df.plot.bar(stacked=True, figsize=(15, 15))
    plt.title(title)
    plt.show()


def histogram(df: pd.DataFrame, column, sex, title):
    df_aux = df.loc[df['sex'] == sex][column]
    plt.hist(df_aux, label=sex)
    plt.axvline(x=df_aux.mean(), c='r', label='Media = '+str(df_aux.mean()))
    plt.title(title)
    plt.legend()
    plt.show()


def plot_all_occupations(df: pd.DataFrame):
    occupations = df['occupation'].unique()
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
    i, j = 0, 0
    for occ in occupations:
        df_occ = df.loc[df['occupation'] == occ]
        pivot_df = df_occ[['sex', 'income']].pivot_table(index='sex', columns='income', aggfunc=len)
        pivot_df.plot.bar(stacked=True, figsize=(15, 15), ax=axes[i, j])
        axes[i, j].set_title(occ)
        j += 1
        if j == 3:
            i += 1
            j = 0
    # fig.suptitle('Todas las ocupaciones por sexo e income', fontsize=16)
    # fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    plt.show()


def plot_hours(df: pd.DataFrame):
    clr = {'Male': 'firebrick', 'Female': 'blueviolet'}
    colors = df["sex"].apply(lambda x: clr[x])
    szs = {'<=50K': 50, '>50K': 100}
    sizes = df['income'].apply(lambda x: szs[x])

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(df.occupation, df.hour_per_week, sizes=sizes, c=colors, alpha=0.4)
    plt.xticks(rotation='vertical')
    plt.show()


def classification(df: pd.DataFrame):
    random_state = 0
    X = df.drop(columns='income')
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    clf = DecisionTreeClassifier(max_depth=3, random_state=random_state)
    clf.fit(X_train, y_train)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=X.columns,
                                    class_names=y.unique(),
                                    filled=True, rounded=True,
                                    special_characters=True)
    graphviz.Source(dot_data)

    print('Cross validation accuracy = {}%'.format(cross_val_score(clf, X_test, y_test, cv=5).mean() * 100))
    y_pred = clf.predict(X_test)
    print("Test accuracy : %0.2f" % (accuracy_score(y_test, y_pred) * 100))
