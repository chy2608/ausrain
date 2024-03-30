import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

def DataSplitter(dataframe, n_yes, want_test):
    
    if n_yes == None:
        n_yes = int(len(dataframe[dataframe["RainTomorrow"] == 1]) * 0.9)
    else:
        pass

    yes_data = dataframe[dataframe["RainTomorrow"] == 1].sample(n = n_yes)
    no_data = dataframe[dataframe["RainTomorrow"] == 0].sample(n = n_yes)

    try:
        splitted_pool = pd.concat([yes_data, no_data], axis = 0).drop(columns = ['Unnamed: 0'])
    except KeyError:
        splitted_pool = pd.concat([yes_data, no_data], axis = 0)

    if want_test == True:
        try:
            test_pool = (dataframe.drop(labels = list(splitted_pool.index.values))).drop(columns = ['Unnamed: 0'])
        except KeyError:
            test_pool = dataframe.drop(labels = list(splitted_pool.index.values))
        

        return [splitted_pool, test_pool]
    else:
        return splitted_pool

def DataNormaliser(dataframe, normaliser):

    if normaliser == None:
        normaliser = make_column_transformer((
            StandardScaler(), ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
        ))

        normaliser.fit(dataframe)
    else:
        pass

    dataframe_norm = pd.DataFrame(data = normaliser.transform(dataframe))
    dataframe_onehot = dataframe.iloc[:, 15:]

    normalised_df = pd.concat([dataframe_norm.reset_index(drop = True), dataframe_onehot.reset_index(drop = True)], axis = 1)
    normalised_df.columns = dataframe.columns

    return [normaliser, normalised_df]