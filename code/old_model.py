import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
import shap
import seaborn as sns
import normalisation 
import shap_evaluator

data = pd.read_csv("encoded.csv")

fig, ax = plt.subplots(figsize = (14, 12))

loc = "Portland"

portland = data[data["Location"] == loc]

rep = 1
i = 1

while i <= rep:
    n_yes = int(len(portland[portland["RainTomorrow"] == 1]) * 0.9)

    yes_data = portland[portland["RainTomorrow"] == 1].sample(n = n_yes)
    no_data = portland[portland["RainTomorrow"] == 0].sample(n = n_yes)

    sample = pd.concat([yes_data, no_data]).drop(columns = ["Date", "Location", 'Unnamed: 0'])
    test_set = (data.copy()).drop(labels = list(sample.index.values)).drop(columns = ["Date", "Location", 'Unnamed: 0'])

    feature_labels = (sample.drop(columns = ["RainTomorrow"])).columns

    training_feats_norm, training_labels, test_feats_norm, test_labels, normaliser = normalisation.DataNormaliser(sample, test_set)
    

    # normaliser = make_column_transformer((
    #     StandardScaler(), ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
    #                     'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
    # ))


    # training_feats = (sample.copy()).drop(columns = ["RainTomorrow"])
    # training_labels = (sample.copy())["RainTomorrow"]

    # normaliser.fit(training_feats)

    # training_feats_norm = normaliser.transform(training_feats)
    # training_feats_norm = pd.concat([pd.DataFrame(data = training_feats_norm).reset_index(drop=True), training_feats.iloc[:, 15:].reset_index(drop=True)], axis = 1)
    # training_feats_norm.columns = training_feats.columns

    # test_feats = test_set.drop(columns = ["Date", "Location", "RainTomorrow"])
    # test_labels = test_set["RainTomorrow"]

    # test_feats_norm = normaliser.transform(test_feats)
    # test_feats_norm = (pd.concat([pd.DataFrame(data = test_feats_norm).reset_index(drop=True), test_feats.iloc[:, 15:].reset_index(drop=True)], axis = 1)).iloc[:, 1:]
    # test_feats_norm.columns = training_feats.columns

    model = keras.models.Sequential(
        [
            keras.layers.Dense(units = 40, activation = "relu"), 
            keras.layers.Dense(units = 32, activation = "relu"),
            keras.layers.Dense(units = 16, activation = "relu"),
            keras.layers.Dense(units = 1, activation = "sigmoid")
        ]
    )

    thresh = 0.7

    model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.Adam(learning_rate = 0.001), metrics = [keras.metrics.BinaryAccuracy(threshold=thresh)])

    history = model.fit(training_feats_norm, training_labels, epochs = 100, verbose = 1, validation_split = 0.2)

    loss, acc = model.evaluate(test_feats_norm, test_labels)

    with open(r"model data\xmodel_eval_" + loc + ".txt", "a+") as file:
        file.write(str(acc) + "," + str(loss) + "\n")

    #ax.plot(history.history["loss"], color = "blue", lw = 1, alpha = 0.6)
    #ax.plot(history.history["val_loss"], color = "orange", lw = 1, alpha = 0.6)

    shap_evaluator.CalculateShap(100, normaliser, sample, test_set, feature_labels, model, r"model")


    # sample_size = 100

    # shap_test_pool = test_set.drop(columns = ["Date", "Location", "Unnamed: 0"])
    # shap_test_norm = normaliser.transform(shap_test_pool)
    # shap_test_norm = pd.concat([pd.DataFrame(data = shap_test_norm).reset_index(drop=True), shap_test_pool.iloc[:, 15:].reset_index(drop=True)], axis = 1)

    # testyes =  shap_test_norm[shap_test_norm["RainTomorrow"] == 1].sample(n = int(sample_size/2)) # ensure the distribution of yesrain and norain are equal
    # testno = shap_test_norm[shap_test_norm["RainTomorrow"] == 0].sample(n = int(sample_size/2))
    # test = pd.concat([testyes, testno], ignore_index = True).drop(columns = ["RainTomorrow"])
    # test.columns = feature_labels
    # test.to_csv(r"model data\base\shap_test.csv")

    # playground = pd.DataFrame(data = training_feats_norm).sample(n = 100)

    # e = shap.KernelExplainer(model, playground)

    # feat_importance = e.shap_values(test)
    # shap_df = pd.DataFrame(data = np.reshape(feat_importance, (sample_size, 52)), columns = feature_labels)
    
    # fig2, ax2 = plt.subplots(figsize = (20, 15))
    # plot = sns.stripplot(data = shap_df, ax = ax2)
    # fig2.savefig(r"model data\base\shap.png")

    # fig3, ax3 = plt.subplots(figsize = (20, 15))
    # mean_shap = shap_df.mean()
    # mean_shap.to_csv(r"model data\base\mean_shap.csv")

    # mean_shap.plot(kind = 'barh', ax = ax3, color=(mean_shap > 0).map({True: 'g',
    #                                                                         False: 'r'}))
    
    # fig3.savefig(r"model data\base\mean_shap.png")
        
    i += 1

# ax.set_xlabel("Epoch")
# ax.set_ylabel("Loss")
# fig.savefig(r"model data\xloss.png")