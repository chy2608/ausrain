import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import joblib
from globals import DataNormaliser

data = pd.read_csv("encoded.csv")

loc_list = ["Portland"]

for location in loc_list:
    predict_loc = location
    import_dir = r"model data\moredata_reg_0.05_lr_0.0002"
    export_dir = r"predictions"

    predict_data = data[data["Location"] == predict_loc]

    rep = 30
    i = 1

    while i <= rep:
        adapted_normaliser = joblib.load(import_dir + r"\normalisers\normaliser" + f"_{i}.bin")
        model = keras.models.load_model(import_dir + r"\models\model" + f"_{i}.keras")

        try:
            predict_X = predict_data.drop(columns = ["Date", "Location", "RainTomorrow", "Unnamed: 0"])
        except KeyError:
            predict_X = predict_data.drop(columns = ["Date", "Location", "RainTomorrow"])

        predict_Y = predict_data["RainTomorrow"]
        predict_X_norm = DataNormaliser(predict_X, adapted_normaliser)[1]

        loss, acc, recall, prec = model.evaluate(predict_X_norm, predict_Y)
        with open(export_dir + r"\predict_eval"+ f"_{location}.txt", "a+") as file:
            file.write(f"{loss},{acc},{recall},{prec}\n")

        print("\nEvaluation " + str(i) + " of " + str(rep) + f" done.\n")  

        i += 1
    
    print(f"Location {loc_list.index(location)+1} of {len(loc_list)} done. \n")