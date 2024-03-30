import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from globals import DataSplitter, DataNormaliser
import seaborn as sns
from mpl_toolkits import mplot3d


data = pd.read_csv("weatherAUSclean.csv")

export_dir = r"model data\base"

loc = ["MelbourneAirport", "Melbourne", "Watsonia", "Portland", "MountGambier"]

location_data = data[data["Location"].isin(loc)]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colourmap = {1: "C1", 0: "C0"}

train_pool, test_pool = DataSplitter(location_data, n_yes = None, want_test = True)
category = train_pool[["Pressure3pm", "Humidity3pm", "RainToday", "RainTomorrow"]]
group = list(category["RainTomorrow"])
for g in np.unique(group):
    i = np.where(group == g)
    ax.scatter(category["Pressure3pm"].to_numpy()[i], category["Humidity3pm"].to_numpy()[i], category["RainToday"].to_numpy()[i], label=g, c = colourmap[g], alpha=0.4, edgecolors = None)

ax.set_xlabel("Pressure3pm")
ax.set_ylabel("Humidity3pm")
ax.set_zlabel("RainToday")
ax.legend()

plt.show()