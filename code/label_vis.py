import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 

raw = pd.read_csv(r"label_skew.csv")

n = 20

fig, ax = plt.subplots(figsize = (15, 10))

sns.histplot(ax = ax, data = raw, x = "yes_percentage", kde = True, bins = np.linspace(0, 1 + 1/n, n))

plt.savefig(r"figs\data_skew.png")

