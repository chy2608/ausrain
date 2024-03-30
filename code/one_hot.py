import numpy as np
import pandas as pd

raw = pd.read_csv("weatherAUS.csv")

encoded = pd.get_dummies(raw, columns = ["WindDir9am" , "WindDir3pm"], dtype = int)

encoded.to_csv("encoded.csv")