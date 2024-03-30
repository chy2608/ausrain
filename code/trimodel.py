import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from globals import DataSplitter, DataNormaliser
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("weatherAUSclean.csv")

export_dir = r"model data"

loc = ["MelbourneAirport", "Melbourne", "Watsonia", "Portland", "MountGambier"]

location_data = data[data["Location"].isin(loc)] #[["Pressure3pm", "Humidity3pm", "RainToday", "RainTomorrow"]]

train_pool, test_pool = DataSplitter(location_data, n_yes = None, want_test = True) #train pool now has equal split of yes/no rain

feature_labels = (train_pool.drop(columns = ["Date", "Location", "RainTomorrow"])).columns

X_train = train_pool.drop(columns = ["Date", "Location", "RainTomorrow"])

Y_train = train_pool["RainTomorrow"]

adapted_normaliser, X_train_norm = DataNormaliser(X_train, normaliser = None)

X_train_norm = X_train_norm[["Pressure3pm", "Humidity3pm", "RainToday"]]

#knn 

# initK = int(np.sqrt(len(train_pool)))
# n = 10
# step = 10
# K_cycle = np.concatenate((np.arange(initK-n*step, initK, step), np.arange(initK, initK+n*step, step)))
# K_cycle = K_cycle[K_cycle > 0]

# parameters = {'n_neighbors': list(K_cycle),
#               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#               'p': [1,2]}

# KNN = KNeighborsClassifier()
# knn_cv = GridSearchCV(KNN, parameters, cv=10, verbose = 2)
# knn_cv.fit(X_train_norm, Y_train)
# with open(export_dir + r"\model_mash.txt", "a+") as file:
#     file.write("knn")
#     file.write(f"{knn_cv.best_params_},{knn_cv.best_score_}\n")


#dec tree
# parameters = {'criterion': ['gini', 'entropy'],
#      'splitter': ['best', 'random'],
#      'max_depth': [2*n for n in range(1,10)],
#      'max_features': ['sqrt', 'log2']} 

# tree = DecisionTreeClassifier()
# tree_cv = GridSearchCV(tree,parameters,cv=10, error_score='raise', verbose = 2)
# tree_cv.fit(X_train_norm, Y_train)
# with open(export_dir + r"\model_mash.txt", "a+") as file:
#     file.write("dectree")
#     file.write(f"{tree_cv.best_params_},{tree_cv.best_score_}\n")

#svm
parameters = {
    'C':[0.01,0.1,1,10],
    'kernel' : ["linear","rbf","sigmoid"],
    'degree' : [1,3,5,7],
    'gamma' : [0.01,1,10,500]
    }
svm = SVC()
svm_cv = GridSearchCV(svm,parameters,cv=10, verbose = 3)
svm_cv.fit(X_train_norm, Y_train)
with open(export_dir + r"\model_mash.txt", "a+") as file:
    file.write("svm")
    file.write(f"{svm_cv.best_params_},{svm_cv.best_score_}\n")

#random forest
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
rfc=RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10, verbose = 2)
rfc_cv.fit(X_train_norm, Y_train)
with open(export_dir + r"\model_mash.txt", "a+") as file:
    file.write("randfor")
    file.write(f"{rfc_cv.best_params_},{rfc_cv.best_score_}\n")