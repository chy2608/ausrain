import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
import joblib
from globals import DataSplitter, DataNormaliser


data = pd.read_csv("encoded.csv")

export_dir = r"north aus"

loc = ["Cairns", "Darwin"]

location_data = data[data["Location"].isin(loc)]

rep = 30
i = 1

agg_feat_importance = []
figL, axL = plt.subplots(figsize = (14, 12))

while i <= rep:
    
    train_pool, test_pool = DataSplitter(location_data, n_yes = None, want_test = True) #train pool now has equal split of yes/no rain

    feature_labels = (train_pool.drop(columns = ["Date", "Location", "RainTomorrow"])).columns

    X_train = train_pool.drop(columns = ["Date", "Location", "RainTomorrow"])
    
    Y_train = train_pool["RainTomorrow"]

    adapted_normaliser, X_train_norm = DataNormaliser(X_train, normaliser = None)

    X_test = test_pool.drop(columns = ["Date", "Location", "RainTomorrow"])
    Y_test = test_pool["RainTomorrow"]

    X_test_norm = DataNormaliser(X_test, adapted_normaliser)[1]

    model = keras.models.Sequential(
        [ 
            keras.layers.Dense(units = 16, activation = "relu", kernel_regularizer=keras.regularizers.L2(0.05)),
            keras.layers.Dense(units = 16, activation = "relu", kernel_regularizer=keras.regularizers.L2(0.05)),
            keras.layers.Dense(units = 1, activation = "sigmoid")
        ]
    )

    thresh = 0.7

    metrics_list = [keras.metrics.BinaryAccuracy(threshold=thresh)]

    # max_epoch, min_lr, max_lr, n = 100, 0.0002, 0.01, 4

    # def AdaptiveLearning(epoch): 
    #     if epoch == 0:
    #         return max_lr
    #     else:
    #         return (max_lr-min_lr)/(max_epoch**n)*(epoch-max_epoch)**n+min_lr
    
    # lr_scheduler = keras.callbacks.LearningRateScheduler(AdaptiveLearning)

    model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.Adam(learning_rate = 0.0002), metrics = metrics_list)

    history = model.fit(X_train_norm, Y_train, epochs = 50, verbose = 1, validation_split = 0.2) #, callbacks = [lr_scheduler])

    loss, acc = model.evaluate(X_test_norm, Y_test)

    with open(export_dir + r"\model_eval.txt", "a+") as file:
        file.write(f"{loss},{acc}\n")

    axL.plot(history.history["loss"], color = "blue", lw = 1, alpha = 0.6)
    axL.plot(history.history["val_loss"], color = "orange", lw = 1, alpha = 0.6)
    axL.plot(history.history["binary_accuracy"], color = "green", lw = 1, alpha = 0.6)

    input_tensor = tf.convert_to_tensor(X_train_norm)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = model(input_tensor)

    gradients = tape.gradient(output, input_tensor)
    if i == 1:
        agg_feat_importance = np.mean(np.abs(gradients.numpy()), axis=0)
    else:
        agg_feat_importance += np.mean(np.abs(gradients.numpy()), axis=0)

    joblib.dump(adapted_normaliser, export_dir + r"\normalisers\normaliser" + f"_{i}.bin", compress = True)
    model.save(export_dir + r"\models\model" + f"_{i}.keras")

    print("Model and normaliser saved")

    print("\nEvaluation " + str(i) + " of " + str(rep) + " done\n")

    i += 1

axL.set_xlabel("Epoch")
axL.set_ylabel("Loss")
figL.savefig(export_dir + r"\loss.png")

figF, axF = plt.subplots(figsize = (20, 15))
mean_feat_importance = agg_feat_importance / rep
feat_importance_arr = np.reshape(np.array(mean_feat_importance), (len(mean_feat_importance), 1))
feat_importance_df = pd.DataFrame(data = feat_importance_arr, columns = ["importance"], index = feature_labels).sort_values(by = ["importance"], ascending = True)
feat_importance_df.plot(kind = 'barh', ax = axF)
figF.savefig(export_dir + r"\feature_importance.png")
