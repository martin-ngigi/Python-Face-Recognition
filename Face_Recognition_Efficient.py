#link -> https://www.kaggle.com/code/pramod722445/face-recognition-eff
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

#To increase GPU usage (Thats if your machine support GPU)
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSeession
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_function=0.95 #or 1
# session=InteractiveSession(config=config)
#If dont use above code, Tensorflow will use only 4 GB


# style your matplotlib
mpl.style.use("seaborn-darkgrid")
# run this block

from tqdm import tqdm

if __name__ == "__main__":
    # link -> https://www.kaggle.com/code/pramod722445/face-recognition-eff
    files = os.listdir("Dataset/Images/")
    print(files)

    #--->if you want tp select only two files, rather than using all the files
    #files = ['Johnny_Galeck1', 'Martin_Wainaina']

    image_array = []  # it's a list later i will convert it to array
    label_array = []
    #path = "../input/face-recognition-30/dataset/"
    path = "Dataset/Images/"
    # loop through each sub-folder in train
    for i in range(len(files)):
        # files in sub-folder
        file_sub = os.listdir(path + files[i])

        for k in tqdm(range(len(file_sub))):
            try:
                img = cv2.imread(path + files[i] + "/" + file_sub[k])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))#We will change size of the face from 96,96, due to limitted GPU memory
                image_array.append(img)
                label_array.append(i)
            except:
                pass

    #Free some RAM memory
    import gc
    gc.collect()

    #Scale Image array and convert it into numpy array
    image_array = np.array(image_array) / 255.0
    label_array = np.array(label_array)

    #Use the split image_array and label_array into train and test array
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(image_array, label_array, test_size=0.15)

    #from tensorflow.keras import layers, callbacks, utils, applications, optimizers
    #from tensorflow.keras.models import Sequential, Model, load_model
    from keras import layers, callbacks, utils, applications, optimizers
    from keras.models import Sequential, Model, load_model

    print("Number of files = ", len(files))

    # Train model
    #Make sure internet is turned on
    model = Sequential()
    # I will use MobileNetV2 as an pretrained model
    pretrained_model = tf.keras.applications.EfficientNetB0(input_shape=(64, 64, 3), include_top=False,
                                                            weights="imagenet")  # change input file from 96, 96 - to 96,96
    model.add(pretrained_model)
    model.add(layers.GlobalAveragePooling2D())
    # add dropout to increase accuracy by not overfitting
    model.add(layers.Dropout(0.3))
    # add dense layer as final output
    model.add(layers.Dense(1))
    model.build(input_shape=(None,64,64,3))
    model.summary()

    #compile model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    # creating a chechkoint to save model at best accuarcy
    #create a trained_model
    ckp_path = "trained_model/model"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                          monitor="val_mae",
                                                          mode="auto",
                                                          save_best_only=True,
                                                          save_weights_only=True)

    # create a lr reducer which decrease learning rate when accuarcy does not increase
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.9, monitor="val_mae",
                                                     mode="auto", cooldown=0,
                                                     patience=5, verbose=1, min_lr=1e-6)
    # patience : wait till 5 epoch
    # verbose : show accuracy every 1 epoch
    # min_lr=minimum learning rate
    #

    EPOCHS = 200 #decrease Epoch from 300 to 200
    BATCH_SIZE = 64

    history = model.fit(X_train,
                        Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[model_checkpoint, reduce_lr]
                        )

    #load best model
    model.load_weights(ckp_path)

    #converter model tflite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    #see the results
    prediction_val = model.predict(X_test, batch_size=BATCH_SIZE)
    print("\nResults of prediction_val")
    #predicted value
    print(prediction_val[:20])
    #Original label
    print("\n Results of Y_test[:20]")
    print(Y_test[:20])