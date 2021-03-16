import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import keras


class MyModel():

    def __init__(self, dimentions):

        self.model = keras.models.Sequential()
        self.model.add(keras.Input(shape=(dimentions[0],)))
        for l in dimentions[1:-1]:
            self.model.add(keras.layers.Dense(l, activation='relu'))
        self.model.add(keras.layers.Dense(dimentions[-1], activation='sigmoid'))


    def summary(self):
        self.model.summary()

    def compile(self):
        loss_fn = keras.losses.CategoricalCrossentropy()

        self.model.compile(  optimizer="adam",
                        loss=loss_fn,
                        # metrics="accuracy",
                    )
        self.trainable_variables = self.model.trainable_variables

    def set_training_settings(self, iter=1, epochs=1, batch_size=1, vali_split=0.1):
        self.iter       = iter
        self.epochs     = epochs
        self.batch_size = batch_size
        self.vali_split = vali_split

    def train(self, x, y):
        print(x.shape, x.shape)
        self.model.fit( x,
                        y,
                        epochs          =self.epochs,
                        batch_size      =self.batch_size,
                        validation_split=self.vali_split)

    def predict(self, x):
        return self.model.predict(x)

    def eval(self, x , y):
        import numpy as np
        pred = self.model.predict(x)
        print(pred)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        print("################")
        print(f"Accuracy of {np.sum(pred==y)/y.shape[0]/y.shape[1]}")
        print("################")

    def save(self, name = "default.h5"):
        self.model.save(name)

    def load(self, name = "default.h5"):
        self.model = keras.models.load_model(name)
