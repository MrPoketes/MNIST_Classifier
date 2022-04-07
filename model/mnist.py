import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt


class MNISTModel:
    def __init__(self, epochs: int, batch_size=128, hidden_units=256, dropout=0.45):
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.dropout = dropout

    def load_data(self):
        dataset = tf.keras.datasets.mnist
        (train_data, train_labels), (test_data, test_labels) = dataset.load_data()

        # Get the number of unique labels
        self.num_labels = len(np.unique(test_labels))

        image_size = train_data.shape[1]
        self.input_size = image_size * image_size

        # Convert labels to 1-hot-encoding
        self.train_y = to_categorical(train_labels, self.num_labels)
        self.test_y = to_categorical(test_labels, self.num_labels)

        # resize and normalize data for training and testing
        self.train_x = np.reshape(
            train_data,
            [-1, self.input_size],
        )
        self.train_x = self.train_x.astype("float32") / 255

        self.test_x = np.reshape(test_data, [-1, self.input_size])
        self.test_x = self.test_x.astype("float32") / 255

    def create_model(self):
        self.model = Sequential(
            [
                Dense(self.hidden_units, input_dim=self.input_size, activation="relu"),
                Dropout(self.dropout),
                Dense(self.hidden_units, activation="relu"),
                Dropout(self.dropout),
                Dense(self.num_labels, activation="softmax"),
            ]
        )

    def run_model(self):
        self.load_data()
        self.create_model()
        self.summary()
        # self.plot_model()

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        history = self.model.fit(
            self.train_x, self.train_y, self.batch_size, self.epochs, verbose="1"
        )
        # self.plot_training(history)
        self.evaluate_model()

    def evaluate_model(self):
        loss, acc = self.model.evaluate(
            self.test_x, self.test_y, self.batch_size, verbose=0
        )
        print("Test accuracy: %.1f%%", (100 * acc))
        print("Test loss: %.1f%%", loss)

    def summary(self):
        self.model.summary()

    def plot_model(self):
        plot_model(self.model, show_shapes=True)

    def plot_training(self, history):
        # Loss
        plt.subplot(211)
        plt.title("Cross entropy loss")
        plt.plot(history.history["loss"], color="blue", label="train")
        plt.plot(history.history["val_loss"], color="green", label="test")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        # Accuracy
        plt.subplot(211)
        plt.title("Accuracy")
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="test")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()