import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    dataset = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = dataset.load_data()
    # Convert values to be between 0 and 1
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    return train_data, train_labels, test_data, test_labels


train_data, train_labels, test_data, test_labels = load_data()


def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_data, train_labels, test_data, test_labels, EPOCHS):
    return model.fit(
        train_data,
        train_labels,
        epochs=EPOCHS,
        validation_data=(test_data, test_labels),
    )


def run_model(EPOCHS):
    train_data, train_labels, test_data, test_labels = load_data()
    model = create_model()

    history = train_model(
        model, train_data, train_labels, test_data, test_labels, EPOCHS
    )

    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=1)
    print("Loss " + str(test_loss))
    print("Accuracy " + str(test_acc * 100) + "%")
    diagnosis(history)
    get_predictions(model, test_data, test_labels)


def plot_image(index, predictions, true_label, img):
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    true_label, img = true_label[index], img[index]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions),
            class_names[true_label],
        ),
        color=color,
    )


def plot_value_array(index, predictions, true_label):
    true_label = true_label[index]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


def get_predictions(model, test_data, test_labels):
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_data)
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_data)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


def diagnosis(history):
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
