from typing import Sequence
from evaluation.data import GermanData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate

from evaluation.models._kfold import kfold_accuracy


def create_model():
    # create model
    model = Sequential([
        Dense(2, activation='softmax', input_shape=(61,))
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    german_data = GermanData('evaluation/data/datasets/input_german.csv', 'evaluation/data/datasets/labels_german.csv')
    model = create_model()
    model.summary()

    acc = kfold_accuracy(create_model, *german_data.whole_data, epochs=30)
    print("KFold accuracy", acc)

    model.fit(german_data.X_train, german_data.y_train, epochs=30)
    model.save('evaluation/models/model_german')
