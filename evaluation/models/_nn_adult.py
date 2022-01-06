from evaluation.data import AdultData
from evaluation.data._german_data import GermanData
from keras.models import Sequential
from keras.layers import Dense


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    german_data = AdultData('evaluation/data/datasets/adult.csv')
    model = create_model()
    model.fit(german_data.X_train, german_data.y_train, batch_size=16, epochs=10)
    scores = model.evaluate(german_data.X_test, german_data.y_test, verbose=0)
    print(scores)
    model.save('evaluation/models/model_adult')
