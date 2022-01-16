from data._german_data import GermanData
from keras.models import Sequential
from keras.layers import Dense


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=61, activation='sigmoid'))
    # model.add(Dense(2, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_nn():
    german_data = GermanData('data/input_german.csv', 'data/labels_german.csv')
    model = create_model()
    model.fit(german_data.X_train, german_data.y_train, batch_size=16, epochs=18)
    scores = model.evaluate(german_data.X_test, german_data.y_test, verbose=0)
    print(scores)
    return model
