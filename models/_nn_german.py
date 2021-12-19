from data.GermanData import GermanData
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=61, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    german_data = GermanData('data/input_german.csv', 'data/labels_german.csv', valid_frac=0.01)
    print(german_data.X_train.shape)

    model = create_model()
    model.fit(german_data.X_train, german_data.y_train['Good'], batch_size=16, epochs=10)
    scores = model.evaluate(german_data.X_test, german_data.y_test['Good'], verbose=0)
    print(scores)
    model.save('models/model_german')
