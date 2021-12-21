from src.constraints import ValueChangeDirection, Freeze, OneHot
from src.counterfactuals.base import CounterfactualMethod
from data.GermanData import GermanData

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow import keras

from typing import Union, List, Any, Callable, Optional


class Cadex(CounterfactualMethod):
    '''
    Creates a counterfactual explanation based on a pre-trained model using CADEX method
    The model has to be a Keras classifier model
    '''

    def __init__(self,
                 pretrained_model,
                 n_changed: int = 5,
                 max_epochs: int = 1000,
                 optimizer: Optional[tf.keras.optimizers.Optimizer] = Adam(0.01),
                 loss: Optional[tf.keras.losses.Loss] = CategoricalCrossentropy(),
                 transform: Optional[Callable[[pd.Series], pd.Series]] = None,
                 inverse_transform: Optional[Callable[[pd.Series], pd.Series]] = None,
                 constraints: Optional[List[Any]] = None) -> None:

        self._constraints = constraints if constraints is not None else []
        self._opt = optimizer
        self._loss = loss
        self._model = pretrained_model
        self._max_epochs = max_epochs
        self._n_changed = n_changed

        self._transform = transform
        self._inverse_transform = inverse_transform

        self._mask = None
        self._C = None
        self._columns = None
        self._dtype = None

    def generate(self, x: pd.Series) -> Union[pd.Series, None]:
        x = self._transform_input(x)
        cf = self._gradient_descent(x)
        return self._inverse_transform_input(cf)

    def _gradient_descent(self, x: tf.Variable) -> tf.Variable:
        y = self._get_predicted_class(x)
        y_expected = np.array([0, 1]) if y == 0 else np.array([1, 0])

        input_shape = x.shape[1:]
        gradient = self._get_gradient(x, y_expected)
        self._initialize_mask(input_shape, gradient)
        self._initialize_C(input_shape)

        for _ in range(self._max_epochs):
            gradient = self._get_gradient(x, y_expected)
            updated_mask = self._update_mask(gradient)
            gradient = tf.convert_to_tensor(gradient * updated_mask)
            self._opt.apply_gradients(zip([gradient], [x]))
            corrected_input = self._correct_categoricals(x)
            if self._get_predicted_class(corrected_input) == np.argmax(y_expected):
                return corrected_input

    def _get_predicted_class(self, x: tf.Variable):
        return np.argmax(self._model(x), axis=1)[0]

    def _get_gradient(self, x, y_true):
        with tf.GradientTape() as t:
            t.watch(x)
            y_pred = self._model(x)
            loss = self._loss(tf.constant([y_true]), y_pred)

        return t.gradient(loss, x)

    def _update_mask(self, gradient):
        new_mask = self._mask.copy()
        for i in range(len(gradient)):
            if not ((self.C[i] > 0 and gradient[i] < 0) or (self.C[i] < 0 and gradient[i] > 0) or self.C[i] == 0):
                new_mask[i] = 0
        return new_mask

    def _correct_categoricals(self, x) -> tf.Variable:
        corrected_x = x.numpy()[0]
        for constraint in self._constraints:
            if isinstance(constraint, OneHot):
                feature = corrected_x[constraint.start_column, constraint.end_column]
                print(feature)
                max_feature = np.argmax(feature)
                print(max_feature)
                corrected_x[constraint.start_column, constraint.end_column] = 0
                corrected_x[constraint.start_column + max_feature] = 1

        return tf.convert_to_tensor([corrected_x])

    def _transform_input(self, x: pd.Series) -> tf.Variable:
        if self._transform:
            self._transform(x)
        self._columns = list(x.index)
        self._dtype = x.dtype
        return tf.Variable(x.to_numpy()[np.newaxis, :], dtype=self._dtype)

    def _inverse_transform_input(self, x: Union[tf.Variable, None]) -> Union[pd.Series, None]:
        if x is None:
            return
        x = pd.Series(data=x.numpy()[0], index=self._columns, dtype=self._dtype)
        if self._inverse_transform:
            return self._inverse_transform(x)
        return x

    def _initialize_mask(self, shape, gradient, dtype="float32"):
        self._mask = np.ones(shape, dtype=dtype)
        for constraint in self._constraints:
            if isinstance(constraint, Freeze):
                for column in constraint.columns:
                    self._mask[column] = 0

        indices = np.argsort(gradient)[::-1][0]
        count = 0
        for i in indices:
            if count < self._n_changed:
                if self._mask[i] == 1:
                    count += 1
            else:
                self._mask[i] = 0

    def _initialize_C(self, shape, dtype="float32"):
        self.C = np.zeros(shape, dtype=dtype)
        for constraint in self._constraints:
            if isinstance(constraint, ValueChangeDirection):
                val = 1 if constraint.direction == "+" else -1
                for column in constraint.columns:
                    self.C[column] = val


if __name__ == "__main__":
    german_data = GermanData('data/input_german.csv', 'data/labels_german.csv')
    test_X = german_data.X_test.iloc[6]
    model = keras.models.load_model('models/model_german')
    cadex = Cadex(model)
    cf = cadex.generate(test_X)
