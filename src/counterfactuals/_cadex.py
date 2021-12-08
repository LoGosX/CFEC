from src.constraints import ValueChangeDirection, Freeze, OneHot
from src.counterfactuals.base import CounterfactualMethod

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from typing import Union, List, Any, Optional


class Cadex(CounterfactualMethod):
    '''
    Creates a counterfactual explanation based on a pre-trained model using CADEX method
    The model has to be a Keras classifier model, where in the final classification layer, each class label must
    have a separate unit.
    '''

    def __init__(self, pretrained_model, constraints: Optional[List[Any]] = None) -> None:
        self.model = pretrained_model
        self._constraints = constraints if constraints is not None else []

        self.x = None
        self.y_expected = None
        self.y_expected_class = None
        self.mask = None
        self.C = None

    def generate(self, x: Union[pd.Series, np.ndarray], max_epoch=1000, threshold=0.5) -> Union[
        pd.DataFrame, np.ndarray]:
        self.x = tf.Variable(x, dtype=tf.float32)
        y_original = self._get_predicted_class(self.x)
        self.y_expected_class = abs(y_original - 1)
        if y_original == 0:
            self.y_expected = tf.constant([[0, 1]], dtype=tf.float32)
        else:
            self.y_expected = tf.constant([[1, 0]], dtype=tf.float32)

        opt = Adam()

        input_shape = self.x.shape[1:]
        self._initialize_mask(input_shape)
        self._initialize_C(input_shape)

        for _ in range(max_epoch):
            gradients = self._get_gradient()
            opt.apply_gradients(zip([gradients], [self.x]))
            x_corrected = self._correct_categoricals(threshold)
            if self._get_predicted_class(x_corrected) == self.y_expected_class:
                return x_corrected

    def _get_predicted_class(self, x: tf.Variable):
        return self.model(x).numpy().argmax()

    def _correct_categoricals(self, threshold):
        corrected_x = self.x.numpy()[0]
        for constraint in self._constraints:
            if isinstance(constraint, OneHot):
                # if second best is bigger than threshold than flip
                feature = corrected_x[constraint.start_column, constraint.end_column]
                sorted_features = sorted(zip(enumerate(feature)), key=lambda feat: feat[1], reverse=True)
                if sorted_features[1][1] > threshold:
                    corrected_x[constraint.start_column, constraint.end_column] = 0
                    corrected_x[constraint.start_column + sorted_features[1][0]] = 1

                else:
                    corrected_x[constraint.start_column, constraint.end_column] = 0
                    corrected_x[constraint.start_column + sorted_features[0][0]] = 1

        return tf.convert_to_tensor([corrected_x])

    def _update_mask(self, gradient):
        new_mask = self.mask.copy()
        for i in range(len(gradient)):
            if not ((self.C[i] > 0 and gradient[i] < 0) or (self.C[i] < 0 and gradient[i] > 0) or self.C[i] == 0):
                new_mask[i] = 0
        return new_mask

    def _get_gradient(self):
        with tf.GradientTape() as t:
            t.watch(self.x)
            y_pred = self.model(self.x)
            loss = tf.keras.losses.categorical_crossentropy(self.y_expected, y_pred)

        gradients = t.gradient(loss, self.x).numpy().flatten()
        updated_mask = self._update_mask(gradients)
        return tf.convert_to_tensor([gradients * updated_mask])

    def _initialize_mask(self, shape, dtype="float32") -> np.ndarray:
        self.mask = np.ones(shape, dtype=dtype)
        for constraint in self._constraints:
            if isinstance(constraint, Freeze):
                for column in constraint.columns:
                    self.mask[column] = 0

    def _initialize_C(self, shape, dtype="float32") -> np.ndarray:
        self.C = np.zeros(shape, dtype=dtype)
        for constraint in self._constraints:
            if isinstance(constraint, ValueChangeDirection):
                val = 1 if constraint.direction == "+" else -1
                for column in constraint.columns:
                    self.C[column] = val
