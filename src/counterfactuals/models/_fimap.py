import operator
from typing import Union, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda, ActivityRegularization, Dense, Dropout, Input, Add, Concatenate, \
    Multiply
from sklearn.preprocessing import LabelBinarizer
from counterfactuals.base import CounterfactualMethod

from counterfactuals.constraints import Freeze, OneHot, Nominal
from counterfactuals.preprocessing import DataFrameMapper


def _freeze_layers(model: tf.keras.Model) -> None:
    for layer in model.layers:
        layer.trainable = False


def _build_s(input_shape) -> tf.keras.Model:
    x = Input(shape=input_shape)
    y = Dense(200, activation='relu')(x)
    y = Dropout(0.2)(y)
    y = Dense(200, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(200, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(1, activation='sigmoid')(y)

    s = tf.keras.Model(inputs=x, outputs=y)
    s.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return s


def _freeze_constraints_to_mask(layer_size: int, freeze_constraints: List[Freeze], dtype='float32') -> np.ndarray:
    mask = np.zeros(shape=(layer_size,), dtype=dtype)
    for constraint in freeze_constraints:
        columns = np.asarray(constraint.columns)
        mask[columns] = 1
    return mask


def _gumbel_distribution(shape):
    u_dist = K.random_uniform(tf.shape(shape), 0, 1)
    return -K.log(-K.log(u_dist + K.epsilon()) + K.epsilon())


class _GumbelSoftmax(Layer):

    def __init__(self, tau: float, num_classes: int):
        super(_GumbelSoftmax, self).__init__()
        self.tau = tau
        self.num_classes = num_classes

    def call(self, inputs, **kwargs):
        x = inputs + _gumbel_distribution(inputs)
        x = K.softmax(x / self.tau)
        return K.stop_gradient(K.one_hot(K.argmax(x), self.num_classes))


def _get_span(inputs: tf.Tensor, start: int, end: int) -> tf.Tensor:
    return Lambda(lambda x: x[:, start:end])(inputs)


def _get_freeze_mask(shape, constraints: List[Any], mapper: DataFrameMapper, dtype="float32") -> np.ndarray:
    mask = np.ones(shape, dtype=dtype)
    for constraint in constraints:
        if isinstance(constraint, Freeze):
            for column in constraint.columns:
                start, end = mapper.transformed_column_span(column)
                mask[start:end] = 0.
    return mask


def _build_g(input_shape, layers: Optional[List[tf.keras.layers.Layer]], one_hot_columns: List[Tuple[int, int]],
             freeze_mask: np.ndarray,
             l1: float, l2: float, tau: float) -> Tuple[tf.Tensor, tf.Tensor, tf.keras.Model]:
    x = Input(shape=input_shape)
    if layers:
        y = x
        for layer in layers:
            y = layer(y)
    else:
        y = Dense(100, activation='relu')(x)
        y = Dropout(0.2)(y)
        y = Dense(100, activation='relu')(y)
        y = Dropout(0.2)(y)

    y = Dense(np.prod(input_shape))(y)

    # freeze columns
    freeze_mask = freeze_mask.reshape(1, -1)
    y = Multiply()([y, freeze_mask])

    spans = []
    one_hot_columns = sorted(one_hot_columns, key=operator.itemgetter(0))
    last_end = 0
    for start, end in one_hot_columns:
        if start - last_end > 0:
            # continuous span
            # add to input
            span = _get_span(y, last_end, start)
            span = ActivityRegularization(l1=l1)(span)
            input_span = _get_span(x, last_end, start)
            perturbed = Add()([span, input_span])
            spans.append(perturbed)
        span = _get_span(y, start, end)
        span = _GumbelSoftmax(tau, num_classes=end - start)(span)
        span = ActivityRegularization(l2=l2)(span)
        spans.append(span)
        last_end = end

    if spans:
        if last_end < input_shape[0]:
            span = _get_span(y, last_end, input_shape[1])
            span = ActivityRegularization(l1=l1)(span)
            input_span = _get_span(x, last_end, start)
            perturbed = Add()([span, input_span])
            spans.append(perturbed)
        y = Concatenate(axis=-1)(spans)

    g = tf.keras.Model(inputs=x, outputs=y)

    return x, y, g


def _build_sg_combined(x_g: tf.Tensor, y_g: tf.Tensor, g: tf.keras.Model, s: tf.keras.Model):
    _freeze_layers(s)

    layer = s.layers[1](y_g)  # layer 0 is the input layer, which we're replacing
    for i in range(2, len(s.layers)):
        layer = s.layers[i](layer)
    sg = tf.keras.Model(inputs=x_g, outputs=layer)

    sg.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return g, sg


def _get_nominal_columns(constraints: List[Any]) -> List[str]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, Nominal):
            columns.extend(constraint.columns)
    return columns


def _get_continuous_columns(columns: List[str], nominal_columns: List[str]) -> List[str]:
    return [column for column in columns if column not in nominal_columns]


class Fimap(CounterfactualMethod):

    def __init__(self, tau: float = 0.1, l1: float = 0.01, l2: float = 0.1, constraints: Optional[List[Any]] = None,
                 s: Optional[tf.keras.Model] = None, g_layers: Optional[List[tf.keras.layers.Layer]] = None):
        self._constraints = constraints if constraints is not None else []
        self._s = s
        self._g: tf.keras.Model = None
        self._g_layers = g_layers
        self._sg: tf.keras.Model = None
        self._input_shape = None
        self._nominal_columns = _get_nominal_columns(self._constraints)
        self._continuous_columns: List[str] = None
        self._freeze_mask: np.ndarray = None
        self._tau = tau
        self._l1 = l1
        self._l2 = l2
        self._mapper = DataFrameMapper(nominal_columns=self._nominal_columns)
        self._y_label_binarizer = LabelBinarizer()

    def fit(self, x: pd.DataFrame, y: pd.Series, epochs:int = 5, **kwargs) -> None:
        x = self._mapper.fit_transform(x)
        y = self._y_label_binarizer.fit_transform(y)
        input_shape = x.shape[1:]
        self._freeze_mask = _get_freeze_mask(input_shape, self._constraints, self._mapper)
        s = self._s
        if s is None:
            s = _build_s(input_shape=input_shape)
            s.fit(x, y, epochs=epochs)
        x_g, y_g, g = _build_g(input_shape=input_shape,
                               layers=self._g_layers,
                               one_hot_columns=self._mapper.one_hot_spans,
                               freeze_mask=self._freeze_mask,
                               l1=self._l1, l2=self._l2,
                               tau=self._tau)
        g, sg_combined = _build_sg_combined(x_g=x_g, y_g=y_g, g=g, s=s)
        sg_combined.fit(x, 1 - y, epochs=epochs)
        self._s = s
        self._g = g
        self._sg = sg_combined

    def generate(self, x: pd.Series) -> pd.DataFrame:
        x = self._mapper.transform(x.to_frame().T)
        perturbed = self._g.predict(x)
        return self._mapper.inverse_transform(perturbed)
