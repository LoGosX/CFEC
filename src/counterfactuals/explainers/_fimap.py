import operator
from typing import Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import ReLU
from numpy.typing import NDArray
from tensorflow.keras.layers import Layer, Lambda, ActivityRegularization, Dense, Dropout, Input, Add, Concatenate, \
    Multiply
from sklearn.preprocessing import LabelBinarizer
from counterfactuals.base import BaseExplainer

from counterfactuals.constraints import Freeze, ValueNominal, ValueMonotonicity
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
    y = Dense(1)(y)

    s = tf.keras.Model(inputs=x, outputs=y)

    return s


def _freeze_constraints_to_mask(layer_size: int, freeze_constraints: List[Freeze], dtype='float32') \
        -> NDArray[np.float32]:
    mask = np.zeros(shape=(layer_size,), dtype=dtype)
    for constraint in freeze_constraints:
        columns = np.asarray(constraint.columns)
        mask[columns] = 1
    return mask


def _gumbel_distribution(shape):
    u_dist = K.random_uniform(tf.shape(shape), 0, 1)
    return -K.log(-K.log(u_dist + K.epsilon()) + K.epsilon())


class _GumbelSoftmax(Layer):

    def __init__(self, tau: float, num_classes: int, freeze: bool = False):
        super(_GumbelSoftmax, self).__init__()
        self.tau = tau
        self.num_classes = num_classes
        self.freeze = freeze

    def call(self, inputs, **kwargs):
        if self.freeze:
            return K.stop_gradient(K.one_hot(K.argmax(inputs), self.num_classes))
        x = inputs + _gumbel_distribution(inputs)
        x = K.softmax(x / self.tau)
        return K.stop_gradient(K.one_hot(K.argmax(x), self.num_classes))


def _get_span(inputs: tf.Tensor, start: int, end: int) -> tf.Tensor:
    return Lambda(lambda x: x[:, start:end])(inputs)


def _get_span_no_connect(start: int, end: int):
    return Lambda(lambda x: x[:, start:end])


def _get_freeze_mask(shape, constraints: List[Any], mapper: DataFrameMapper, dtype="float32") -> NDArray[np.float32]:
    mask = np.ones(shape, dtype=dtype)
    for constraint in constraints:
        if isinstance(constraint, Freeze):
            for column in constraint.columns:
                start, end = mapper.transformed_column_span(column)
                mask[start:end] = 0.
    return mask


def _build_g(input_shape,
             layers: Optional[List[tf.keras.layers.Layer]],
             one_hot_columns: List[Tuple[int, int]],
             increasing_columns: List[int],
             decreasing_columns: List[int],
             freeze_mask: NDArray[np.float32],
             l1: float, l2: float, tau: float) -> tf.keras.Model:
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
            single_columns = [Lambda(lambda _x: _x[:, i])(span) for i in range(start - last_end)]
            for i, col in enumerate(single_columns):
                if (last_end + i) in increasing_columns:
                    single_columns[i] = ReLU()(col)
                elif (last_end + i) in decreasing_columns:
                    single_columns[i] = ReLU(max_value=0, negative_slope=-1.)(col)
            span = Concatenate()(single_columns)
            span = ActivityRegularization(l1=l1)(span)
            input_span = _get_span(x, last_end, start)
            perturbed = Add()([span, input_span])
            spans.append(perturbed)

        freeze = bool(np.any(freeze_mask[0, start:end] == 0))
        if freeze:
            span = _get_span(x, start, end)
        else:
            span = _get_span(y, start, end)
        span = _GumbelSoftmax(tau, num_classes=end - start, freeze=freeze)(span)
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

    return g


class _GNet(tf.keras.Model):

    def __init__(self, output_size: int,
                 freezed_columns: List[int],
                 freezed_one_hots: List[int],
                 one_hot_columns: List[Tuple[int, int]],
                 increasing_columns: List[int],
                 decreasing_columns: List[int],
                 tau: float):
        super(_GNet, self).__init__()
        self._increasing_columns = increasing_columns
        self._decreasing_columns = decreasing_columns
        self._one_hot_columns = one_hot_columns
        self._output_size = output_size
        self._freezed_columns = freezed_columns

        self.dense1 = Dense(100, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(100, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(output_size)
        self._continuous = [Lambda(lambda x: x[:, i]) for i in freezed_columns]
        self._continuous_activations: List[Layer] = []

        for i, _ in enumerate(self._continuous_activations):
            if i in freezed_columns:
                self._continuous_activations.append(Lambda(lambda x: x * 0.))
            elif i in decreasing_columns:
                self._continuous_activations.append(ReLU(max_value=0., negative_slope=-1.))
            elif i in increasing_columns:
                self._continuous_activations.append(ReLU())

        self._one_hots = [Lambda(lambda x: x[:, start:end]) for (start, end) in one_hot_columns]
        self._gumbels_or_zero = [
            (_GumbelSoftmax(tau, end - start) if start in freezed_one_hots else Lambda(lambda x: x))
            for (start, end) in one_hot_columns
        ]

        freeze_mask = np.ones((output_size,))
        for column in freezed_columns:
            freeze_mask[column] = 0.

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)

        return self.classifier(x)


def _fit_g(s, g, x, y, epochs):
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    batch_size = x_train.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x_perturbed = g(x_batch_train, training=True)
                s_pred = s(x_perturbed, training=True)
                loss_value = loss_fn(y_batch_train, s_pred)

            g_grads = tape.gradient(loss_value, g.trainable_weights)
            optimizer.apply_gradients(zip(g_grads, g.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))


def _get_nominal_columns(constraints: List[Any]) -> List[str]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, ValueNominal):
            columns.extend(constraint.columns)
    return columns


def _get_continuous_columns(columns: List[str], nominal_columns: List[str]) -> List[str]:
    return [column for column in columns if column not in nominal_columns]


def _get_increasing_columns(constraints: List[Any], mapper: DataFrameMapper) -> List[int]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, ValueMonotonicity) and constraint.direction == 'increasing':
            columns.extend(constraint.columns)

    return [mapper.transformed_column_span(col)[0] for col in columns]


def _get_decreasing_columns(constraints: List[Any], mapper: DataFrameMapper) -> List[int]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, ValueMonotonicity) and constraint.direction == 'decreasing':
            columns.extend(constraint.columns)

    return [mapper.transformed_column_span(col)[0] for col in columns]


class Fimap(BaseExplainer):

    def __init__(self, tau: float = 0.1, l1: float = 0.01, l2: float = 0.1, constraints: Optional[List[Any]] = None,
                 s: Optional[tf.keras.Model] = None, g_layers: Optional[List[tf.keras.layers.Layer]] = None):
        self._constraints = constraints if constraints is not None else []
        self._s = s
        self._g: tf.keras.Model = None
        self._g_layers = g_layers
        self._sg: tf.keras.Model = None
        self._input_shape: Tuple[int]
        self._nominal_columns = _get_nominal_columns(self._constraints)
        self._continuous_columns: List[str]
        self._tau = tau
        self._l1 = l1
        self._l2 = l2
        self._mapper = DataFrameMapper(nominal_columns=self._nominal_columns)
        self._y_label_binarizer = LabelBinarizer()
        self._increasing_columns = _get_increasing_columns(self._constraints, self._mapper)
        self._decreasing_columns = _get_decreasing_columns(self._constraints, self._mapper)

    def fit(self, x: pd.DataFrame, y: pd.Series, epochs: int = 5, **kwargs) -> None:
        x = self._mapper.fit_transform(x)
        y = self._y_label_binarizer.fit_transform(y)
        input_shape = x.shape[1:]
        freeze_mask = _get_freeze_mask(input_shape, self._constraints, self._mapper)
        s = self._s
        if s is None:
            s = _build_s(input_shape=input_shape)
            s.fit(x, y, epochs=epochs)
        g = _build_g(input_shape=input_shape,
                     layers=self._g_layers,
                     one_hot_columns=self._mapper.one_hot_spans,
                     freeze_mask=freeze_mask,
                     increasing_columns=self._increasing_columns,
                     decreasing_columns=self._decreasing_columns,
                     l1=self._l1, l2=self._l2,
                     tau=self._tau)
        _fit_g(s, g, x, 1 - y, epochs)
        self._s = s
        self._g = g

    def generate(self, x: pd.Series) -> pd.DataFrame:
        x = self._mapper.transform(x.to_frame().T)
        perturbed = self._g.predict(x)
        return self._mapper.inverse_transform(perturbed)
