from typing import List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


class DataFrameMapper(BaseEstimator, TransformerMixin):
    def __init__(self, nominal_columns: List[str]):
        self._nominal_columns = nominal_columns
        self._continuous_columns: List[str] = None
        self._nominal_columns_original_positions: List[int] = None
        self._data_frame_columns = None
        self._one_hot_encoder: OneHotEncoder = None
        self._standard_scaler: StandardScaler = None
        self._original_columns: List[str] = None
        self._n_continuous_columns: int = None

    @property
    def nominal_columns(self):
        return self._nominal_columns

    @property
    def one_hot_spans(self) -> List[Tuple[int, int]]:
        if self._one_hot_encoder is None:
            return None
        spans = []
        one_hot_start = self._n_continuous_columns
        for category in self._one_hot_encoder.categories_:
            n_categories = len(category)
            spans.append((one_hot_start, one_hot_start + n_categories))
            one_hot_start += n_categories
        return spans

    @property
    def _n_one_hot_columns(self):
        if self._one_hot_encoder is None:
            return None
        return sum(len(category) for category in self._one_hot_encoder.categories_)

    def transformed_column_span(self, column: str) -> Tuple[int, int]:
        for col, span in zip(self._nominal_columns, self.one_hot_spans):
            if col == column:
                return span

        # column is continuous
        idx = self._continuous_columns.index(column)
        return idx, idx + 1

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        self._fit_transform(x, fit=True)
        return self

    def transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        return self._fit_transform(x, fit=False)

    def fit_transform(self, x: pd.DataFrame, y=None, **fit_params):
        self.fit(x)
        return self.transform(x)

    def _fit_transform(self, x: pd.DataFrame, fit: bool):
        self._original_columns = list(x.columns)
        nominal_columns_original_positions = []
        for i, column in enumerate(x.columns):
            if column in self._nominal_columns:
                nominal_columns_original_positions.append(i)

        ohe = OneHotEncoder(sparse=False) if fit else self._one_hot_encoder

        nominal_columns = x[self._nominal_columns]

        if fit:
            one_hot_encoded = ohe.fit_transform(nominal_columns)
        else:
            one_hot_encoded = ohe.transform(nominal_columns)

        x = x.drop(columns=self._nominal_columns)
        self._continuous_columns = list(x.columns)
        sc = StandardScaler() if fit else self._standard_scaler
        if fit:
            x_numpy = sc.fit_transform(x)
        else:
            x_numpy = sc.transform(x)
        self._n_continuous_columns = x_numpy.shape[1]

        self._standard_scaler = sc
        self._one_hot_encoder = ohe
        self._nominal_columns_original_positions = nominal_columns_original_positions

        if not fit:
            return np.hstack([x_numpy, one_hot_encoded])

    def inverse_transform(self, x: np.ndarray) -> pd.DataFrame:
        n_one_hot = self._n_one_hot_columns
        one_hot_columns = x[:, -n_one_hot:]
        continuous_columns = x[:, :-n_one_hot]

        reconstructed_continuous = self._standard_scaler.inverse_transform(continuous_columns)
        reconstructed_labels = self._one_hot_encoder.inverse_transform(one_hot_columns)

        reconstructed = reconstructed_continuous.astype(object)
        for i, column in enumerate(sorted(self._nominal_columns_original_positions)):
            reconstructed = np.insert(reconstructed, column, reconstructed_labels[:, i], axis=1)

        return pd.DataFrame(data=reconstructed, columns=self._original_columns)
