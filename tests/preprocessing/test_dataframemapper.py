import pytest

from counterfactuals.preprocessing import DataFrameMapper
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def _transform_to_numpy(df):
    df_numpy = df.copy()

    df_numpy = df_numpy.drop(columns=['b'])
    df_numpy = StandardScaler().fit_transform(df_numpy)
    return np.hstack([df_numpy, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)])


def _default_dataframe():
    return pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                        columns=['a', 'b', 'c'])


def test_dataframemapper():
    df2 = _default_dataframe()
    mapper = DataFrameMapper(nominal_columns=['b'])

    mapper.fit(x=df2)
    mapped = mapper.transform(df2)

    df_numpy = _transform_to_numpy(df2)

    assert np.array_equal(df_numpy, mapped)


def test_dataframemapper_fittransform():
    df2 = _default_dataframe()
    mapper = DataFrameMapper(nominal_columns=['b'])

    mapped = mapper.fit_transform(df2)

    df_numpy = _transform_to_numpy(df2)

    assert np.array_equal(df_numpy, mapped)


def test_all_continuous():
    df = pd.DataFrame(data=np.random.rand(3, 3))
    df_transformed = StandardScaler().fit_transform(df)

    mapper = DataFrameMapper(nominal_columns=[])

    mapper_df_transformed = mapper.fit_transform(df)

    assert np.array_equal(df_transformed, mapper_df_transformed), "Transformed arrays are not equal"

    assert df == mapper.inverse_transform(mapper_df_transformed), "Inverse-transformed arrays are not equal"
