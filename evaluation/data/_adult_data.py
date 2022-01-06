from typing import Optional, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from counterfactuals.constraints import Freeze, OneHot, ValueMonotonicity, ValueNominal


class AdultData:
    def __init__(self, dataset_file: str, columns_to_drop: Optional[List[str]] = None, test_frac=0.2, random_state=42):

        df = pd.read_csv(dataset_file)
        self.df = df

        # fill missing values with mode
        df[df == '?'] = np.nan

        df.drop(columns=['education.num', 'fnlwgt'], inplace=True)

        for col in ['workclass', 'occupation', 'native.country']:
            df[col].fillna(df[col].mode()[0], inplace=True)

        categorical_columns = ['workclass', 'education', 'marital.status', 'occupation',
                               'relationship', 'race', 'sex', 'native.country']
        freeze_columns = ['race', 'sex', 'native.country']

        columns_to_drop = columns_to_drop if columns_to_drop is not None else []
        self.categorical_columns = [column for column in categorical_columns if column not in columns_to_drop]
        self.freeze_columns = [column for column in freeze_columns if column not in columns_to_drop]
        self.target_column = 'income'

        self.label_encoders = []
        for feature in categorical_columns + [self.target_column]:
            lb = LabelEncoder()
            self.df[feature] = lb.fit_transform(self.df[feature])
            self.label_encoders.append(lb)

        self.standard_scalers = []
        for feature in self.df.columns:
            if feature not in self.categorical_columns and feature != self.target_column:
                sc = StandardScaler()
                self.df[feature] = sc.fit_transform(self.df[feature].values.reshape(-1, 1))
                self.standard_scalers.append(sc)

        self.df.drop(columns=columns_to_drop, inplace=True)
        self.df_inputs = self.df.drop(columns=[self.target_column])
        self.df_labels = self.df[self.target_column]

        self.constraints = [
            ValueNominal(columns=self.categorical_columns), Freeze(columns=self.freeze_columns)
        ]
        self.additional_constraints = []
        if 'credit' not in columns_to_drop:
            self.additional_constraints.append(Freeze(['credit']))
        if 'age' not in columns_to_drop:
            self.additional_constraints.append(ValueMonotonicity(['age'], 'increasing'))

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.df_inputs, self.df_labels, test_size=test_frac, random_state=random_state)

    def inverse_transform(self, X: pd.DataFrame, y=None):
        for label_encoder, column in self.categorical_columns:
            X[column] = label_encoder.inverse_transform(X[column])

        if y is not None:
            if isinstance(y, pd.DataFrame):
                y[self.target_column] = self.label_encoders[-1].inverse_transform(y[self.target_column])
            else:
                y = pd.DataFrame(data=self.label_encoders[-1].inverse_transform(y), columns=[self.target_column])
            return X, y
        else:
            return X
