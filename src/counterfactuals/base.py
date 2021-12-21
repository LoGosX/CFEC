"""Base classes for all counterfactual generation models"""

from abc import ABC, abstractmethod

import pandas as pd


class CounterfactualMethod(ABC):

    @abstractmethod
    def generate(self, x: pd.Series) -> pd.DataFrame:
        """
        Generate counterfactuals for a single example x
        :param x: pd.Series of shape (n_features,)
            single example to generate counterfactuals for
        :return: pd.DataFrame of shape (n_counterfactuals, n_features)
            generated counterfactuals, one per row
        """
        pass
