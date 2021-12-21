import numpy as np
import pandas as pd


def compare(x: pd.Series, cf: pd.Series) -> pd.DataFrame:
    df = pd.concat([x, cf], axis=1)
    df.columns = ["X", "X'"]
    df = df[df["X"] != df["X'"]]
    df["change"] = df["X'"] - df["X"]
    df = df.loc[(df["change"]/df["X"]).sort_values(ascending=False).index]
    return df

