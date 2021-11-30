from typing import List


class MinMaxValue:
    def __init__(self, columns: List[int], min_value=None, max_value=None):
        self.columns = columns
        self.min_value = min_value
        self.max_value = max_value
