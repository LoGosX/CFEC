from typing import List


class ValueChangeDirection:
    def __init__(self, columns: List[int], direction: str = '+'):
        self.columns = columns
        self.direction = direction