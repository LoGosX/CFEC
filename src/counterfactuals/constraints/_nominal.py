from dataclasses import dataclass
from typing import List, Dict


@dataclass(init=False)
class ValueNominal:
    columns: List[str]
    values: List[str]

    def __init__(self, columns: List[str], constraints: Dict[str, str]):
        self.columns = columns
        self.values = list(constraints.values())
