from dataclasses import dataclass


@dataclass
class OneHot:
    start_column: int
    end_column: int
