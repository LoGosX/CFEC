from src.counterfactuals.base import CounterfactualMethod

#from counterfactuals.constraints import Freeze, OneHot

class Cadex(CounterfactualMethod):
    def __init__(self):
        super(Cadex, self).__init__([0,1])
