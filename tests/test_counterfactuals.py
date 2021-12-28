import pytest

from counterfactuals.base import BaseExplainer


def test_always_pass():
    assert True


def test_abc_counterfactual_method():
    with pytest.raises(TypeError):
        "CounterfactualMethod should be an abstract class"
        _ = BaseExplainer()
