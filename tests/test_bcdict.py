from __future__ import annotations

import sys
from collections.abc import Collection

import pytest

from bcdict import BCDict

try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
except ImportError:
    pass


class SimpleObj:
    """Simple helper class used in tests."""

    def __init__(self, data: str):
        self.data = data

    def upper(self) -> str:
        return self.data.upper()

    def __getitem__(self, item: int) -> str:
        return self.data[item]


@pytest.fixture
def s1():
    return SimpleObj("s1")


@pytest.fixture
def s2():
    return SimpleObj("s2")


@pytest.fixture
def data(s1, s2):
    data = {
        "A": s1,
        "B": s2,
    }
    return data


@pytest.fixture
def dict_data():
    return BCDict(
        {
            "A": {"one": 1, "two": 2},
            "B": {"three": 3, "four": 4},
        }
    )


@pytest.fixture
def num_dict():
    return BCDict(
        {
            "A": 4,
            "B": 5,
        }
    )


@pytest.fixture
def d(data):
    return BCDict(data)


def test_explicit_function_access(d):
    assert d.a.upper() == {"A": "S1", "B": "S2"}, d.a.upper()


def test_implicit_function_access(d):
    assert d.upper() == {"A": "S1", "B": "S2"}, d.upper()


def test_explicit_attribute_access(d):
    assert d.a.data == {"A": "s1", "B": "s2"}, d.a.data


def test_implicit_attribute_access(d):
    assert d.data == {"A": "s1", "B": "s2"}, d.data


def test_explicit_slicing_access(d):
    assert d.a[1] == {"A": "1", "B": "2"}, d.a[1]


def test_implicit_slicing_access(d):
    assert d[1] == {"A": "1", "B": "2"}, d[1]


def test_parent_element_access(d, s1):
    assert d["A"] is s1


def test_broadcast_slicing(dict_data):
    slice_ = {"A": "two", "B": "three"}
    assert dict_data[slice_] == BCDict({"A": 2, "B": 3})


def test_pipe_simple(d):
    assert d.pipe(lambda s: s.data) == {"A": "s1", "B": "s2"}


def test_pipe_with_args(d):
    args = ["X", {"A": "A", "B": "B"}]
    res = d.pipe(lambda s, *args: s.data + "".join(args), *args)
    assert res == {"A": "s1XA", "B": "s2XB"}, res


def test_pipe_with_kwargs(d):
    kwargs = {"z": "Z", "q": {"A": "Q1", "B": "Q2"}}
    res = d.pipe(lambda s, **kwargs: s.data + kwargs["z"] + kwargs["q"], **kwargs)
    assert res == {"A": "s1ZQ1", "B": "s2ZQ2"}, res


@pytest.mark.skipif(
    "pandas" not in sys.modules or "sklearn" not in sys.modules,
    reason="requires extra-tests dependencies",
)
def _integration_test():
    """Run a complete train/test/evaluate pipeline with `BCDict`s."""
    from pprint import pprint

    def get_random_data(datasets: Collection) -> dict[str, pd.DataFrame]:
        """Just create some random data."""
        columns = list("ABCD") + ["target"]
        dfs = {}
        for name in datasets:
            dfs[name] = pd.DataFrame(
                np.random.random((10, len(columns))), columns=columns
            )
        return dfs

    datasets = ["noord", "brabant", "limburg"]

    # make dict with three dataframes, one for each grid:
    train_dfs = BCDict(get_random_data(datasets))
    test_dfs = BCDict(get_random_data(datasets))

    features = list("ABCD")
    target = "target"

    # get X, y *for all 3 grids at once*:
    X_train = train_dfs[features]
    y_train = train_dfs[target]

    # get X, y *for all 3 grids at once*:
    X_test = test_dfs[features]
    y_test = test_dfs[target]

    # creates models for all 3 grids at once:
    # we call the `train` function on each dataframe in X_train, and pass the
    # corresponding y_train series into the function.
    def train(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
        """We use this function to train a model."""
        model = LinearRegression()
        model.fit(X, y)
        return model

    models = X_train.pipe(train, y_train)

    # Apply each model to the correct grid.
    # `models` is a BCDict.
    # When calling the `predict` function, it knows that `test_dfs` is a dict with
    # the same keys as `models`. When calling predict on each model, the corresponding
    # dataframe from `test_dfs` is passed to the function.
    preds = models.predict(X_test)

    # now we pipe all predictions and the
    scores = y_test.pipe(r2_score, preds)
    pprint(scores)
    # {'brabant': -2.2075573154836925,
    #  'limburg': -1.3066288799673251,
    #  'noord': -0.8467452520467658}

    assert list(scores.keys()) == datasets
    assert all((isinstance(v, float) for v in scores.values()))

    # Conclusion: no single for loop or dict comprehension used to train 3 models
    # predict and evaluate 3 grids :)


@pytest.mark.parametrize(
    "expr,result",
    [
        ("d+5", {"A": 9, "B": 10}),
        ("d-5", {"A": -1, "B": 0}),
        ("d*5", {"A": 20, "B": 25}),
        ("d/2", {"A": 2, "B": 2.5}),
        ("d//2", {"A": 2, "B": 2}),
        ("d%2", {"A": 0, "B": 1}),
        ("d>>2", {"A": 1, "B": 1}),
        ("d<<2", {"A": 16, "B": 20}),
        ("d**2", {"A": 16, "B": 25}),
        ("d<5", {"A": True, "B": False}),
        ("d<=5", {"A": True, "B": True}),
        ("d==5", False),  # implemented in `dict`, not overriden
        ("d!=5", True),  # implemented in `dict`, not overriden
        ("d>5", {"A": False, "B": False}),
        ("d>=5", {"A": False, "B": True}),
    ],
)
def test_operators_simple(num_dict, expr, result):
    d = num_dict  # noqa:F841
    if result in (True, False):
        assert eval(expr) is result, expr
    else:
        assert eval(expr) == result, expr


@pytest.mark.parametrize(
    "expr,result",
    [
        ("d+{'A': 1, 'B': 2}", {"A": 5, "B": 7}),
        ("d-{'A': 1, 'B': 2}", {"A": 3, "B": 3}),
        ("d*{'A': 1, 'B': 2}", {"A": 4, "B": 10}),
        ("d/{'A': 1, 'B': 2}", {"A": 4, "B": 2.5}),
        ("d//{'A': 1, 'B': 2}", {"A": 4, "B": 2}),
        ("d%{'A': 1, 'B': 2}", {"A": 0, "B": 1}),
        ("d>>{'A': 1, 'B': 2}", {"A": 2, "B": 1}),
        ("d<<{'A': 1, 'B': 2}", {"A": 8, "B": 20}),
        ("d**{'A': 0, 'B': 2}", {"A": 1, "B": 25}),
        ("d<{'A': 5, 'B': 5}", {"A": True, "B": False}),
        ("d<={'A': 3, 'B': 5}", {"A": False, "B": True}),
        ("d=={'A': 3, 'B': 5}", False),  # implemented in `dict`, not overriden
        ("d!={'A': 3, 'B': 5}", True),  # implemented in `dict`, not overriden
        ("d>{'A': 3, 'B': 5}", {"A": True, "B": False}),
        ("d>={'A': 3, 'B': 5}", {"A": True, "B": True}),
    ],
)
def test_operators_broadcast(num_dict, expr, result):
    d = num_dict  # noqa:F841
    if result in (True, False):
        assert eval(expr) is result, expr
    else:
        assert eval(expr) == result, expr
