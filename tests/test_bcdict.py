from __future__ import annotations

import re
import sys
from collections.abc import Collection

import pytest

import bcdict
from bcdict import BCDict
from bcdict.bcdict import to_list

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


def test_conflicting_item_access():
    d = BCDict({1: "Hello", 2: "World"})
    assert d[1] == "Hello"


def test_conflicting_item_access_with_accessor():
    d = BCDict({1: "Hello", 2: "World"})
    assert d.a[1] == {1: "e", 2: "o"}


def test_elements_with_attribute_a():
    class Obj:
        def __init__(self):
            self.a = "this is A"

    d = BCDict({"A": Obj(), "B": Obj()})
    assert isinstance(d.a, BCDict.DictAccessor)
    assert d.a.a == {"A": "this is A", "B": "this is A"}


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
    assert isinstance(res, BCDict)
    assert res == {"A": "s1XA", "B": "s2XB"}, res


def test_pipe_with_kwargs(d):
    kwargs = {"z": "Z", "q": {"A": "Q1", "B": "Q2"}}
    res = d.pipe(lambda s, **kwargs: s.data + kwargs["z"] + kwargs["q"], **kwargs)
    assert isinstance(res, BCDict)
    assert res == {"A": "s1ZQ1", "B": "s2ZQ2"}, res


def test_apply_simple(d):
    f = lambda s: s.data
    assert bcdict.apply(f, d) == {"A": "s1", "B": "s2"}


def test_apply_with_args(d):
    args = ["X", {"A": "A", "B": "B"}]
    f = lambda s, *args: s.data + "".join(args)
    res = bcdict.apply(f, d, *args)
    assert isinstance(res, BCDict)
    assert res == {"A": "s1XA", "B": "s2XB"}, res


def test_apply_with_kwargs(d):
    kwargs = {"z": "Z", "q": {"A": "Q1", "B": "Q2"}}
    f = lambda s, **kwargs: s.data + kwargs["z"] + kwargs["q"]
    res = bcdict.apply(f, d, **kwargs)
    assert isinstance(res, BCDict)
    assert res == {"A": "s1ZQ1", "B": "s2ZQ2"}, res


def test_bootstrap(num_dict):
    keys = ["A", "B"]
    f = lambda x, m: x * m
    res = bcdict.bootstrap(keys, f, num_dict, m=2)
    assert isinstance(res, BCDict)
    assert res == {"A": 8, "B": 10}


def test_bootstrap_arg(num_dict):
    keys = ["A", "B"]
    f = lambda x, m: x * m
    res = bcdict.bootstrap_arg(keys, f, num_dict)
    assert isinstance(res, BCDict)
    assert res == {"A": "AAAA", "B": "BBBBB"}


def test_bootstrap_kwarg_with_broadcast(num_dict):
    keys = ["A", "B"]
    f = lambda n, s: str(n) + s
    res = bcdict.bootstrap_kwarg(keys, f, n=num_dict, argname="s")
    assert isinstance(res, BCDict)
    assert res == {"A": "4A", "B": "5B"}


def test_bootstrap_kwarg():
    keys = ["A", "B"]
    f = lambda n, s: str(n) + s
    res = bcdict.bootstrap_kwarg(keys, f, n=3, argname="s")
    assert isinstance(res, BCDict)
    assert res == {"A": "3A", "B": "3B"}


@pytest.mark.skipif(
    "pandas" not in sys.modules or "sklearn" not in sys.modules,
    reason="requires extra-tests dependencies",
)
def test_integration_test():
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
        ("d.eq(5)", {"A": False, "B": True}),
        ("d.ne(5)", {"A": True, "B": False}),
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


def test_to_list():
    d1 = BCDict({"a": 1, "b": 2})
    d2 = BCDict({"a": 3, "b": 4})

    res = to_list(d1, d2)
    assert res == {"a": [1, 3], "b": [2, 4]}


@pytest.mark.parametrize(
    "d1, d2",
    [
        (BCDict({"a": 1, "b": 2}), BCDict({"a": 3})),
        (BCDict({"a": 1}), BCDict({"a": 3, "b": 4})),
    ],
)
def test_to_list_raises_on_different_keys(d1, d2):
    with pytest.raises(ValueError):
        to_list(d1, d2)


def test_to_list_raises_on_empty_input():
    with pytest.raises(ValueError):
        to_list()


def test_bcdict_kwargs_unpacking(num_dict):
    def add(A, B):
        return A + B

    assert add(**num_dict) == 9


def test_bcdict_iteration(num_dict):
    keys = [k for k in num_dict]
    assert keys == ["A", "B"]


@pytest.mark.parametrize("type_", (tuple, list))
def test_bcdict_unpack(type_):
    d = BCDict(
        {
            "A": type_((1, 2)),
            "B": type_((3, 4)),
        }
    )
    x, y = d.unpack()
    assert x == {"A": 1, "B": 3}
    assert y == {"A": 2, "B": 4}


def test_bcdict_unpack_different_len_raises():
    """Test that tuple unpacking with tuples of different lengths raises."""
    d = BCDict(
        {
            "A": (1, 2),
            "B": (3, 4, 5),
        }
    )
    with pytest.raises(ValueError):
        d.unpack()


def test_version_is_semver_string():
    semver_pattern = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    version = bcdict.__version__
    print(f"Version is {version}")
    assert re.match(semver_pattern, version)
