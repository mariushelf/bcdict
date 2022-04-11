[![Tests](https://github.com/mariushelf/bcdict/actions/workflows/tests.yml/badge.svg)](https://github.com/mariushelf/bcdict/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mariushelf/bcdict/branch/master/graph/badge.svg)](https://codecov.io/gh/mariushelf/bcdict)
[![PyPI version](https://badge.fury.io/py/bcdict.svg)](https://pypi.org/project/bcdict/)

# Broadcast Dictionary


Python dictionary with broadcast support.

# Usage

```python
from bcdict import BCDict
>>> d = BCDict({"a": "hello", "b": "world!"})
>>> d
{'a': 'hello', 'b': 'world!'}
```


Regular element access:
```python
>>> d['a']
'hello'
```


Regular element assignments
```python
>>> d['a'] = "Hello"
>>> d
{'a': 'Hello', 'b': 'world!'}
```

Calling functions:
```python
>>> d.upper()
{'a': 'HELLO', 'b': 'WORLD!'}
```

Slicing:
```python
>>> d[1:3]
{'a': 'el', 'b': 'or'}
```

Applying functions:
```python
>>> d.pipe(len)
{'a': 5, 'b': 6}
```

When there is a conflict between an attribute in the values and an attribute in
`BCDict`, use the attribute accessor explicitly:

```python
>>> d.a.upper()
{'a': 'HELLO', 'b': 'WORLD!'}
```

Slicing with conflicting keys:
```python
>>> n = BCDict({1:"hello", 2: "world"})
>>> n[1]
'hello'
>>> # Using the attribute accessor:
>>> n.a[1]
{1: 'e', 2: 'o'}
```

# Full example

Here we create a dictionary with 3 datasets and then train, apply and validate
a linear regression on all 3 datasets without a single for loop or dictionary
comprehension.

```python
from collections.abc import Collection
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

# Conclusion: not a single for loop or dict comprehension used to train 3 models
# predict and evaluate 3 data sets :)

```



Original repository: [https://github.com/mariushelf/bcdict](https://github.com/mariushelf/bcdict)

Author: Marius Helf 
  ([helfsmarius@gmail.com](mailto:helfsmarius@gmail.com))


# License

MIT -- see [LICENSE](LICENSE)

