[![Tests](https://github.com/mariushelf/bcdict/actions/workflows/cicd.yaml/badge.svg)](https://github.com/mariushelf/bcdict/actions/workflows/cicd.yaml)
[![codecov](https://codecov.io/gh/mariushelf/bcdict/branch/master/graph/badge.svg)](https://codecov.io/gh/mariushelf/bcdict)
[![PyPI version](https://badge.fury.io/py/bcdict.svg)](https://pypi.org/project/bcdict/)
[![Documentation Status](https://readthedocs.org/projects/bcdict/badge/?version=latest)](https://bcdict.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


# Broadcast Dictionary


Python dictionary with broadcast support.

Behaves like a regular dictionary.

Allows to apply operations to all its values at once.
Whithout loops, whithout dict comprehension.

## Installation

```bash
pip install bcdict
```

## Usage

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

## Next steps

See the [introduction notebook](docs/source/examples/introduction.ipynb) and other
[examples](docs/source/examples/examples.md).

Also check out the full documentation on
[bcdict.readthedocs.io](https://bcdict.readthedocs.io/en/latest/).


## Changelog

### v0.4.3
* fix: unpickling causes recursion error

### v0.4.2
* docs: improve the documenation

### v0.4.1
* fix: sphinxcontrib-mermaid gets installed as default dependency, should be dev dependency

### v0.4.0
* new functions `eq()` and `ne()` for equality/inequality with broadcast support

### v0.3.0
* new functions in `bcdict` package:
  * `apply()`
  * `broadcast()`
  * `broadcast_arg()`
  * `broadcast_kwarg()`
* docs: write some documentation and host it on [readthedocs](https://bcdict.readthedocs.io/en/latest/)

### v0.2.0
* remove `item()` function. Use `.a[]` instead.

### v0.1.0
* initial release


Original repository: [https://github.com/mariushelf/bcdict](https://github.com/mariushelf/bcdict)

Author: Marius Helf
([helfsmarius@gmail.com](mailto:helfsmarius@gmail.com))
