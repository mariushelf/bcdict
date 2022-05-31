from __future__ import annotations

import operator
from collections.abc import Collection, Sequence
from typing import Any, Callable, Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def to_list(*args: dict) -> dict:
    """Convert a list of dicts to a dict of lists."""
    if len(args) == 0:
        raise ValueError("Input can't be empty.")
    keys = [set(d.keys()) for d in args]
    ref = keys[0]
    for test in keys[1:]:
        if ref != test:
            raise ValueError("All input dictionaries must have the same keys.")

    res = {key: list(d[key] for d in args) for key in keys[0]}
    return res


def apply(f: Callable, *args: Any, **kwargs: Any) -> BCDict[K, Any]:
    """Apply callable on each element of some dicts.

    The first argument that is a BCDict serves as reference. `f` is called for each
    of its elements.

    args and kwargs are passed to `f` and broadcast if applicable.

    If there is no BCDict in the args or kwargs, a ValueError is raised.

    Parameters
    ----------
    f : Callable
        function or callable that is called
    args : BCDict | Any
        positional arguments passed to f
    kwargs : BCDict | Any
        keyword arguments passed to f

    Returns
    -------
    return_value : BCDict
        a broadcast dictionary with the same keys as the first BCDict in the arguments.
        Its values are the return values from the respective call to `f`.

    Examples
    --------
    >>> d = BCDict({"A": 2, "B": 3})
    >>> factor = BCDict({"A": 4, "B": 5})
    >>> f = lambda x1, x2, x3: x1 * x2 + x3
    >>> apply(f, d, factor, 1)
    BCDict({'A': 9, 'B': 16})

    # 2 * 4 + 1 = 9
    # 3 * 5 + 1 = 16
    """
    ref = None
    for arg in args:
        if isinstance(arg, BCDict):
            ref = arg
            break
    if ref is None:
        for kwarg in kwargs.values():
            if isinstance(kwarg, BCDict):
                ref = kwarg
                break
    if ref is None:
        raise ValueError("No BCDict in arguments")
    keys = ref
    return _broadcast_call(keys, f, *args, **kwargs)


def bootstrap(keys: list[str], f: Callable, *args, **kwargs):
    """Call f for for every key.

    args and kwargs are passed to `f` and broadcast if applicable.

    The result is a BCDcict with an entry for each element of keys and the respective
    return value of `f` as values.

    The keys are not passed to `f`, but only used as dictionary keys.
    """
    return _broadcast_call(keys, f, *args, **kwargs)


def bootstrap_arg(keys: list[str], f: Callable, *args, **kwargs):
    """Same as `bootstrap()`, but pass key as positional argument.

    When calling `f` for a key, the key is passed as the first positional argument."""
    return _broadcast_call(keys, f, *args, **kwargs, __key_as_arg=True)


def bootstrap_kwarg(keys: list[str], f: Callable, *args, argname: str, **kwargs):
    """Same as `bootstrap()`, but pass key as keyword argument.

    When calling `f` for a key, the key is passed as argument with name `argname`."""
    return _broadcast_call(keys, f, *args, **kwargs, __key_as_arg=argname)


def _broadcast_call(
    keys, f: Callable, *args, __key_as_arg: bool | str = False, **kwargs
):
    result: BCDict = BCDict()
    for key in keys:
        pipeargs, pipekwargs = BCDict._broadcast_args(key, keys, *args, **kwargs)
        if __key_as_arg:
            if __key_as_arg is True:
                pipeargs.insert(0, key)
            else:
                pipekwargs[__key_as_arg] = key
        result[key] = f(*pipeargs, **pipekwargs)
    return result


class BCDict(dict, Generic[K, V]):
    """Dictionary with broadcast support.

    Allows to apply functions to all its elements, or
    retrieve attributes of all its elements.

    Parameters
    ----------
    ipython_safe : bool, optional
        whether to use some black magic to prevent lengthy formatter errors in jupyter
        notebook or lab. Can be problematic if you want to access attributes in the
        dictionary's values that start with "_ipython_" or "_repr_". Then you need
        to use the `.a` accessor or disdable `ipython_safe`. Default: True
    *args
        forwarded to `dict`
    **kwargs
        forwarded to `dict`

    Examples
    --------

    >>> d = BCDict({"a": "hello", "b": "world!"})
    >>> d
    BCDict({'a': 'hello', 'b': 'world!'})


    Regular element access:

    >>> d['a']
    'hello'

    Regular element assignments

    >>> d['a'] = "Hello"
    >>> d
    BCDict({'a': 'Hello', 'b': 'world!'})

    Calling functions:

    >>> d.upper()
    BCDict({'a': 'HELLO', 'b': 'WORLD!'})

    Slicing:

    >>> d[1:3]
    BCDict({'a': 'el', 'b': 'or'})

    Applying functions:

    >>> d.pipe(len)
    BCDict({'a': 5, 'b': 6})

    When there is a conflict between an attribute in the values and an attribute in
    `BCDict`, use the attribute accessor explicitly:

    >>> d.a.upper()
    BCDict({'a': 'HELLO', 'b': 'WORLD!'})

    Slicing with conflicting keys:

    >>> n = BCDict({1:"hello", 2: "world"})
    >>> n[1]
    'hello'
    >>> n.a[1]
    BCDict({1: 'e', 2: 'o'})
    """

    class DictAccessor:
        """Internal helper class.

        This is what BCDict.a returns.
        """

        def __init__(self, data: dict[K, V]):
            self.__data: dict[K, V] = data  # this is the broadcast dict

        def __getattr__(self, item: str) -> BCDict[K, Any]:
            return BCDict({k: getattr(v, item) for k, v in self.__data.items()})

        def __getitem__(self, item: Any) -> BCDict[K, Any]:
            f = lambda d, item: d[item]  # noqa
            return apply(f, self.__data, item)

        def __setattr__(self, item: str, value: Any) -> None:
            if item.startswith("_DictAccessor__"):
                super().__setattr__(item, value)
            else:
                apply(setattr, self.__data, item, value)

        def __setitem__(self, item: str, value: Any) -> None:
            def f(d, item, value):
                d[item] = value

            apply(f, self.__data, item, value)

    def __init__(self, *args: Any, ipython_safe: bool = True, **kwargs: Any):
        self.__ipython_safe = ipython_safe
        super().__init__(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> BCDict[K, Any]:
        """Call each element of the dictionary with args and kwargs.

        args and kwargs are broadcasted if applicable.
        """
        result: BCDict[K, Any] = BCDict()
        for k, v in self.items():
            pipeargs, pipekwargs = self._broadcast_args(k, self.keys(), *args, **kwargs)
            result[k] = v(*pipeargs, **pipekwargs)
        return result

    @property
    def broadcast(self) -> BCDict.DictAccessor:
        """Attribute access. Use this to get an attribute of each value in the
        dictionary which has the same name as an attribute of the `BCDict` class."""
        return self.a

    @property
    def a(self) -> BCDict.DictAccessor:
        """Shorthand version of `broadcast` property."""
        return self.DictAccessor(self)

    def __getitem__(self, item: Any) -> V | BCDict[K, Any]:
        """Slice function.

        When `item` is a key of the BCDict, return the respective value.

        Else, if `item` is a dictionary with the same keys as this BCDict, then
        slice each value of this BCDict with the corresponding element of `item`, and
        return the result as a new BCDict.

        Else, slice each value in the dictionary with `item` and return a new dict.

        To slice each value with an item that is also in this dictionary, use
        the `item()` function instead.
        """
        try:
            return super().__getitem__(item)
        except (KeyError, TypeError):
            if isinstance(item, dict) and set(item.keys()) == set(self.keys()):
                # broadcast slice
                return BCDict({k: v[item[k]] for k, v in self.items()})
            return self.a[item]

    def __getattr__(self, item: str) -> Any:
        if (
            item.startswith("_ipython_") or item.startswith(("_repr_"))
        ) and self.__ipython_safe:
            # prevent FormatterWarning in ipython notebooks
            raise AttributeError()
        return getattr(self.a, item)

    def __setattr__(self, key, value) -> None:
        if key in dir(self) or (key.startswith("_BCDict__")):
            # if key is in the BCDict class, overwrite it in this class
            super().__setattr__(key, value)
        else:
            setattr(self.a, key, value)
            # apply(setattr, self, key, value)

    def pipe(self, f: Callable, *args: Any, **kwargs: Any) -> BCDict[K, Any]:
        """Apply callable on each element of the dict.

        args and kwargs are passed to `f` and broadcasted if applicable.
        """
        return apply(f, self, *args, **kwargs)

    def __add__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.add)

    def __mul__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.mul)

    def __matmul__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.matmul)

    def __sub__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.sub)

    def __mod__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.mod)

    def __truediv__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.truediv)

    def __floordiv__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.floordiv)

    def __pow__(self, power: dict | Any) -> BCDict:
        return self.__generic_operator(power, operator.pow)

    def __lshift__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.lshift)

    def __rshift__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.rshift)

    def __lt__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.lt)

    def __le__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.le)

    def __gt__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.gt)

    def __ge__(self, other: dict | Any) -> BCDict:
        return self.__generic_operator(other, operator.ge)

    # __eq__, __ne__ not supported because they are defined in the `dict` class
    # and we don't want to override them. Using `eq()` and `ne()` instead.

    def eq(self, other: dict | Any) -> BCDict:
        """Element-wise equality with broadcast support."""
        return self.__generic_operator(other, operator.eq)

    def ne(self, other: dict | Any) -> BCDict:
        """Element-wise inequality with broadcast support."""
        return self.__generic_operator(other, operator.ne)

    def __generic_operator(
        self, other: Any | dict, f: Callable[[Any, Any], Any]
    ) -> BCDict:
        if isinstance(other, dict) and set(self.keys()) == set(other.keys()):
            return BCDict({k: f(v, other[k]) for k, v in self.items()})
        else:
            return BCDict({k: f(v, other) for k, v in self.items()})

    @staticmethod
    def _broadcast_args(
        key: str, keys: Collection[str], *args: Any | dict, **kwargs: Any | dict
    ) -> tuple[list, dict]:
        """Get broadcasted args and kwargs for applying a function to the element
        `key`.

        Parameters
        ----------
        key : str
            current key
        keys : list[str]
            all keys in the dictionary
        args : Any, dict[str, Any]
            positional arguments
        kwargs : Any, dict[str, Any]
            keyword arguments

        Returns
        -------
        pipeargs :
            applicable, correctly broadcasted args
        pipekwargs :
            applicable, correctly broadcasted kwargs
        """
        keys = set(keys)
        broadcast_args = set()
        broadcast_kwargs = set()
        for ix, v in enumerate(args):
            if isinstance(v, dict) and set(v.keys()) == keys:
                broadcast_args.add(ix)
        for k, v in kwargs.items():
            if isinstance(v, dict) and set(v.keys()) == keys:
                broadcast_kwargs.add(k)
        pipeargs = [
            arg[key] if ix in broadcast_args else arg for ix, arg in enumerate(args)
        ]
        pipekwargs = {
            kw_name: kw_val[key] if kw_name in broadcast_kwargs else kw_val
            for kw_name, kw_val in kwargs.items()
        }
        return pipeargs, pipekwargs

    def unpack(self):
        """Convert BCDict of tuples into tuple of ``BCDict``."""

        # check that all values are tuples
        if not all(isinstance(v, Sequence) for v in self.values()):
            raise ValueError("all values must be sequences")

        # check that all values have the same length
        lengths = set(map(len, self.values()))
        if not len(lengths) == 1:
            raise ValueError(
                f"all values must be sequences of the same length, but lengths are {lengths}"
            )
        tuple_length = lengths.pop()
        result = tuple([BCDict() for _ in range(tuple_length)])
        for i in range(tuple_length):
            for k, v in self.items():
                result[i][k] = v[i]

        return result

    def __repr__(self):
        return f"BCDict({super().__repr__()})"
