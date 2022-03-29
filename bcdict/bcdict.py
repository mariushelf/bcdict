from __future__ import annotations

import operator
from collections.abc import Collection
from typing import Any, Callable, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BCDict(dict, Generic[K, V]):
    """Dictionary which allows to apply functions to all its elements, or
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
    {'a': 'hello', 'b': 'world!'}


    # regular element access:
    >>> d['a']
    'hello'

    # regular element assignments
    >>> d['a'] = "Hello"
    >>> d
    {'a': 'Hello', 'b': 'world!'}

    Calling functions:
    >>> d.upper()
    {'a': 'HELLO', 'b': 'WORLD!'}

    Slicing:
    >>> d[1:3]
    {'a': 'el', 'b': 'or'}

    Applying functions:
    >>> d.pipe(len)
    {'a': 5, 'b': 6}

    When there is a conflict between an attribute in the values and an attribute in
    `BCDict`, use the attribute accessor and item() function explicitly:

    >>> d.a.upper()
    {'a': 'HELLO', 'b': 'WORLD!'}

    Slicing with conflicting keys:
    >>> n = BCDict({1:"hello", 2: "world"})
    >>> n[1]
    'hello'
    >>> n.item(1)
    {1: 'e', 2: 'o'}
    """

    class _DictAccessor:
        """Internal helper class.

        This is what BCDict.a returns.
        """

        def __init__(self, data: dict[K, V]):
            self.__data: dict[K, V] = data  # this is the broadcast dict

        def __getattr__(self, item: str) -> BCDict[K, Any]:
            return BCDict({k: getattr(v, item) for k, v in self.__data.items()})

        def __getitem__(self, item: Any) -> BCDict[K, Any]:
            return BCDict({k: v[item] for k, v in self.__data.items()})  # type: ignore

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
    def broadcast(self) -> BCDict._DictAccessor:
        """Attribute access. Use this to get an attribute of each value in the
        dictionary which has the same name as an attribute of the `BCDict` class."""
        return self.a

    @property
    def a(self) -> BCDict._DictAccessor:
        """Shorthand version of `broadcast` property."""
        return self._DictAccessor(self)

    def item(self, *item: Any) -> BCDict[K, Any]:
        """Slice each value in the dictionary with `item` and return a new dict."""
        if len(item) != 1:
            return BCDict({k: v[slice(*item)] for k, v in self.items()})
        else:
            item = item[0]
            return BCDict({k: v[item] for k, v in self.items()})

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
        if self.__ipython_safe and (
            item.startswith("_ipython_") or item.startswith(("_repr_"))
        ):
            # prevent FormatterWarning in ipython notebooks
            raise AttributeError()
        return getattr(self.a, item)

    def pipe(self, f: Callable, *args: Any, **kwargs: Any) -> BCDict[K, Any]:
        """Apply callable on each element of the dict.

        args and kwargs are passed to `f` and broadcasted if applicable.
        """
        result: BCDict[K, Any] = BCDict()
        keys = self.keys()
        for k, v in self.items():
            pipeargs, pipekwargs = self._broadcast_args(k, keys, *args, **kwargs)
            result[k] = f(v, *pipeargs, **pipekwargs)
        return result

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

    # eq, ne not supported because they are defined in the `dict` class
    # and we don't want to override them

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
