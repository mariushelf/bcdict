![Tests](https://github.com/mariushelf/bcdict/actions/workflows/tests.yml/badge.svg?branch=master)

# Broadcast Dictionary


Dictionary with broadcast support.

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
`BCDict`, use the attribute accessor and item() function explicitly:

```python
>>> d.a.upper()
{'a': 'HELLO', 'b': 'WORLD!'}
```

Slicing with conflicting keys:
```python
>>> n = BCDict({1:"hello", 2: "world"})
>>> n[1]
'hello'
>>> n.item(1)
{1: 'e', 2: 'o'}
```



Original repository: [https://github.com/mariushelf/bcdict](https://github.com/mariushelf/bcdict)

Author: Marius Helf 
  ([helfsmarius@gmail.com](mailto:helfsmarius@gmail.com))


# License

MIT -- see [LICENSE](LICENSE)

