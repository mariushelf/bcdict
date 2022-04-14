from .bcdict import BCDict, to_list

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version

__version__ = version("bcdict")
