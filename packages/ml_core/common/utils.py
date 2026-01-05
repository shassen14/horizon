# packages/ml_core/common/utils.py

from typing import MutableMapping


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flattens a nested dictionary.
    Example: {'a': {'b': 1}} -> {'a.b': 1}
    """
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
