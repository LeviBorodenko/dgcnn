"""
Some helpful utility functions.
"""
__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


def is_positive_integer(obj, name: str) -> None:
    """Checks if obj is a positive integer and
    raises appropriate errors if it isn't.

    Arguments:
        obj: object to be checked.
        name (str): name of object for error messages.
    """

    if type(obj) != int:
        raise ValueError(f"{name} should be an integer.")
    elif obj <= 0:
        raise ValueError(f"{name} should be an positive integer.")
