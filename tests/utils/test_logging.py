import pytest

from smoe.utils.logging import get_logger


def err_func():
    return 1 / 0


def test_log():
    logger = get_logger("test")  # noqa: F841


def test_err_func():
    with pytest.raises(ZeroDivisionError):
        res = err_func()  # noqa: F841
