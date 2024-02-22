import random
import re
import string
from argparse import ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def string2list(string, sep=","):
    if isinstance(string, list) or string is None:
        return string
    else:
        split_string = string.split(sep)
        return [int(num) for num in split_string]


def extract_numbers(string):
    """Extract numbers (int, float) from a given string."""
    pattern = r"[-+]?\d*\.\d+|\d+"
    matches = re.findall(pattern, string)
    numbers = [float(match) if "." in match else int(match) for match in matches]
    return numbers


def calculate_non_ascii_ratio(string):
    """Calculate the non-ASCII ratio of a given string."""
    if len(string) == 0:
        non_ascii_ratio = 0.0
    else:
        non_ascii_count = sum(1 for char in string if ord(char) >= 128)
        non_ascii_ratio = non_ascii_count / len(string)
    return non_ascii_ratio


def remove_non_ascii_code(string):
    """Use a regular expression to remove all non-ASCII characters"""
    string = re.sub(r"[^\x00-\x7F]+", "", string)
    return string


def replace_non_ascii_code(string):
    """
    Replace common non-ASCII characters with their ASCII counterparts in the given string.

    :param string: Input string with non-ASCII characters.
    :return: String with non-ASCII characters replaced.
    """
    string = re.sub(r"“|”", '"', string)
    string = re.sub(r"‘|’", "'", string)
    string = re.sub(r"—|–", "-", string)
    string = re.sub(r"…", "...", string)

    return string


def get_random_string(length: int = 8) -> str:
    """Generate a unique random string.

    Args:
        length (int, optional): Length of the random string. Defaults to 16.

    Returns:
        str: A unique random string.
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))
