import re
import types
from argparse import ArgumentTypeError


def str2none(v, extended=True):
    if isinstance(v, types.NoneType):
        return v
    if v.lower() in ("none",) + (("null",) if extended else ()):
        return None
    else:
        raise ArgumentTypeError("NoneType value expected.")


def str2bool(v, extended=True):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true",) + (("yes", "t", "y", "1") if extended else ()):
        return True
    elif v.lower() in ("false",) + (("no", "f", "n", "0") if extended else ()):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def string2number_list(string, sep=","):
    if isinstance(string, list) or string is None:
        return string
    else:
        split_string = string.split(sep)
        return [float(num) if "." in num else int(num) for num in split_string]


def extract_numbers(string, match_float=True, match_sign=True):
    """Extract numbers from a given string."""
    pattern = r"\d+"
    if match_float:
        pattern = r"\d*\.\d+|" + pattern
    if match_sign:
        pattern = r"[-+]?" + pattern
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
