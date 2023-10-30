import random
import string


def get_random_string(length: int = 8) -> str:
    """Generate a unique random string.

    Args:
        length (int, optional): Length of the random string. Defaults to 16.

    Returns:
        str: A unique random string.
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))
