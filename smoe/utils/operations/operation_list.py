from typing import List, Union

import numpy as np


def chunk_list(input_list, num_chunks):
    """
    Split the input list into a specified number of chunks.

    Args:
    - input_list (list): The list to be evenly divided.
    - num_chunks (int): The desired number of chunks.

    Returns:
    - list of lists: List of chunks after even division.

    Example:
    >>> input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> num_chunks = 5
    >>> result = chunk_list(input_list, num_chunks)
    >>> print(result)  # Output: [[1, 2], [3, 4], [5, 6], [7, 8], [9]]
    """
    avg_chunk_size = len(input_list) // num_chunks
    remainder = len(input_list) % num_chunks

    chunks = []
    start = 0
    for _ in range(num_chunks):
        chunk_size = avg_chunk_size + 1 if remainder > 0 else avg_chunk_size
        chunks.append(input_list[start : start + chunk_size])
        start += chunk_size
        remainder -= 1

    return chunks


def chunk_list_with_yield(input_list, num_chunks):
    """
    Split the input list into a specified number of chunks using a generator.

    Args:
    - input_list (list): The list to be evenly divided.
    - num_chunks (int): The desired number of chunks.

    Yields:
    - list of lists: Chunks yielded one at a time.

    Example:
    >>> input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> num_chunks = 5
    >>> for chunk in chunk_list_with_yield(input_list, num_chunks):
    >>>     print(chunk)  # Output: [1, 2]  [3, 4]  [5, 6]  [7, 8]  [9]
    """
    avg_chunk_size = len(input_list) // num_chunks
    remainder = len(input_list) % num_chunks

    start = 0
    for _ in range(num_chunks):
        chunk_size = avg_chunk_size + 1 if remainder > 0 else avg_chunk_size
        yield input_list[start : start + chunk_size]
        start += chunk_size
        remainder -= 1


def split_list(input_list, split_length, drop_last=False):
    """
    Split a list into sublists, each with a length of split_length.

    Args:
    - input_list (list): The list to be split.
    - split_length (int): Length of each sublist.
    - drop_last (bool): Whether to drop the last sublist if its length is insufficient. Default is False.

    Returns:
    - list of lists: List of split sublists.

    Example:
    >>> input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> split_length = 5
    >>> result = split_list(input_list, split_length, drop_last=False)
    >>> print(result)  # Output: [[1, 2, 3, 4, 5], [6, 7, 8, 9]]
    """
    if split_length <= 0:
        raise ValueError("split_length must be a positive integer!")

    num_elements = len(input_list)
    num_splits = num_elements // split_length

    sublists = [
        input_list[i * split_length : (i + 1) * split_length] for i in range(num_splits)
    ]

    if not drop_last and num_splits * split_length < num_elements:
        sublists.append(input_list[num_splits * split_length :])

    return sublists


def split_list_with_yield(input_list, split_length, drop_last=False):
    """
    Split a list into sublists using a generator, each with a length of split_length.

    Args:
    - input_list (list): The list to be split.
    - split_length (int): Length of each sublist.
    - drop_last (bool): Whether to drop the last sublist if its length is insufficient. Default is False.

    Yields:
    - list of lists: Sublists yielded one at a time.

    Example:
    >>> input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> split_length = 5
    >>> result = split_list_with_yield(input_list, split_length, drop_last=False)
    >>> for sublist in result:
    >>>     print(sublist)  # Output: [1, 2, 3, 4, 5]  [6, 7, 8, 9]
    """
    if split_length <= 0:
        raise ValueError("split_length must be a positive integer!")

    num_elements = len(input_list)
    num_splits = num_elements // split_length

    start = 0
    for _ in range(num_splits):
        sublist = input_list[start : start + split_length]
        yield sublist
        start += split_length

    if not drop_last and start < num_elements:
        sublist = input_list[start:]
        yield sublist


def replicate_elements(input_list, num_copies: Union[int, float, List[int]]):
    """
    Replicate each element in the original list a fixed or variable number of times,
    with support for decimals indicating a proportional number of additional elements.

    Args:
    - input_list (list): The original list of elements.
    - num_copies: The number of times each element should be replicated.
      It can take multiple forms:
        - An integer: Each element in the list is replicated this many times.
        - A float: The integer part dictates the fixed number of times each element is replicated.
          The fractional part is used to determine a proportional number of extra elements to be
          randomly chosen and replicated once. For example, if num_copies is 2.5 and the list
          has 4 elements, each element is replicated 2 times, and additionally, 2 (50% of 4)
          randomly chosen elements are replicated once more.
        - A list of integers: Each element in the input list is replicated according to the
          corresponding number of times specified in this list. This allows for variable replication
          per element. The length of this list must match the length of the input list.

    Returns:
    - list: The new list with replicated elements.
    """
    if isinstance(num_copies, (int, float)):
        # Fixed replication, integer part and proportional part handled
        int_part = int(num_copies)  # Integer part for the definite copies
        num_copies_list = [int_part] * len(input_list)

        if isinstance(num_copies, float):
            frac_part = (
                num_copies - int_part
            )  # Fractional part for the proportional extra copies
            extra_copies_count = round(frac_part * len(input_list))

            if extra_copies_count > 0:
                # Choose the items to be replicated
                extra_num_copies_array = np.concatenate(
                    (
                        np.ones(extra_copies_count, dtype=int),
                        np.zeros(len(input_list) - extra_copies_count, dtype=int),
                    ),
                    axis=0,
                )
                np.random.shuffle(extra_num_copies_array)

                # Add to the "num_copies_list"
                num_copies_list = (
                    np.array(num_copies_list) + extra_num_copies_array
                ).tolist()

    elif isinstance(num_copies, list):
        # Variable replication based on the list
        if len(input_list) != len(num_copies):
            raise ValueError(
                "Lengths of input_list and num_copies_list must be the same."
            )

        num_copies_list = num_copies

    else:
        raise ValueError(
            "Invalid type for num_copies. It should be an int, float, or a list."
        )

    new_list = []
    for item, num_copies_item in zip(input_list, num_copies_list):
        new_list.extend([item] * num_copies_item)

    return new_list


def all_elements_equal(input_list):
    """Check if all elements in the list are equal."""
    if not input_list:
        return True
    return len(set(input_list)) == 1


def mean_value_of_elements(input_list):
    """Return the mean value of elements in the list."""
    total_value = 0
    total_cnt = 0
    for element in input_list:
        if element is not None:
            total_value += element
            total_cnt += 1
    if total_cnt > 0:
        return total_value / total_cnt
    else:
        return 0
