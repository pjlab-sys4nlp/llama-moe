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


def all_elements_equal(input_list):
    """Check if all elements in the list are equal."""
    if not input_list:
        return True
    return len(set(input_list)) == 1
