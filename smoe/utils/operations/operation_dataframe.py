import pandas as pd


def chunk_dataframe(df, num_chunks):
    """
    Split the input DataFrame into a specified number of chunks.

    Args:
    - df (pd.DataFrame): The DataFrame to be evenly divided.
    - num_chunks (int): The desired number of chunks.

    Returns:
    - list of pd.DataFrame: List of chunks after even division.

    Example:
    >>> input_df = pd.DataFrame({'A': range(1, 10)})
    >>> num_chunks = 5
    >>> result = chunk_dataframe(input_df, num_chunks)
    >>> for chunk in result:
    >>>     print(chunk)  # Output: DataFrame chunks printed one by one
    """
    avg_chunk_size = len(df) // num_chunks
    remainder = len(df) % num_chunks

    chunks = []
    start = 0
    for _ in range(num_chunks):
        chunk_size = avg_chunk_size + 1 if remainder > 0 else avg_chunk_size
        chunks.append(df.iloc[start : start + chunk_size])
        start += chunk_size
        remainder -= 1

    return chunks


def chunk_dataframe_with_yield(df, num_chunks):
    """
    Split the input DataFrame into a specified number of chunks using a generator.

    Args:
    - df (pd.DataFrame): The DataFrame to be evenly divided.
    - num_chunks (int): The desired number of chunks.

    Yields:
    - pd.DataFrame: DataFrame chunks yielded one at a time.

    Example:
    >>> input_df = pd.DataFrame({'A': range(1, 10)})
    >>> num_chunks = 5
    >>> for chunk in chunk_dataframe_with_yield(input_df, num_chunks):
    >>>     print(chunk)  # Output: DataFrame chunks printed one by one
    """
    avg_chunk_size = len(df) // num_chunks
    remainder = len(df) % num_chunks

    start = 0
    for _ in range(num_chunks):
        chunk_size = avg_chunk_size + 1 if remainder > 0 else avg_chunk_size
        yield df.iloc[start : start + chunk_size]
        start += chunk_size
        remainder -= 1
