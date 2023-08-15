def chunk_list(input_list, num_chunks):
    """
    将输入列表均分为 num_chunks 数量的块。

    参数：
    input_list (list): 要均分的输入列表。
    num_chunks (int): 希望均分的块的数量。

    返回：
    list of lists: 均分后的块列表。

    示例：
    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_chunks = 5
    result = chunk_list(input_list, num_chunks)
    print(result)  # 输出：[[1, 2], [3, 4], [5, 6], [7, 8], [9]]
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
    将输入列表均分为 num_chunks 数量的块。

    参数：
    input_list (list): 要均分的输入列表。
    num_chunks (int): 希望均分的块的数量。

    返回：
    list of lists: 均分后的块列表。

    示例：
    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_chunks = 5
    for chunk in chunk_list_with_yield(input_list, num_chunks):
        print(chunk)  # 输出：[[1, 2], [3, 4], [5, 6], [7, 8], [9]]
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
    分割列表为若干子列表，每个子列表长度为 split_length。

    参数：
    input_list (list): 要分割的输入列表。
    split_length (int): 子列表的长度。
    drop_last (bool): 如果为 True，在最后一个子列表长度不足时是否丢弃它。默认为 False。

    返回：
    list of lists: 分割后的子列表。

    示例：
    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    split_length = 5
    result = split_list(input_list, split_length, drop_last=False)
    print(result)  # 输出：[[1, 2, 3, 4, 5], [6, 7, 8, 9]]
    """
    if split_length <= 0:
        raise ValueError("split_length 必须为正整数")

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
    使用生成器将列表分割为若干子列表，每个子列表长度为 split_length。

    参数：
    input_list (list): 要分割的输入列表。
    split_length (int): 子列表的长度。
    drop_last (bool): 如果为 True，在最后一个子列表长度不足时是否丢弃它。默认为 False。

    生成器：
    生成分割后的子列表。

    示例：
    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    split_length = 5
    result = split_list_with_yield(input_list, split_length, drop_last=False)
    for sublist in result:
        print(sublist)  # 输出：[[1, 2, 3, 4, 5], [6, 7, 8, 9]]
    """
    if split_length <= 0:
        raise ValueError("split_length 必须为正整数")

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
