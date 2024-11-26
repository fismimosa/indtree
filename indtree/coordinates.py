import numpy as np


def integer_bfs_index(n):
    return np.arange(n).reshape(-1, 1)


def int_to_binary_array(number, length=None):
    binary_str = bin(number)[2:]
    if length:
        binary_str = binary_str.zfill(length)
    binary_list = [int(x) for x in binary_str]

    return np.array(binary_list)


def binary_bfs_index(n, offset=1, max_len=None):
    if max_len is None:
        max_len = n.bit_length()
    out = np.empty((n, max_len), dtype=int)
    for i in range(n):
        out[i] = int_to_binary_array(i + offset, max_len)
    return out
