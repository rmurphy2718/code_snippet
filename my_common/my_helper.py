import torch
import numpy as np
import scipy.sparse as sps
from pathlib import Path
from collections import OrderedDict


def p_freq_table(arr):
    """
    Return a frequency table for printing, i.e. count how many unique
    Regardless of torch or numpy
    """
    if isinstance(arr, np.ndarray):
        return np.unique(arr, return_counts=True)
    elif torch.is_tensor(arr):
        out = torch.unique(arr, return_counts=True)
        return tuple(tt.cpu() for tt in out)
    else:
        raise ValueError("Unhandled array type in frequency table")


def enum_sprint(en):
    """ Return a string representing enum members for printing """
    return list(en.__members__.keys())


def concat_list(lst, ax):
    """
    Take a list of torch tensors or numpy arrays,
    concatenate them along given axis
    """
    if torch.is_tensor(lst[0]):
        return torch.cat(lst, dim=ax)
    elif isinstance(lst[0], np.ndarray):
        return np.concatenate(lst, axis=ax)
    else:
        raise ValueError("Illegal array-like in lst")


def change_extension(fp, new_extension):
    assert isinstance(fp, str) and isinstance(new_extension, str)
    if new_extension[0] != '.':
        new_extension = "." + new_extension
    p = Path(fp)
    return str(p.with_suffix(new_extension))


def is_coo(mat):
    return isinstance(mat, sps.coo.coo_matrix)


def is_ndarray(mat):
    return isinstance(mat, np.ndarray) and not isinstance(mat, np.matrix)


def is_tensor(mat):
    return isinstance(mat, torch.Tensor)


def is_positive_int(x):
    return isinstance(x, int) and x > 0


def diags_all_one(mat):
    """
    Check if the diagonal of mat are all ones.
    :param mat: ndarray, spmatrix or tensor
    """
    if is_ndarray(mat) or sps.isspmatrix(mat):
        diag = mat.diagonal()
        return np.array_equal(diag, np.ones_like(diag))
    elif is_tensor(mat):
        diag = mat.diagonal()
        return torch.equal(diag, torch.ones_like(diag))
    else:
        raise ValueError("mat is neither of (ndarray, spmatrix, torch.tensor).")


def is_2d_symmetric(mat):
    """ Test if array is 2d and symmetric """
    if is_ndarray(mat):
        if mat.ndim != 2:
            return False
        if mat.shape[0] != mat.shape[1]:
            return False
        else:
            return np.allclose(mat, mat.T)
    else:
        raise NotImplementedError("is_symmetric only implemented for numpy right now")


def igraph_to_numpy(graph):
    """ Get symmetric numpy array from an igraph graph  """
    adj = np.array(graph.get_adjacency(), dtype=np.float)
    return adj


def string_to_number(string):
    """ Cast as numeric type if possible"""
    # integer check
    if string.isnumeric():
        return int(string)
    # Float check
    try:
        ret = float(string)
        return ret
    except ValueError:
        return string


def args_list_to_dict(args):
    """
    Convert list of strings [key1, val1, key2, val2] to dictionary
    """
    if args is None:
        out_dict = {}
    else:
        n = len(args)
        out_dict = dict()
        assert n % 2 == 0
        for ii in range(0, n, 2):
            k = args[ii]
            v = args[ii + 1]
            if isinstance(v, str):
                v = string_to_number(v)
                if v == "True":
                    v = True
                elif v == "False":
                    v = False

            out_dict[k] = v

    # sort by key
    sorted_out_dict = OrderedDict(sorted(out_dict.items()))

    return sorted_out_dict


if __name__ == "__main__":
    A = torch.randint(-5, 9, (2, 3))
    B = torch.randint(-5, 9, (5, 3))
    r1 = concat_list([A, B], 0)
    print(r1.shape, type(r1))

    a = np.random.randint(-5, 9, (2, 5))
    b = np.random.randint(-5, 9, (20, 5))
    r2 = concat_list([a, b], 0)
    print(r2.shape, type(r2))


    v1 = torch.arange(5).reshape(5, 1)
    v2 = torch.arange(8, 5, -1).reshape(3, 1)
    concat_list([v1, v2], 0)

    v3 = np.arange(5).reshape(5, 1)
    v4 = np.arange(8, 5, -1).reshape(3, 1)
    concat_list([v3, v4], 0)
