from enum import Enum
from functools import partial

import numpy as np

from my_common.my_helper import is_ndarray, is_2d_symmetric


class Canonical_Order(Enum):
    # for adjacency matrices
    id = 1
    identity = 1

    rev = 2
    reverse = 2

    degA = 3
    degreeAscending = 3

    degD = 4
    degreeDescending = 4


class Canonical_Orderings:

    def __init__(self, canon_args=None):
        """
        Sets
         1. self.do_ordering, a boolean
         2. self.canon dictionary: the ordering to apply to training & test data

        :param canon_args: either list of strings corresponding to Canonical_Order Enum, or None
        """

        assert isinstance(canon_args, list) or canon_args is None

        if canon_args is None:
            self.do_ordering = False
            self.canon = {'train': Canonical_Order.identity,
                          'val': Canonical_Order.identity}  # for the sake of logging
        else:  # already a list
            assert 1 <= len(canon_args) <= 2
            assert all(arg in Canonical_Order.__members__ for arg in canon_args)
            self.canon = {'train': Canonical_Order[canon_args[0]]}
            # If only one argument is provided, order training and validation graphs by the same canon
            if len(canon_args) == 1:
                self.canon['val'] = Canonical_Order[canon_args[0]]
            else:
                self.canon['val'] = Canonical_Order[canon_args[1]]

            if self.canon['train'] == Canonical_Order.identity and self.canon['val'] == Canonical_Order.identity:
                self.do_ordering = False
            else:
                self.do_ordering = True

    def __str__(self):
        _str = "Canonical Orderings\n"
        _str += f"do_ordering = {self.do_ordering}\n"
        _str += f"canon = {self.canon}\n"

        return _str

    def get_canon_function(self, cv_split):
        assert isinstance(cv_split, str), "cv_split must be a string"
        assert cv_split in ['train', 'val'], f"Unrecognized data split: '{cv_split}'.  Should be 'train' or 'val'"

        if self.canon[cv_split] == Canonical_Order.reverse:
            return np.flip  # it will flip for all axes by default
        elif self.canon[cv_split] == Canonical_Order.degreeAscending:
            return self.sort_by_degree
        elif self.canon[cv_split] == Canonical_Order.degreeDescending:
            return partial(self.sort_by_degree, ascend=False)
        elif self.canon[cv_split] == Canonical_Order.identity:
            # we should never fall in this case; identity ordering is skipped when creating graphs
            return lambda x: x
        else:
            raise NotImplementedError(f"Haven't implemented cannon function {self.canon[cv_split].name}.")

    @staticmethod
    def sort_by_degree(mat, ascend=True):
        assert is_ndarray(mat)
        assert is_2d_symmetric(mat)

        degrees = np.sum(mat, axis=1)  # not the exact degrees due to the identity matrix added for GIN
        if ascend:
            argsort = np.argsort(degrees)
        else:
            argsort = np.argsort(-degrees)

        # Order the vertices by their degrees
        mat_ordered = mat[argsort, :]
        mat_ordered = mat_ordered[:, argsort]

        return mat_ordered  # return the ordered matrix (a new copy)


if __name__ == "__main__":
    # Testing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default=None, nargs="+", help="Order adjacency matrices by the specified \
                        canonical order. One argument for all data and two arguments for train and valid respectively.")
    args = parser.parse_args()

    co = Canonical_Orderings(args.canonical)
    print(co)

