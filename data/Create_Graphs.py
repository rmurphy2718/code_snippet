import os
DEBUG = int(os.environ.get("pydebug", 0)) > 0
if DEBUG:
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=27189, stdoutToServer=True, stderrToServer=True, suspend=False)

import numpy as np
import torch
from util.primes import Primes
from scipy.sparse import coo_matrix
from collections import OrderedDict
from itertools import permutations, islice, count
import networkx as nx
from math import gcd, sqrt
from common.helper import PickleUtil, print_stats
from util.constants import LType, DEFAULT_RANDOM_GENERATION_SEED

SYNTHETIC_DIR = "../Synthetic_Data/"
BASE_DIR = SYNTHETIC_DIR + "processed/"
SYNTHETIC_GRAPHS_PREFIX = "graphs_Kary_Deterministic_Graphs"
SYNTHETIC_Y_PREFIX = "y_Kary_Deterministic_Graphs"
L_TARGET_PREFIX = "y_l_map"


def are_coprime(x, y):
    """
    Are x and y coprime integers?
    """
    assert isinstance(x, int) and isinstance(y, int), "Arguments must be positive integers"
    assert x > 0 and y > 0, "Arguments must be positive integers"
    return gcd(x, y) == 1


class Synthetic_Graphs():

    def __init__(self, NN, num_y, num_perm, ltype=LType.brute, sparse=False,
                 sample_permutations=False, random_state=DEFAULT_RANDOM_GENERATION_SEED, force_N_prime=True):
        """
        :parameter sample_permutations: Return uniformly sampled graphs (adjacency matrices) from each isomorphic class
                                       In the RP paper, this value was set to False.
        :parameter random_state: Random state used for generating permutations when sample_permutations is True
        """

        assert isinstance(NN, int) and NN > 9, "NN should be an integer greater than or equal to 10"
        assert isinstance(num_y, int) and 1 < num_y < NN - 1,\
            "num_y (i.e., the number of distinct skip lengths) should be an integer between 2 and NN-1"
        assert isinstance(num_perm, int) and 1 < num_perm <= 10000, \
            "num_perm (i.e., number of permutations for the same isomorphism class) should be int between 2 and 10000"

        assert isinstance(ltype, LType)
        assert isinstance(sparse, bool)
        assert isinstance(sample_permutations, bool)
        if sample_permutations:
            assert isinstance(random_state, int) and random_state > 0
            print_stats("Using randomly sampled permutations to create graphs within class")
        else:
            print_stats("Using deterministic permutations to create graphs within class")
            random_state = None

        assert isinstance(force_N_prime, bool)
        if force_N_prime:
            self.enforce_primality(NN)
        else:
            print_stats("Allowing N to be nonprime")

        self.NN = NN
        self.num_y = num_y
        self.num_perm = num_perm
        self.base_dir = ""
        self.ltype = ltype
        self.sparse = sparse
        self.sample_permutations = sample_permutations
        self.random_state = random_state
        self.force_N_prime = force_N_prime

    @staticmethod
    def enforce_primality(NN):
        if not Primes.is_prime(NN):
            Primes.print_closest_primes(NN)
            raise ValueError("\n\n\nEntered graph size must be prime!\nSee options above.\nTo relax, set force_N_prime=False")

    @classmethod
    def coprime_skip_lengths(self, NN):
        assert NN > 9, "NN should be greater than or equal to 10"
        avail = list()
        for LL in range(2, NN - 1):
            if are_coprime(int(NN), LL):
                avail.append(LL)
                # if len(NL_dict[NN]) > 4:
                #     continue
        return avail

    @classmethod
    def run_generator(self, LL, NN):
        """
        Generate a sequence of numbers for which a 4-regular circle graph can be created
        :param LL: Skip connection length
        :param NN: Order of the graph
        :return: A generated sequence
        """
        x = [0]
        for ii in range(0, NN):
            xnew = (x[-1] + LL) % NN
            x.append(xnew)
        return x

    @classmethod
    def create_circle_graph(cls, LL, NN):
        """ Make an edge list for a CSL(NN, LL) graph. """
        assert are_coprime(LL, NN)
        edge_list = []
        # Add edges from generator and immediate nbors
        walk = cls.run_generator(LL, NN)
        for ii in range(0, len(walk) - 1):
            # From walk
            edge_list.append((walk[ii], walk[ii + 1]))
            # From nbors
            edge_list.append((ii, (ii + 1) % NN))
        #
        # Make a full blown graph object for the sake of isomorphism testing.
        G = nx.Graph()
        G.add_nodes_from(list(np.arange(0, NN)))
        G.add_edges_from(edge_list)
        return G

    @classmethod
    def legal_skip_lengths(cls, num_y, NN, ltype):
        """
        Return a list of graphs and skip length values s.t. all graphs returned are pairwise nonisomorphic.
        There are two notions of nonisomorphic: analytically and distinguishable by networkx's isomorphic function.
        The original Relational Pooling paper forced difference per nx.

        As noted in https://arxiv.org/pdf/1905.12560.pdf,
        "Two CSL graphs G(n,k) and G(n',k') are not isomorphic unless n = n' and k ≡ ±k' (mod n)"

        :parameter num_y: Number of graphs to return, each corresponding to a representative from the target class.
        :parameter NN: Size of graph
        :return g_list:  A list of `num_y` networkx graphs that are pairwise nonisomorphic
        :return good_L:  A list of `num_y` integers representing the skip lengths of the returned graphs
        """
        assert isinstance(ltype, LType)

        # Get available L values
        if ltype in (LType.brute, LType.analytic_int_l):  # Later, we can separate brute_int and brute_prime if desired
            avav_L_vals = cls.coprime_skip_lengths(NN)
        else:
            raise ValueError("Unrecognized L type")

        g_list = list()
        good_L = list()
        for LL in avav_L_vals:
            # Determine if LL is good, create a CSL(NN, LL) graph.
            if ltype == LType.analytic_int_l:
                # ------------------------------------------------------------------------------------------
                # Valid as long as LL <  (NN+1)/2
                # ------------------------------------------------------------------------------------------
                # CSL(N, L1) and CSL(N, L2) graphs are isomorphic when
                # L1 + L2 = N     NOTE: 1 < L_1 < L_2 < N   w.l.o.g
                # But the list avav_L_vals does not contain duplicates, so
                # L1 <= L2 - 1
                # So we must have
                # L2 + (L2 - 1) < N, or
                # L2 < (N+1)/2     <- and note L2 plays the role of LL in this loop.
                if LL < (NN+1)/2:
                    legal = True
                    g = cls.create_circle_graph(LL, NN)
                else:
                    legal = False
            elif ltype == LType.brute:  # We can add different types of brute later if we want.
                # ------------------------------------------------------------------------------------------
                # Use NetworkX's function to determine whether graphs are isomorphic
                # Note: networkx mail fail and think nonisomorphic CSL graphs are isomorphic
                # ------------------------------------------------------------------------------------------
                g = cls.create_circle_graph(LL, NN)
                legal = True
                for idx, h in enumerate(g_list):
                    if nx.is_isomorphic(g, h):
                        # print(f"L={LL} is isomorphic with L={good_L[idx]}!")
                        legal = False
                        break
            else:
                raise ValueError("Invalid L type")

            if legal:
                good_L.append(LL)
                g_list.append(g)
                if len(good_L) == num_y:
                    break

        if num_y > len(good_L):
            error_string = "Requested number of skip lengths exceeds the maximum number of non-isomorphic graphs! "
            error_string += f"Requested: {num_y}, Maximum:{len(good_L)}."
            raise ValueError(error_string)

        return good_L, g_list

    @classmethod
    def make_skip_graphs(cls, LL, NN, num_perms, g, dic, sample_permutations=False,
                         random_state=DEFAULT_RANDOM_GENERATION_SEED, force_N_prime=True):
        """Subroutine that appends or instantiates adjacency matrices to dic
           We make num_perms non-isomorphic graphs from the class G_{skip}(n, l)
           :param LL: Skip length
           :param NN: Number of vertices
           :param num_perms: Number of permutations (non-isomorphic) graphs to make
           :param g: Graph correspond to (NN,LL,nem_perm) with identity permutation
           :param dic: Dictionary with keys=(n,l,perm_num) and values=adjmat
        """
        assert LL < NN - 1, "LL must be less than NN-1"
        assert isinstance(force_N_prime, bool)
        if force_N_prime:
            cls.enforce_primality(NN)
        else:
            print_stats("Not enforcing graph size to be prime!")

        mats = [nx.to_numpy_matrix(g)]
        relabel_dic = dict().fromkeys(range(NN))

        if sample_permutations:
            # random state has only a localized effect
            # add LL to random state to avoid identical permutations for every isomorphic class
            rng = np.random.RandomState(seed=random_state + LL)
        else:
            # Note that even adjacency matrices are not permuted here, all baselines and proposed methods will still
            # randomly (uniformly) permute adjacency matrices in the first step of forward(), due to the use of pi-SGD
            perms = permutations(list(range(NN)))  # Iterator of all perms
            next(perms)  # Skip identity perm

        # generate graphs until we have enough
        while len(mats) < num_perms:
            if sample_permutations:
                # Uniformly sample graphs from the non-isomorphic class (with replacement)
                perm = rng.permutation(NN)
                perm = perm.tolist()
            else:
                perm = next(perms)

            # relabel: make old_label -> new_label mapping
            for kk in relabel_dic.keys():
                relabel_dic[kk] = perm[kk]

            # Perform relabeling
            h = nx.relabel_nodes(g, relabel_dic)

            # Convert to a matrix
            A = nx.to_numpy_matrix(h, nodelist=list(range(NN)))  # must specify nodelist as original order

            # Check to see if it's equal to any previous
            same = False
            for mat in mats:
                if np.array_equal(A, mat):
                    same = True
                    break

            # It's original, append
            if not same:
                mats.append(A)

        # Add to dictionary
        for ii in range(num_perms):
            dic[(NN, LL, ii)] = mats[ii]

    @staticmethod
    def add_eye(in_mats, to_sparse):
        """
        Add 1 to the diagonal of all the matrices in in_mats
        :parameter in_mats: List of adjacency matrices
        :parameter to_sparse: Boolean, should we convert matrices to sparse coo_matrix?
        """
        assert isinstance(in_mats, list)
        assert isinstance(to_sparse, bool)

        out_mats = list()
        for mat in in_mats:
            out_mat = mat + np.eye(mat.shape[0])
            if to_sparse:
                out_mat = coo_matrix(out_mat)
            out_mats.append(out_mat)
        return out_mats

    @classmethod
    def make_y_tensor(self, graph_key_list):
        """
        Make a response tensor containing 0, 1, 2, ...
        which are class targets corresponding to skip lenghts

        Careful!  In this output, "2" does not mean skip length of 2
        """
        # Find out how many targets (y, corresponding to a skip L) there are in this graph collection
        # Also: biggest number of permutations, which we need to create
        # the vertex features.
        #
        l_to_targets = dict()
        for key in graph_key_list:
            nn, ll, pp = key
            if ll not in l_to_targets:
                l_to_targets[ll] = len(l_to_targets)  # map l to an integer 0, 1, 2, etc

        y = torch.empty([len(graph_key_list)]).long()
        ii = 0
        for key in graph_key_list:
            _, ll, pp = key
            y[ii] = l_to_targets[ll]
            ii += 1

        return y, l_to_targets

    def create_graphs(self, base_dir):
        """
        Create and pickle.dump CSL graphs to .pkl.
        Generate and torch.save corresponding targets (not skip lengths!).
        """
        self.base_dir = base_dir

        # obtain the set of legal skip lengths such that all (NN,LL) are non-isomorphic
        L_vals, L_graphs = self.legal_skip_lengths(self.num_y, self.NN, self.ltype)

        # adj_mat_dic an orderedDict that will store {(# vertices, skip length, permutation id):dense ndarray graph}
        adj_mat_dic = OrderedDict()
        for LL, gg in zip(L_vals, L_graphs):
            self.make_skip_graphs(LL=LL, NN=self.NN, num_perms=self.num_perm, g=gg, dic=adj_mat_dic,
                                  sample_permutations=self.sample_permutations, random_state=self.random_state,
                                  force_N_prime=self.force_N_prime)

        # Add an identity matrix to each adjacency matrix in the graph list
        # Transforms graphs to sps coo sparse matrices if self.sparse is True
        adjlist = self.add_eye(list(adj_mat_dic.values()), self.sparse)
        # Dump the matrices
        PickleUtil.write(adjlist, os.path.join(base_dir, self._adj_filename()))
        # Generate the targets corresponding to different skip lengths
        y_tensor, l_target_dict = self.make_y_tensor(
            list(adj_mat_dic.keys())
        )
        # Save the targets
        torch.save(y_tensor, os.path.join(base_dir, self._y_filename()))
        PickleUtil.write(l_target_dict, os.path.join(base_dir, self._l_target_filename()))

    def load_graphs(self, base_dir=None):
        """
        Load Synthetic Graphs.
        If a base_dir is specified, look for graphs with given ID (based on NN, num_y, num_perm) in base_dir
        Otherwise, look in the self.base_dir that was specified when create_graphs was called.
        :parameter base_dir: String representing directory path.
        """
        search_dir = ""
        if base_dir is None:
            search_dir = self.base_dir  # File existence checked below
        elif isinstance(base_dir, str):
            search_dir = base_dir

        if not os.path.isdir(search_dir):
            raise FileNotFoundError(f"Directory {search_dir} does not exist")

        adjmats = PickleUtil.read(os.path.join(search_dir, self._adj_filename()))
        y = torch.load(os.path.join(search_dir, self._y_filename()))
        return adjmats, y

    @classmethod
    def _graph_id(self, NN, num_y, num_perm, ltype, sparse, sampled_perm, sample_random_state):
        """ Workhorse method to get the graph ID string with or without instantiating the object"""
        assert isinstance(sampled_perm, bool)

        if isinstance(ltype, str):
            ltype = LType[ltype]

        if ltype == LType.brute:
            L_str = "nx_different_L"
        elif ltype == LType.analytic_int_l:
            L_str = "analytic_int_L"
        else:
            raise ValueError(f"Unrecognized ltype {ltype}")

        if sampled_perm:
            perm_str = "perms_rand"
        else:
            perm_str = "perms_det"

        sps_str = "sparse" if sparse else "dense"
        id_params = [NN, num_y, num_perm, L_str, sps_str, perm_str]

        # Add seed value for sampling permutations within a class IF NOT DEFAULT
        # (If deterministic samples, sample_random_state is None and we still don't add it to the path)
        if sample_random_state is None or sample_random_state == DEFAULT_RANDOM_GENERATION_SEED:
            srs_string = ""
            print(f"sample_random_state left at DEFAULT {DEFAULT_RANDOM_GENERATION_SEED} or None, not added to filepath")
        else:
            srs_string = f"_permseed_{sample_random_state}"
            print(f"sample_random_state is non-default {sample_random_state}, added to filepath")

        # Finalize
        id_str = "_".join(map(str, id_params))
        id_str += srs_string  # Add like this so we don't get double underscore
        return id_str

    def get_graph_id(self):
        """  Return the ID string of a synthetic graph"""
        return self._graph_id(self.NN, self.num_y, self.num_perm, self.ltype, self.sparse,
                              self.sample_permutations, self.random_state)

    @staticmethod
    def get_synthetic_style_graph_id(NN, num_y, num_perm, ltype, sparse, sample_permutations, random_state=None):
        """  Return the ID string of a synthetic graph
        :parameter NN: Number of vertices
        :parameter num_y: Number of targets (i.e. skip lengths)
        :parameter num_perm: Number of instances (permutations) of each class
        """
        assert isinstance(sparse, bool)
        assert isinstance(sample_permutations, bool)

        return Synthetic_Graphs._graph_id(NN, num_y, num_perm, ltype, sparse, sample_permutations, random_state)

    def _adj_filename(self):
        return SYNTHETIC_GRAPHS_PREFIX + "_" + self.get_graph_id() + ".pkl"

    def _y_filename(self):
        return SYNTHETIC_Y_PREFIX + "_" + self.get_graph_id() + ".pt"

    def _l_target_filename(self):
        return L_TARGET_PREFIX + "_" + self.get_graph_id() + ".pkl"


if __name__ == '__main__':
    # -------------------------------------------------------------------------------
    # Test if the graphs generated by Synthetic_Graphs() are identical to the old ones
    # -------------------------------------------------------------------------------
    print("Using new code to create graphs in RP paper...")
    graphs = Synthetic_Graphs(41, 10, 15, ltype=LType.brute, sparse=True, sample_permutations=False)
    graphs.create_graphs(BASE_DIR)

    g_new, targets_new = graphs.load_graphs()

    print("Loading graphs from RP paper directly...")
    g_old = PickleUtil.read(os.path.join(BASE_DIR, "".join([SYNTHETIC_GRAPHS_PREFIX, ".pkl"])))
    targets_old = torch.load(os.path.join(BASE_DIR, "".join([SYNTHETIC_Y_PREFIX, ".pt"])))

    print(f"size of new targets: {targets_new.size()}")
    print(f"size of old targets: {targets_old.size()}")
    print(torch.all(torch.eq(targets_new, targets_old)))

    print(f"Type of first graph: {type(g_new[0])}")

    # use todense for comparison
    for g1, g2 in zip(g_new, g_old):
        if not np.array_equal(g1.todense(), g2.todense()):
            raise ValueError("There exist different graphs!")

    print("The graphs were identical")

