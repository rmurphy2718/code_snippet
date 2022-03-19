# Examples on how to run:
# python create_data.py --experiment_data rp_paper
# Absolute balanced:
# python create_data.py --experiment_data customized -N 71 -L 12 -np 30 --print_splits
# Closed to balanced (pls see comments in balanced_cv_split):
# python create_data.py --experiment_data customized -N 31 -L 7 -np 17 --print_splits
# python create_data.py -data ba --sample_random_state 42 -gf first_degree --swapping_scheme first_two --do_swapping --n_splits 2 -ba_n 10 -ba_N 10 --ba_m_type constant --ba_m_args const 2


import pickle
import os
import random

import networkx as nx
import torch
from collections import OrderedDict
import scipy.sparse as sps
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.sparse import coo_matrix

from data.BA_data import BaData
from data.er_data import ErData
from data.task_functions import GraphTasks
from data.canonical_orderings import Canonical_Order, Canonical_Orderings
from util.constants import ExperimentData, SMP_NUM_SAMPLES, \
    TaskType, SwappingScheme, GraphFunction
from common.helper import PickleUtil, Util, print_stats
from my_common.my_helper import p_freq_table, igraph_to_numpy, is_tensor
from configs.graph_config import GraphDataArgParser, SYNTHETIC_GRAPHS, SYNTHETIC_Y, SMP_DATA_DIR, SMP_SINGLE_GRAPH_SIZE

DEBUG = int(os.environ.get("pydebug", 0)) > 0
if DEBUG:
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=27189, stdoutToServer=True, stderrToServer=True, suspend=False)


def scale_targets(yy):
    assert is_tensor(yy)
    assert yy.ndim == 1
    ybar = torch.mean(yy)
    std = torch.std(yy)
    out = torch.div(yy - ybar, std)
    return out


def paper_cv_split(num_graphs, cv_fold, num_classes=10):
    """
    The cross-validation split for the Relational Pooling paper only.

    Return a tuple of the train and val indices, depending on the cv_fold.
    Assumes 5-fold cross validation.
    This method shuffles the index (with a seed)
    The shuffle is consistent across machines with python3"""
    #
    # Extract indices of train and val in terms of the shuffled list, 2
    # Balanced across test and train
    # Assumes 10-class
    #
    random.seed(1)
    num_per_class = int(num_graphs / num_classes)
    val_size = int(0.2 * num_per_class)  # int() always takes the floor
    idx_to_classes = {}
    val_idx = []
    train_idx = []
    for cc in range(num_classes):
        idx_to_classes[cc] = list(range(cc * num_per_class, (cc + 1) * num_per_class))
        random.shuffle(idx_to_classes[cc])
        # These indices correspond to the validation for this class.
        class_val_idx = slice(cv_fold * val_size, cv_fold * val_size + val_size, 1)
        # Extract validation.
        vals = idx_to_classes[cc][class_val_idx]
        val_idx.extend(vals)
        train_idx.extend(list(set(idx_to_classes[cc]) - set(vals)))
    #
    return tuple(train_idx), tuple(val_idx)


def balanced_cv_split(num_graphs, targets=None, num_classes=10, n_splits=5, random_state=1, shuffle=True):
    """
    A general balanced cv split
    Use: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    See also: https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

    Note: The biggest difference in the logic between this function and paper_cv_fold is that
    this function will split data as close as possible to the target proportion (e.g., 5-fold cv => train: 80%, valid: 20%)
    , while paper_cv_fold will take the floor of (0.2 * num_per_class) as val_size so that all valid folds are of equal size.
    For example, python create_data.py --experiment_data customized -N 31 -L 7 -np 17 --print_splits
    will generate splits as follows (under my random random number generator):
        Fold: 0	Val_size:24     Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([4, 3, 4, 3, 3, 4, 3]))
        Fold: 1	Val_size:24     Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([4, 3, 3, 4, 3, 4, 3]))
        Fold: 2	Val_size:24     Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([3, 4, 3, 4, 3, 3, 4]))
        Fold: 3	Val_size:24     Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([3, 4, 3, 3, 4, 3, 4]))
        Fold: 4	Val_size:23     Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([3, 3, 4, 3, 4, 3, 3]))
    On the other hand, paper_cv_fold (if in use) will generate splits as follows:
        Fold: 0	Val_size:21		Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([3, 3, 3, 3, 3, 3, 3]))
        Fold: 1	Val_size:21		Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([3, 3, 3, 3, 3, 3, 3]))
        Fold: 2	Val_size:21		Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([3, 3, 3, 3, 3, 3, 3]))
        Fold: 3	Val_size:21		Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([3, 3, 3, 3, 3, 3, 3]))
        Fold: 4	Val_size:21		Freq_val_targets:(tensor([0, 1, 2, 3, 4, 5, 6]), tensor([3, 3, 3, 3, 3, 3, 3]))
    In summary, if num_permutations % n_splits != 0, then targets may have different occurrences (by at most one) in different folds;
    if, in addition, num_skip_lengths % n_splits != 0, then val_size may be different (by at most one) for different folds.

    :return: a split indices generator that yields train/valid indices once
    """
    assert isinstance(num_graphs, int) and isinstance(num_classes, int)
    assert num_graphs > 1 and num_classes > 1
    assert num_graphs % num_classes == 0   # May relax later but fine for now
    if targets is not None:
        assert targets.dim() == 1 and targets.size(0) == num_graphs
        assert num_classes == len(torch.unique(targets))
    else:
        # If no target is provided, assume the targets to be [0, 1, ..., num_classes-1]
        num_per_class = int(num_graphs / num_classes)
        targets = torch.tensor([], dtype=torch.long)
        for i in range(num_classes):
            targets = torch.cat((targets, torch.tensor([i]).repeat(num_per_class)), dim=0)

    X_idx = np.zeros(num_graphs)
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    print_stats(f"Splitting data into {skf.get_n_splits(X_idx, targets)} folds...")

    # skf.split returns a generator (that yields train/valid indices)
    return skf.split(X_idx, targets)


def print_splits(cv_indices, targets):
    for fold, idx in cv_indices.items():
        print_stats(f"Fold: {fold}", Train=idx['train'])
    for fold, idx in cv_indices.items():
        train_targets = targets[torch.tensor(idx['train'])]
        print_stats(f"Fold: {fold}", Train_size=len(idx['train']), Freq_train_targets=p_freq_table(train_targets))
    for fold, idx in cv_indices.items():
        print_stats(f"Fold: {fold}", Val=idx['val'])
    for fold, idx in cv_indices.items():
        val_targets = targets[torch.tensor(idx['val'])]
        print_stats(f"Fold: {fold}", Val_size=len(idx['val']), Freq_val_targets=p_freq_table(val_targets))


class RpData:
    def __init__(self, args):
        # Variables that constitute a full RpData "package" for downstream training file.
        # Must be created in member functions before saving to disk
        self.adjmats = list()
        self.y = None
        self.are_onehot = None
        self.cv_indices = OrderedDict()
        self.target_dim = 0
        self.task_type = None

        # Other variables
        self.ed_type = args.experiment_data
        self.graph_function = args.graph_function
        self.swapping_scheme = args.swapping_scheme
        self.do_swapping = args.do_swapping
        self.ordering = args.ordering

        assert isinstance(self.ordering, Canonical_Orderings)
        if self.do_swapping and self.ordering.do_ordering:
            raise ValueError("cannot do swapping and ordering at the same time")

        if self.ed_type in (ExperimentData.rp_paper, ExperimentData.customized):
            self.num_vertices = args.num_vertices
            self.num_skip_lengths = args.num_skip_lengths
            self.num_permutations = args.num_permutations
            self.graph_generator = args.graph_generator
        elif self.ed_type == ExperimentData.smp:
            # Parameters for SMP task are in a dictionary
            # A dictionary setup may be more flexible as we add experiments
            assert isinstance(args.smp_config, dict)
            assert 'cycle_size' in args.smp_config
            assert 'n_vert' in args.smp_config
            assert 'set' in args.smp_config
            assert 'prop' in args.smp_config
            assert 'var_size' in args.smp_config
            if not args.smp_config['var_size']:
                assert args.smp_config['prop'] == 1.
            self.smp_config = args.smp_config
        elif self.ed_type == ExperimentData.er_edges:
            assert isinstance(args.er_edges_config, dict)
            assert 'n_vert' in args.er_edges_config
            assert 'n_graphs' in args.er_edges_config
            assert 'beta_1' in args.er_edges_config
            assert 'beta_2' in args.er_edges_config
            assert 'scale_targets' in args.er_edges_config
            assert 'seed' in args.er_edges_config
            self.er_edges_config = args.er_edges_config
        elif self.ed_type == ExperimentData.ba:
            assert isinstance(args.ba_config, dict)
            assert isinstance(self.graph_function, GraphFunction)
            self.ba_config = args.ba_config
            self.sample_random_state = args.sample_random_state
        elif self.ed_type == ExperimentData.ger:
            assert isinstance(args.ger_config, dict)
            assert isinstance(self.graph_function, GraphFunction)
            self.ger_config = args.ger_config
            self.sample_random_state = args.sample_random_state
        else:
            raise NotImplementedError("Invalid experiment_data enum in RpData")

        self.sparse = args.sparse
        self.recreate_graphs = args.recreate_graphs  # Only used for CSL at the moment
        self.print_splits = args.print_splits
        self.shuffle = args.shuffle  # Used in CV splitting
        self.random_state = args.shuffle_random_state  # Used in CV
        self.n_splits = args.n_splits

        # Build out file name
        self.data_dir = args.data_dir
        self.out_file_name = args.get_data_path()

    def nx_to_np_adj(self, graph):
        """ Convert networkx to numpy adjacency matrix """
        adj = nx.to_numpy_matrix(graph)
        adj += np.eye(adj.shape[0])  # Add ones to the diagonals.
        return adj

    def create_ba(self):
        """
        Create Barabasi-Albert graphs
        """
        graph_function = GraphTasks.get_function(self.graph_function)
        m_kwargs = self.ba_config['m_kwargs']
        ba_kwargs = self.ba_config['ba_kwargs']

        ba_generator = BaData(self.ba_config['n_vert'], self.ba_config['n_graphs'], self.ba_config['m_type'],
                              m_kwargs, ba_kwargs, self.sample_random_state)
        graphs = ba_generator.sample_ba_graphs()
        y_list = list()
        for graph in graphs:
            adj = igraph_to_numpy(graph)
            target = graph_function(adj)
            adj += np.eye(adj.shape[0])
            y_list.append(target)
            self.adjmats.append(adj)

        self.y = torch.FloatTensor(y_list)

    def create_generalized_er(self):
        """
        Create Erdos-Renyi graphs using the generalized wrapper
        """
        graph_function = GraphTasks.get_function(self.graph_function)
        er_generator = ErData(graph_size=self.ger_config['n_vert'],
                              n_graphs=self.ger_config['n_graphs'],
                              p_type=self.ger_config['p_type'],
                              p_kwargs=self.ger_config['p_kwargs'],
                              random_state=self.sample_random_state)

        graphs = er_generator.sample_graphs()
        y_list = list()
        for graph in graphs:
            adj = nx.to_numpy_array(graph)
            target = graph_function(adj)
            y_list.append(target)
            adj += np.eye(adj.shape[0])
            self.adjmats.append(adj)

        self.y = torch.FloatTensor(y_list)

    def create_erdos_renyi_edges(self):
        """
        >> This function should be replaced by create_generalized_er

        Task: Count the number of edges in an Erdos-Renyi graph.
        That's proportional to the sum over the adjacency matrix.
        All graphs are the same size.
        Edge connection probability, is drawn from a beta distribution (just b/c it's more flexible than unif)

        Uses parameters from er_config dictionary:
        > Number of graphs
        > Graph size
        > beta_1 and beta_2: parameters of a beta distribution
        > scale_targets: whether to scale (0-mean, 1-std) the targets
        > seed.
        """
        np.random.seed(self.er_edges_config['seed'])
        random.seed(self.er_edges_config['seed'])  # for networkx.

        y_list = list()
        for ii in range(self.er_edges_config['n_graphs']):
            # Sample edge probability
            edge_prob = np.random.beta(a=self.er_edges_config['beta_1'],
                                       b=self.er_edges_config['beta_2'])
            # Sample erdos-renyi graph
            g = nx.gnp_random_graph(n=self.er_edges_config['n_vert'],
                                    p=edge_prob)
            adj = self.nx_to_np_adj(g)

            # Compute target
            target = g.number_of_edges()

            # Store values
            y_list.append(target)
            self.adjmats.append(adj)

        # Prepare targets for PyTorch
        self.y = torch.FloatTensor(y_list)

        if self.er_edges_config['scale_targets']:
            self.y = scale_targets(self.y)
        else:
            print("\n\nNot centering and scaling the targets..\n\n")

    def create_paper_data(self):
        """Creates the dataset used in Relational Pooling for Graph Representations paper """
        assert os.path.exists(SYNTHETIC_GRAPHS) and os.path.exists(SYNTHETIC_Y), "rp paper data missing!"
        self.adjmats = PickleUtil.read(SYNTHETIC_GRAPHS)
        self.y = torch.load(SYNTHETIC_Y)
        self.target_dim = len(torch.unique(self.y))

    def create_customized_data(self):
        """Creates the customized dataset according to specification"""
        graphs_path = os.path.join(self.data_dir, self.graph_generator._adj_filename())
        # Check if the graph file exist by checking the corresponding file names.
        # Generate the graphs if the specified graph files have not been created or a recreation is required.
        if not os.path.exists(graphs_path) or self.recreate_graphs:
            print_stats("Creating synthetic graphs...")
            self.graph_generator.create_graphs(self.data_dir)
            print_stats("Successfully created synthetic graphs.")
        self.adjmats, self.y = self.graph_generator.load_graphs(self.data_dir)
        self.target_dim = len(torch.unique(self.y))

    def create_smp_data(self):
        """
        Data from Structural Message Passing paper, https://github.com/cvignac/SMP
        Data were processed using their pipeline, now we convert it into our own format.
        Some code taken from https://github.com/cvignac/SMP/blob/master/datasets_generation/build_cycles.py
        """
        fp = f"{self.smp_config['cycle_size']}cycles_n{self.smp_config['n_vert']}_{SMP_NUM_SAMPLES}samples_{self.smp_config['set']}.pt"
        fp = os.path.join(SMP_DATA_DIR, fp)
        if os.path.exists(fp):
            print_stats("Loading SMP data from disk...")
            dataset = torch.load(fp)
            print_stats("..loaded successfully")
        else:
            print("\nEntered arguments for SMP task:")
            for kk, vv in self.smp_config.items():
                print(f"{kk} : {vv}")
            raise FileNotFoundError(f"Input data for SMP task not found at {fp}")
        #
        # Load all adjmats and respective targets y
        #
        y_list = list()
        num_to_look = int(self.smp_config['prop'] * SMP_NUM_SAMPLES)  # Both train and test have same nsamples
        for sample in dataset[:num_to_look]:
            graph, _, label = sample

            if not self.smp_config['var_size']:
                # Skip all graphs of undesired size
                if graph.number_of_nodes() != SMP_SINGLE_GRAPH_SIZE[self.smp_config['n_vert']]:
                    continue

            adj = self.nx_to_np_adj(graph)
            self.adjmats.append(adj)

            y_list.append(1 if label == 'has-kcycle' else 0)

        self.y = torch.FloatTensor(y_list)

    def validation_swapping(self):
        """
        Apply a relabeling (swapping rows/cols) to graphs in the validation data.
        Only applicable when n_splits == 2
        """
        if self.do_swapping:
            assert len(self.cv_indices) == 2
            print_stats("Swapping vertices in validation graphs...")
            idx = self.cv_indices[0]

            for jj in idx['val']:
                adj = self.adjmats[jj]
                if self.swapping_scheme == SwappingScheme.first_two:
                    permutation = np.arange(adj.shape[0])
                    permutation[0] = 1
                    permutation[1] = 0
                else:
                    raise ValueError("invalid swapping scheme")

                new_adj = adj[permutation, :]
                new_adj = new_adj[:, permutation]

                self.adjmats[jj] = new_adj

    def reorder_adjmats(self):
        """
        Reorder graphs in the train and valid data by the specified canonical order.
        Only applicable when n_splits == 2
        """
        if self.ordering.do_ordering:
            assert len(self.cv_indices) == 2
            print_stats("Reordering vertices in training and validation graphs...")
            idx = self.cv_indices[0]

            for ss in ['train', 'val']:
                if self.ordering.canon[ss] is not Canonical_Order.identity:
                    cf = self.ordering.get_canon_function(ss)
                    print_stats(f"Reordering vertices in {ss} by {self.ordering.canon[ss].name}...")
                    for jj in idx[ss]:
                        self.adjmats[jj] = cf(self.adjmats[jj])

    def create_paper_splits(self):
        """Creates the cv splits used in Relational Pooling for Graph Representations paper """
        for fold in range(5):
            train, val = paper_cv_split(len(self.adjmats), fold)
            self.cv_indices[fold] = {'train': train, 'val': val}

    def create_balanced_cv_splits(self):
        """
        Creates general balanced cv splits for customized data
        In case of regression, just create regular splits.
        """
        if self.target_dim == 1 and self.task_type == TaskType.regression:
            print_stats(f"Splitting data into {self.n_splits} (`unbalanced`) folds for regression...")
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            cv_splits = kf.split(np.arange(len(self.adjmats)))
        else:
            if self.target_dim > 1:
                num_classes = self.target_dim
            elif self.target_dim == 1 and self.task_type == TaskType.binary_classification:
                num_classes = 2
            else:
                raise NotImplementedError("Don't know how to create balanced CV for this type")

            cv_splits = balanced_cv_split(len(self.adjmats), targets=self.y, num_classes=num_classes,
                                      n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        for fold, (train, val) in enumerate(cv_splits):
            self.cv_indices[fold] = {'train': tuple(train), 'val': tuple(val)}

    def create_data(self):
        """
        Create data and cross validation indices according to the type of data
        Note, the code was developed and generalized as new tasks got introduced.
        Pay attention to which tasks had "access to" newer features.

        The older tasks can of course be reworked to use the older features
        But there isn't a strong reason to do so unless we need to substantially alter their
        existing functionalities.
        """
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # These tasks are older and (originally) didn't use swapping or task functions
        if self.ed_type == ExperimentData.rp_paper:
            self.are_onehot = True
            self.task_type = TaskType.multi_classification
            self.create_paper_data()
            self.create_paper_splits()
        elif self.ed_type == ExperimentData.customized:
            self.are_onehot = True
            self.task_type = TaskType.multi_classification
            self.create_customized_data()
            self.create_balanced_cv_splits()
        elif self.ed_type == ExperimentData.smp:
            self.are_onehot = True
            self.task_type = TaskType.binary_classification
            self.target_dim = 1
            self.create_smp_data()
            self.create_balanced_cv_splits()
        elif self.ed_type == ExperimentData.er_edges:
            self.are_onehot = True
            self.task_type = TaskType.regression
            self.target_dim = 1
            self.create_erdos_renyi_edges()
            self.create_balanced_cv_splits()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Methods that used swapping and task functions
        elif self.ed_type == ExperimentData.ba:
            self.are_onehot = True
            self.task_type, self.target_dim = GraphTasks.get_task_info(self.graph_function)
            self.create_ba()
            self.create_balanced_cv_splits()
            self.validation_swapping()
            self.reorder_adjmats()
        elif self.ed_type == ExperimentData.ger:
            self.are_onehot = True
            self.task_type, self.target_dim = GraphTasks.get_task_info(self.graph_function)
            self.create_generalized_er()
            self.create_balanced_cv_splits()
            self.validation_swapping()
            self.reorder_adjmats()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        else:
            raise NotImplementedError("Invalid ExperimentData type")

        # Always scale targets if computing determinant: the values explode
        if self.graph_function == GraphFunction.det_adj:
            self.y = scale_targets(self.y)

        if self.print_splits:
            print_splits(self.cv_indices, self.y)

    def save_object(self):
        """ Pickle data object"""
        # Check adjacency list
        assert isinstance(self.adjmats, list) and \
               len(self.adjmats) > 0 and \
               self.adjmats[0].size > 0, "Must contain a non-empty list of adjacency matrices"

        assert isinstance(self.adjmats[0], np.ndarray) or sps.isspmatrix(self.adjmats[0])
        if self.sparse and not sps.isspmatrix(self.adjmats[0]):
            for jj in range(len(self.adjmats)):
                adj = self.adjmats[jj]
                self.adjmats[jj] = coo_matrix(adj)

        # Check cross validation indices
        assert isinstance(self.cv_indices, dict) and \
               len(self.cv_indices) > 0, \
               "Must contain a non-empty dictionary of cross validation indices"

        # Check one-hot.
        assert isinstance(self.are_onehot, bool), "Must contain boolean `are_onehot`"

        # Check target dimension
        assert isinstance(self.target_dim, int) and \
               self.target_dim >= 1, \
               "Target dim must be an integer >= 1"

        # Check targets
        assert torch.is_tensor(self.y), "Must contain responses y as a PyTorch tensor."

        # Check Task type, and that targets are reasonable given the task type
        assert isinstance(self.task_type, TaskType)
        if self.task_type == TaskType.multi_classification:
            assert self.target_dim > 1
            assert self.y.dtype == torch.long
            assert torch.equal(torch.unique(self.y), torch.arange(self.target_dim).long())
        elif self.task_type == TaskType.binary_classification:
            assert self.target_dim == 1
            # WithLogitsLoss requires float targets to accommodate soft labels.
            # https://github.com/pytorch/pytorch/issues/2272#issuecomment-319766578
            assert self.y.dtype == torch.float32
            assert torch.equal(torch.unique(self.y), torch.tensor([0., 1.]))
        elif self.task_type == TaskType.regression:
            assert self.target_dim == 1
            assert self.y.dtype == torch.float32

        # Write to disk
        output = Util.dictize(adjmats=self.adjmats,
                              y=self.y,
                              cv_indices=self.cv_indices,
                              are_onehot=self.are_onehot,
                              file_name=self.out_file_name,
                              target_dim=self.target_dim,
                              task_type=self.task_type)

        pickle.dump(output, open(self.out_file_name, 'wb'))
        print_stats(f"Wrote object to {self.out_file_name}")


if __name__ == "__main__":
    rp_data_args = GraphDataArgParser()
    rp_data = RpData(rp_data_args)
    rp_data.create_data()
    rp_data.save_object()

    print("Done")
