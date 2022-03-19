from enum import Enum
import numpy as np

# Seed used to generate graphs. Generation scheme depends on task.
DEFAULT_RANDOM_GENERATION_SEED = 1  # Do NOT change for this project!  If default, we don't add to filepath.
SMP_NUM_SAMPLES = 10000  # Both train and test have same nsamples
MAX_BATCH = 300  # moved from config; outdated?

class Activation(Enum):
    ReLU = 1
    Tanh = 2
    Sigmoid = 3


class GinModelType(Enum):
    regularGin = 1
    dataAugGin = 2
    rpGin = 3


# Some aliases have been added to the Enums for shorter filenames
class ExperimentData(Enum):
    rp_paper = 1
    customized = 2
    smp = 3  # From structural message passing paper
    er_edges = 4  # Erdos Renyi Edges
    ba = 5  # Barabasi-Albert

    ger = 6  # General Erdos Renyi (replaces er_edges but leaving for compatibility)


class LType(Enum):
    # meaningless: analytic_prime_l = 1
    analytic_int_l = 2
    brute = 3


class ReplicatePermType(Enum):
    sampled = 1
    deterministic = 2


class TaskType(Enum):
    binary_classification = 1
    multi_classification = 2
    regression = 3


class GraphFunction(Enum):
    first_degree = 1
    max_degree = 2
    det_adj = 3


# MType for Bernoulli Data
class BaM(Enum):
    const = 1
    constant = 1
    bern = 2
    bernoulli = 2  # The number of new edges from (1+Bernoulli) after some vertices added: prevent tree, more realistic


# Probability for Erdos-Renyi
class ErP(Enum):
    const = 1
    constant = 1

    rand = 2
    random = 2



class SwappingScheme(Enum):
    first_two = 1

class C:
    """Class of plotting constants"""
    search_complete = "search_complete"
    error_type = "error_type"
    phase = "phase"
    total_samples = "total_samples"
    average_samples = "average_samples"
    method = "method"
    order = "order"
    sl = "sl"
    sp = "sp"
    isamp = "isamp"
    rmse = "rmse"
    Tr = "Tr"
    Vl = "Vl"
    train = "train"
    test = "test"
    validation = "validation"
    dt = "dt"
    Te = "Te"
    lr = "lr"
    min = "min"
    mean = "mean"
    black = "black"
    purple = "purple"
    epoch = "epoch"
    total_training_time = "total_training_time"
    ix = "ix"
    mae = "mae"
    nrmse = "nrmse"
    batches = "batches"
    mse = "mse"
    accuracy = "accuracy"
    maccuracy = "maccuracy"
    mawce = "mawce"


class Result(Enum):
    """ For analysis, which type of file? """
    stats = 0
    output = 1

class InputType(Enum):
    set = 1
    graph = 2

class ModelType(Enum):
    rpgin = 1
    lstm = 2
    transformer = 3

class Regularization(Enum):
    """ Type of regularization to use"""
    none = 1
    pw_diff = 2
    perm_grad = 3
    center = 4
    diff_step_center = 4
    edge = 5
    diff_step_edge = 5
    basis = 6
    diff_step_basis = 6


class Task(Enum):
    sum = 1
    max_xor = 2
    median = 3
    prod_median = 4
    longest_seq = 5
    k_ary_distance = 6
    var = 7

    def vector_task(self):
        return self in {self.k_ary_distance}


class ImportanceSample(Enum):
    none = 0
    is_sequence = 1
    is_permutation = 2
    r_permutation = 3
    os_permutation = 4
    m_permutation = 5
    f_permutation = 6
    hf_permutation = 7


class LossFunction(Enum):
    l1 = 0
    mse = 1


class Model(Enum):
    lstm = 1
    gru = 2
    mlp = 3


class Aggregation(Enum):
    last = 1
    attention = 2
    summation = 3


class Constants:
    FOLD_COUNT = 6
    GS_INTERVAL = 50  # frequency of dumping a snapshot of the model to disk
    GS_SEQUENCES = 10000  # sequences sampled to compute norm distributions
    GS_PERMUTATION_BASE = 10  # sequences sampled to compute norm distributions across permutations
    GS_PERMUTATION_COUNT = 1000  # permutations sampled of the above sequences

    MNIST_SEQ_LENS = (5 + 5 * np.arange(10)).tolist()
    MNIST_TASKS = [Task.sum, Task.var, Task.max_xor, Task.median, Task.prod_median, Task.longest_seq]

    VECTOR_SEQ_LENS = [100,200]
    VECTOR_TASKS = [Task.k_ary_distance]
    VECTOR_K = [2,3]
    VECTOR_D = [2,5,8,10]

    INFERENCE_PERMUTATIONS = 20
    LSTM_HIDDEN = 50
    GRU_HIDDEN = 80
    MLP_HIDDEN = 30

    NORM_TYPE = 2

    NAME_MAX = 255 # filename length limit of Linux systems
    LONGEST_FORMAT = 19 # len(".checkpoint.pth.tar")
    MODEL_ID_MAX = NAME_MAX - LONGEST_FORMAT

    @staticmethod
    def get_batch_limit(nograd=False):
        if nograd:
            return 5000
        else:
            return 30

    @staticmethod
    def get_dataset_size(task:Task):
        # returns n_tr, n_te
        if task.vector_task():
            return 12000, 2000
        else:
            # Supports sequences of less than 50 length
            # Each fold will be 20000 sequences maximum and test_examples will be an aditional 20000 sequences
            #return 120000, 20000
            return 1200, 200



class DT(Enum):
    train = 0
    validation = 1
    test = 2
