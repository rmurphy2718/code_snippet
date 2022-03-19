from __future__ import division

import hashlib
import itertools
import os
import subprocess
import pickle
import re
import warnings
import traceback
from collections import Counter
from collections import defaultdict
from copy import deepcopy
from multiprocessing.pool import Pool
from time import time
from types import SimpleNamespace

import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sk_met
import torch
from IPython import get_ipython
from filelock import FileLock

try:
    from graph import PARALLELISM
except:
    PARALLELISM = 1
from matplotlib import pyplot as plt
from numba import njit, prange
from numpy import random as rand
from pandas import DataFrame
import scipy.sparse as sps

try:
    ipython = get_ipython()
    if ipython is not None and ipython.config['IPKernelApp']['parent_appname'] == 'ipython-notebook':
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

warnings.simplefilter("ignore", DeprecationWarning)

np.set_printoptions(precision=3, suppress=True)


# --------------------------------------------------------------------------- #
# Simple helper functions
# --------------------------------------------------------------------------- #

def threaded_map(t):
    """
    This has to be a first class function
    :param t:
    :return:
    """
    lmb, obj = t
    return lmb(obj)


class Util:
    _common_pool = None

    @staticmethod
    def modify_edge_dict(tour_edges, weight):
        return {k: v * weight for k, v in tour_edges.items()}

    @staticmethod
    def p_normalize(p):
        return p / p.sum()

    @staticmethod
    def is_numeric(s):
        try:
            float(s)
            return True
        except:
            return False

    @classmethod
    def parallel_map(cls, func, iterable, unused=None, tqdm_str=None, parallel=True, use_tqdm=False):
        if use_tqdm:
            tqdm_fn = lambda x: tqdm(x, desc=tqdm_str)
        else:
            tqdm_fn = lambda x: x
        if parallel:
            p = cls.get_common_pool()
            return p.map(threaded_map, tqdm_fn(list(itertools.product([func], iterable))))
        else:
            return list(map(threaded_map, tqdm_fn(list(itertools.product([func], iterable)))))

    @classmethod
    def get_common_pool(cls):
        if Util._common_pool is None:
            Util._common_pool = Pool(PARALLELISM)
        return Util._common_pool

    rounding_precision = 4

    @staticmethod
    def get_batch_iterator(n, step, use_tqdm=True):
        itr = np.split(np.arange(n), np.arange(step, n, step))
        if use_tqdm:
            return tqdm(itr)
        else:
            return itr

    @staticmethod
    def make_batches(seq_or_len, step, use_tqdm=True, cut=False, desc=None):
        if np.isscalar(seq_or_len):
            ln = seq_or_len
            seq = np.arange(int(ln))
        else:
            seq = seq_or_len
            ln = seq.shape[0]

        itr = np.split(seq, np.arange(step, ln, step))
        if cut:
            if itr[0].shape[0] != itr[-1].shape[0]:
                itr = itr[:-1]
        if use_tqdm:
            return tqdm(itr, desc=desc)
        else:
            return itr

    @classmethod
    def alm(cls, func, iterable):
        """Shorthand"""
        return np.array(list(map(func, iterable)))

    @classmethod
    def lm(cls, func, iterable):
        """Shorthand"""
        return list(map(func, iterable))

    @staticmethod
    def round(v):
        return np.round(v, Util.rounding_precision)

    @staticmethod
    def time_trial(trials: int, methods):
        times = np.zeros([trials, len(methods)])
        for i in range(trials):
            for j, lmb in enumerate(methods):
                start = time()
                lmb()
                end = time()
                times[i, j] = end - start
        return times.mean(axis=0), times.std(axis=0), times.min(axis=0), times.max(axis=0), times

    @staticmethod
    def check_hop_edges(adj_list, samples):
        for s in samples:
            assert s[1] not in adj_list[s[0]]
            assert s[0] not in adj_list[s[1]]
            assert np.intersect1d(adj_list[s[0]], adj_list[s[1]]).shape[0] > 0

    @classmethod
    def ccdf(cls, sorted_arr):
        mass = np.ones_like(sorted_arr)
        sm = mass.sum()
        return sorted_arr, (sm - (mass.cumsum() - mass)) / sm

    @classmethod
    def unsorted_ccdf(cls, unsorted_arr):
        sorted_arr = np.sort(unsorted_arr)
        mass = np.ones_like(sorted_arr)
        sm = mass.sum()
        return sorted_arr, (sm - (mass.cumsum() - mass)) / sm

    @classmethod
    def update_mean(cls, curr, new, ct):
        curr, new, ct = list(map(float, [curr, new, ct]))
        return (curr * ct + new) / (1 + ct)

    @classmethod
    def append_estimate_dict(cls, dict_a, dict_b):
        if dict_a is None:
            return dict_b

        def _app(k):
            a = b = None
            if k in dict_a:
                a = dict_a[k]
            if k in dict_b:
                b = dict_b[k]
            if a is None:
                return b
            if b is None:
                return a

            if isinstance(a, list):
                return a + b
            elif isinstance(a, np.ndarray):
                return np.concatenate((a, b))
            elif isinstance(a, dict):
                a.update(b)
                return a
            else:
                raise NotImplementedError()

        return {k: _app(k) for k in (dict_a.keys() | dict_b.keys())}

    @classmethod
    def append_estimate_tuple(cls, tuple_a, tuple_b):
        if tuple_a is None:
            return tuple_b

        assert len(tuple_a) == len(tuple_b)

        def _app(k):
            a = tuple_a[k]
            b = tuple_b[k]

            if a is None:
                return b
            if b is None:
                return a

            if isinstance(a, list):
                return a + b
            elif isinstance(a, np.ndarray):
                return np.concatenate((a, b))
            elif isinstance(a, dict):
                a.update(b)
                return a
            else:
                raise NotImplementedError()

        return tuple([_app(k) for k in range(len(tuple_a))])

    @classmethod
    def is_none(cls, x, default_lambda):
        return x if x is not None else default_lambda()


    @classmethod
    def process_pivot_table(cls, pt):
        # pt should be a pivot table
        v_mean = pt.loc[:, 'mean'].values
        v_std = pt.loc[:, 'std'].values
        v_count = pt.loc[:, 'count'].values
        pt.loc[:, 'mean'] = np.vectorize(lambda m, s, c: f"{m:.3f} ({s:.3f})[{c:d}]")(v_mean, v_std, v_count)
        del pt['std']
        del pt['count']
        pt.columns = pt.columns.droplevel(0)
        return pt

    @classmethod
    def process_pivot_table_min_max(cls, pt):
        # pt should be a pivot table
        v_mean = pt.loc[:, 'mean'].values
        v_std = pt.loc[:, 'std'].values
        v_count = pt.loc[:, 'count'].values
        v_min = pt.loc[:, 'min'].values
        v_max = pt.loc[:, 'max'].values
        pt.loc[:, 'mean'] = np.vectorize(lambda m, s, c, mn, mx: f"{m:.3f} ({s:.3f}) {{{mn:.3f},{mx:.3f}}}")(v_mean, v_std, v_count, v_min, v_max)
        del pt['std']
        del pt['count']
        del pt['min']
        del pt['max']
        pt.columns = pt.columns.droplevel(0)
        return pt

    def split_rows(df, columns, phase_col, new_col):
        to_keep = set(df.columns) - set(columns)
        base_df = df.loc[:,list(to_keep)]
        dfs = []
        for  c in columns:
            ndf = base_df.copy()
            ndf.loc[:,phase_col] = c
            ndf.loc[:,new_col] = df.loc[:,c]
            dfs.append(ndf)
        return pd.concat(dfs, sort=True).reindex()

    @classmethod
    def not_none(cls, x):
        return list(filter(lambda x: x is not None, x))

    @classmethod
    def coalesce(cls, *args):
        for a in args[:-1]:
            if args[0] is not None:
                return args[0]
        return args[-1]()

    @classmethod
    def create_edge_frame(cls, pos_samples: np.ndarray, neg_samples: np.ndarray) -> np.ndarray:
        def check_dim(arr):
            if arr is None:
                return np.empty(shape=[0, 2], dtype=np.int64)
            elif arr.ndim == 1:
                return arr[None, :].astype(np.int64)
            else:
                return arr.astype(np.int64)

        pos_samples, neg_samples = check_dim(pos_samples), check_dim(neg_samples)
        edges = np.concatenate([pos_samples, neg_samples])
        state = np.concatenate([np.ones(pos_samples.shape[0]), np.zeros(neg_samples.shape[0])])
        return np.c_[edges, state]

    @classmethod
    def create_edge_data_frame(cls, tr, vl, te, g):
        """

        :type g: SimpleGraph
        """
        lbda = lambda pt, ix: pd.DataFrame(np.c_[pt, np.full(pt.shape[0], ix)])
        full_df = pd.concat(list(map(lbda, [tr, vl, te], range(3))))
        full_df.columns = ["u", "v", "e", "t"]
        full_df["t"] = list(map(lambda x: {0: "tr", 1: "vl", 2: "te"}[x], full_df["t"].tolist()))
        full_df["du"] = g.degree[full_df["u"].values.astype(np.int64)]
        full_df["dv"] = g.degree[full_df["v"].values.astype(np.int64)]
        full_df["ix"] = np.arange(full_df.shape[0])
        return full_df

    @classmethod
    def shape(cls, *args):
        for arg in args:
            print(arg.shape)

    @classmethod
    def create_edge_frame_single(cls, edges, positive):
        p_ex = n_ex = None
        if positive:
            p_ex = edges
        else:
            n_ex = edges
        return Util.create_edge_frame(p_ex, n_ex)

    @classmethod
    def get_norms(cls, iterable, norm_type=2):
        parameter_norms = np.array(list(map(lambda x: torch.norm(x, p=norm_type).cpu().numpy(), iterable)))
        return float(np.linalg.norm(parameter_norms, ord=norm_type))

    @classmethod
    def sum_with_budget_limit(cls, l_tup, lim):
        narr = np.array(l_tup)
        return narr[narr[:, 0] <= lim, 1].sum()

    @classmethod
    def uniq_edge(cls, r):
        return int(min(r[0:2])), int(max(r[0:2]))

    @classmethod
    def uniq_edge_s(cls, u, v):
        return min(u, v), max(u, v)

    @classmethod
    def get_min_max_series(cls, c_min, c_max, ser):
        """
        c_min, c_max = Util.get_min_max_series(c_min, c_max, ser)
        :param c_min:
        :param c_max:
        :param ser:
        :return:
        """
        s_min, s_max = np.min(ser), np.max(ser)
        c_min = s_min if c_min is None else min(s_min, c_min)
        c_max = s_max if c_max is None else max(s_max, c_max)
        return c_min, c_max

    @classmethod
    def last_or_none(cls, arr_1):
        if len(arr_1) == 0:
            return None
        else:
            return arr_1[-1]

    @classmethod
    def dictize(cls, **kwargs):
        return kwargs

    @classmethod
    def dict_def_get(cls, a_dict, k, advanced=False):
        if advanced:
            if k in a_dict:
                return a_dict[k]
            else:
                def _anon(x):
                    x = x.strip()
                    if x in a_dict:
                        return a_dict[x]
                    else:
                        lhs = x.split("=")[0].strip()
                        if lhs in a_dict:
                            return a_dict[lhs] + " =" + x.split("=")[1]
                        else:
                            return x

                return " | ".join(map(_anon, k.split("|")))

        if k in a_dict:
            return a_dict[k]
        else:
            return k

    @classmethod
    def safe_sample(cls, x):
        x_sum = x.sum()
        if x_sum == 0:
            p = None  # uniformly sample
        else:
            p = x / x.sum()
        return np.random.choice(x.shape[0], 1, p=p)

    @classmethod
    def select_hyperparameters(cls, df, minimization_metric, tasks, hp, ag_func="mean"):
        gb = df.groupby(tasks + hp).agg({'accuracy': ag_func, 'maccuracy': ag_func, 'mae': ag_func, 'rmse': ag_func})
        opt_index = []
        for ixrs in set(list(map(lambda x: x[:len(tasks)], gb.index))):
            sub_arr = gb
            for _, v in enumerate(ixrs[:len(tasks)]):
                sub_arr = sub_arr.loc[v]

            # this may screw up if multiple hyperparameter settings are being tested
            # cross that bridge when you get there
            selected_params = sub_arr.index.values[np.argmin(sub_arr[minimization_metric].values)]
            opt_index.append(ixrs + (selected_params,))
        return opt_index

    @staticmethod
    def tex_escape(text):
        """
            :param text: a plain text message
            :return: the message escaped to appear correctly in LaTeX
        """
        text = str(text)
        conv = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
            '<': r'\textless',
            '>': r'\textgreater',
        }
        regex = re.compile(
            '|'.join(re.escape(np.unicode(key)) for key in sorted(conv.keys(), key=lambda item: - len(item))))
        return regex.sub(lambda match: conv[match.group()], text)

    @classmethod
    def unique_edges(cls, edges):
        # todo: remove
        edges = np.c_[np.min(edges, 1), np.max(edges, 1)]
        edges = np.unique(edges, axis=0)
        return edges

    @classmethod
    def sample_ndarray(cls, data, ratio=1.0, num_elements=None, replace=True):
        data_length = len(data)
        samples = int(num_elements) if num_elements is not None else int(ratio * data_length)
        return data[np.random.choice(data_length, samples, replace=replace)]

    @staticmethod
    @njit(parallel=True)
    def get_tpr_fpr(pre_sigmoid_input, target):
        tpr = np.zeros_like(pre_sigmoid_input)
        fpr = np.zeros_like(pre_sigmoid_input)
        for i in prange(pre_sigmoid_input.shape[0]):
            t = pre_sigmoid_input[i]
            _negs = np.less(pre_sigmoid_input, t)
            predictions = np.ones_like(_negs)
            predictions[_negs] = 0
            _tpr = (predictions * target).sum() / target.sum()
            _fpr = (predictions * (1 - target)).sum() / ((1 - target).sum())
            tpr[i], fpr[i] = _tpr, _fpr
        fpr = np.sort(fpr)
        tpr = np.sort(tpr)
        auc = np.sum((fpr[1:] - fpr[:-1]) * tpr[1:])
        return auc

    @staticmethod
    @njit(parallel=True)
    def compute_cmrr(degree_set, cmrr, degrees, rr):
        for i in prange(degree_set.shape[0]):
            d = degree_set[i]
            mask = np.greater_equal(degrees, d)
            if np.sum(mask) == 0:
                cmrr[i] = 0
            else:
                cmrr[i] = np.sum(mask * rr) / np.sum(mask)
        return cmrr

    @staticmethod
    def md5(text):
        text = str(text).strip().lower()
        m = hashlib.md5()
        m.update(text.encode())
        return m.hexdigest()


    @staticmethod
    @njit(parallel=True)
    def compute_binned_mrr(degrees, rr):
        degree_set = np.arange(0, np.max(degrees) + 10, 10)
        cmrr = np.zeros_like(degree_set, dtype=float)
        for i in prange(1, degree_set.shape[0]):
            d_low = degree_set[i - 1]
            d_high = degree_set[i]
            mask = np.greater_equal(degrees, d_low) * np.less(degrees, d_high)
            if np.sum(mask) == 0:
                cmrr[i] = 0
            else:
                cmrr[i] = np.sum(mask * rr) / np.sum(mask)
        return degree_set, cmrr

    @classmethod
    def analyze_pred_accuracy(cls, pre_sigmoid_input, target):
        """
        For Testing AUC

        pre_sigmoid_input = torch.from_numpy(np.random.rand(100))
        target = torch.from_numpy(np.random.binomial(1, pre_sigmoid_input))
        BaseModule.analyze_pred_accuracy(pre_sigmoid_input, target.float())
        """
        edge_predicted = (np.sign(pre_sigmoid_input) + 1) / 2
        non_edge_predicted = 1 - edge_predicted
        acc_result = np.equal(edge_predicted, target)
        accuracy = (acc_result.sum(), len(acc_result))
        precision = (edge_predicted[target == 1].sum(), edge_predicted.sum())
        recall = (edge_predicted[target == 1].sum(), target.sum())
        n_precision = (non_edge_predicted[target == 0].sum(), non_edge_predicted.sum())
        n_recall = (non_edge_predicted[target == 0].sum(), (1 - target).sum())
        au_roc = sk_met.roc_auc_score(target, pre_sigmoid_input)
        _pr, _re, _ = sk_met.precision_recall_curve(target, pre_sigmoid_input)
        au_prc = sk_met.auc(_re, _pr)
        return Util.dictize(accuracy=accuracy, precision=precision,
                            recall=recall, n_precision=n_precision, n_recall=n_recall, au_roc=au_roc, au_prc=au_prc)

    @staticmethod
    def simple_tpr_fpr(tpr, fpr, pre_sigmoid_input, target):
        for i in range(pre_sigmoid_input.shape[0]):
            t = pre_sigmoid_input[i]
            predictions = np.greater_equal(pre_sigmoid_input, t)
            predictions[predictions != 1] = 0
            _tpr = (predictions * target).sum() / target.sum()
            _fpr = (predictions * (1 - target)).sum() / ((1 - target).sum())
            tpr[i], fpr[i] = _tpr, _fpr
        return tpr, fpr

    @staticmethod
    @njit
    def arr_remove(arr, val):
        return arr[arr != val]


class Jacknife:
    @classmethod
    def get_mean_sd(cls, ser):
        n = float(len(ser))
        den = n - 1
        total = np.sum(ser)
        loo_estimates = np.apply_along_axis(lambda i: (total - i) / den, 0, ser)
        mean = np.mean(loo_estimates)
        std = np.std(loo_estimates) * np.sqrt(den / n)
        return mean, std


class Standard:
    @classmethod
    def get_mean_sd(cls, ser):
        mean = np.mean(ser)
        std = np.std(ser)
        return mean, std


class DataFrameBuilder:
    def __init__(self):
        self.df = defaultdict(list)

    def add_row(self, **kwargs):
        for k, v in kwargs.items():
            self.df[k].append(v)
        return kwargs

    def get_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.df)


def sync_data_frames(training_data_frame, test_data_frame):
    tr_col = set(training_data_frame.columns)
    te_col = set(test_data_frame.columns)
    absent_columns = tr_col - te_col
    for col in absent_columns:
        test_data_frame[col] = [0.0 for _ in test_data_frame.index]
    extra_columns = te_col - tr_col
    for col in extra_columns:
        test_data_frame = test_data_frame.drop(col)
    return test_data_frame


def get_feature_name(*args):
    return "_".join(map(str, args))


def tsum(tup_a, tup_b):
    return tuple([a + b for a, b in zip(tup_a, tup_b)])


def tdiv(a_tup):
    return 0.0 if a_tup[1] == 0 else float(a_tup[0]) / float(a_tup[1])


def select_best(df, eval, config):
    """
    eval = {'accuracy':True, 'loss':False, 'precision':True, 'recall':True}
    config = ['lr','weight_decay', 'batch_size']):
    :param df:
    :param eval:
    :param config:
    :return:
    """
    m_epoch = max([float(e) for e in df["epoch"]])
    vdf = df.loc[(df["type"] == "v") & (df["epoch"] == m_epoch), :]
    vdf["config"] = ["|".join([str(r[c]) for c in config]) for _, r in vdf.iterrows()]
    ev_df = vdf.groupby("config").agg({k: 'mean' for k in eval.keys()})
    for key, rev in eval.items():
        ordered = sorted(dict(ev_df[key]).items(), key=lambda t: t[1], reverse=rev)
        ev_df[key] = pd.Series({opt: rank for rank, (opt, _) in enumerate(ordered)})
    ev_df["mean_rank"] = [pd.np.mean([r[c] for c in eval.keys()]) for _, r in ev_df.iterrows()]
    print_stats("Evaluation Data Frame", ev_df=ev_df)
    best_option = sorted(dict(ev_df["mean_rank"]).items(), key=lambda t: t[1])[0]
    print_stats("Best Option", best_option=best_option)
    return tuple(best_option[0].split("|")), ev_df


def print_stats(title, **kwargs):
    _str = f"{time()}\t{title}\t"
    for k, v in kwargs.items():
        _str += f"{k}:{v}\t"
    tqdm.write(_str)


def log_stats(logger_obj, title="", **kwargs):
    _str = f"{time()}\t{title}\n"
    for k, v in kwargs.items():
        _str += f"{k}: {v}\n"

    logger_obj.str_add(_str)

def RMSE(preds, truth):
    return np.sqrt(np.mean(np.square(preds - truth)))


def kuiper_loss(p: Counter, q: Counter):
    p /= sum(p.values())
    q /= sum(q.values())
    keys = sorted(list(set(p.keys()) | set(q.keys())))
    d_pos = d_neg = 0
    p_cum = q_cum = 0

    for i in keys:
        p_cum += p[i]
        q_cum += q[i]
        d_pos = max(d_pos, p_cum - q_cum)
        d_neg = max(d_neg, q_cum - p_cum)
    return d_pos + d_neg


def compute_average(iterable_obj, init):
    return sum(iterable_obj, init) / float(len(iterable_obj))


def safe_cmp(a, b):
    if a is None:
        return -1
    elif b is None:
        return 1
    else:
        return a - b


def assert_equal(a, b, message=""):
    """Check if a and b are equal."""
    assert a == b, "Error: %s != %s ! %s" % (a, b, message)
    return


def assert_le(a, b, message=""):
    """Check if a and b are equal."""
    assert a <= b, "Error: %d > %d ! %s" % (a, b, message)
    return


def assert_ge(a, b, message=""):
    """Check if a and b are equal."""
    assert a >= b, "Error: %d < %d ! %s" % (a, b, message)
    return


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Check approximate equality for floating-point numbers."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def is_unique(l):
    """Check if all the elements in list l are unique."""
    assert type(l) is list, "Type %s is not list!" % type(l)
    return len(l) == len(set(l))


def rand_bern(p=.5, size=1):
    """Generate Bernoulli random numbers."""
    return rand.binomial(n=1, p=p, size=size)


def rand_choice(d, size=None):
    """Convenience wrapper for rand.choice to draw from a probability dist."""
    assert type(d) is Counter or type(d) is dict
    assert isclose(sum(d.values()), 1.)  # Properly normalized
    return rand.choice(a=d.keys(), p=d.values(), size=size)


def filterNone(l):
    """Returns the list l after filtering out None (but 0's are kept)."""
    return [e for e in l if e is not None]


def flatten(l, r=1):
    """Flatten a nested list/tuple r times."""
    return l if r == 0 else flatten([e for s in l for e in s], r - 1)


def deduplicate(seq):
    """Remove duplicates from list/array while preserving order."""
    seen = set()
    seen_add = seen.add  # For efficiency due to dynamic typing
    return [e for e in seq if not (e in seen or seen_add(e))]


def most_common(cnt):
    """
    Probabilistic version of Counter.most_common() in which the most-common
    element is drawn uniformly at random from the set of most-common elements.
    """
    assert type(cnt) == Counter or type(cnt) == dict

    max_count = max(cnt.values())
    most_common_elements = [i for i in cnt.keys() if cnt[i] == max_count]
    return rand.choice(most_common_elements)


def normalize(counts, alpha=0.):
    """
    Normalize counts to produce a valid probability distribution.

    Args:
        counts: A Counter/dict/np.ndarray/list storing un-normalized counts.
        alpha: Smoothing parameter (alpha = 0: no smoothing;
            alpha = 1: Laplace smoothing).

    Returns:
        A Counter/np.array of normalized probabilites.
    """
    if type(counts) is Counter:
        # Returns the normalized counter without modifying the original one
        temp = sum(counts.values()) + alpha * len(counts.keys())
        dist = Counter({key: (counts[key] + alpha) / temp
                        for key in counts.keys()})
        return dist

    elif type(counts) is dict:
        # Returns the normalized dict without modifying the original one
        temp = sum(counts.values()) + alpha * len(counts.keys())
        dist = {key: (counts[key] + alpha) / temp for key in counts.keys()}
        return dist

    elif type(counts) is np.ndarray:
        temp = sum(counts) + alpha * len(counts)
        dist = (counts + alpha) / temp
        return dist

    elif type(counts) is list:
        return normalize(np.array(counts))

    else:
        raise NameError("Input type %s not understood!" % type(counts))


def num_le(nums, target):
    """
    Given an ascending array and a target value, return the number of elements
        in the array that is less than or equal to the target value.
    """
    # TODO(jiasen): Binary search solution ...

    assert all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)), \
        "Input list is not sorted ascendingly!"

    if target >= nums[-1]:
        return len(nums)

    i = 0

    for i, num in enumerate(nums):
        if num > target:
            return i

    return i


def es_list_to_vs_list(es):
    """
    Args:
        es: a list of 2-tuples.

    Returns:
        A list of ints.
    """
    if not es:
        return list()

    assert isinstance(es[0], tuple)

    return [es[0][0]] + [e[1] for e in es]


def sparse_ndarray_to_sparse_tensor(nd_array, type="float"):
    rows, cols, vals = sps.find(nd_array)  # indices of nonzero rows and cols
    indx_tens = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)], dim=0)
    if type == "float":
        # One may think this should be an int tensor, but we cannot multiply ints with floats in PyTorch
        return torch.sparse.FloatTensor(indx_tens,
                                        torch.tensor(vals, dtype=torch.float32))
    else:
        raise NotImplementedError("Can only convert to a torch.sparse.FloatTensor")

class GitInfo:

    # Ref: https://stackoverflow.com/a/24584384
    @staticmethod
    def is_git_directory(path='.'):
        try:
            return subprocess.call(['git', '-C', path, 'status'], stderr=subprocess.STDOUT, stdout=open(os.devnull, 'w')) == 0
        except:
            # If something goes wrong when getting git info, simply output the error massage and return "not available".
            traceback.print_exc()
            return False

    @staticmethod
    def get_current_branch_name():
        try:
            return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('utf-8').strip()
        except:
            traceback.print_exc()
            return "NO available current branch name!"

    @staticmethod
    def get_current_commit_hash():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode('utf-8').strip()
        except:
            traceback.print_exc()
            return "NO available current commit hash!"


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #


def query_neighbors(v, frac, nmax, quota):
    """
    Query min(frac * len(v.neighbors()), nmax, quota) neighbors of v
    to facilitate later computation of stochastic estimate for the
    proportion aggregation.

    Args:
        v: ig.Vertex, a node.
        frac: maximum fraction of neighbors selected.
        nmax: maximum number of neighbors selected.
        quota: current available quota in total.

    Returbs:
        A list of queried node id'sample (int).
    """
    assert type(v) == ig.Vertex

    neighbors = list(set(v.neighbors()))
    num = min([int(round(frac * len(neighbors))), nmax, quota])
    if num <= 0:  # No neighbors queried
        return set()

    assert_le(num, len(neighbors), "sampling with replacement not possible")
    queried_vs = rand.choice(neighbors, num, replace=False)
    queried = [u.index for u in queried_vs]

    return set(queried)


def label_nodes(g, a, p=None, unlabeled=None, _label='_label'):
    """
    Creates a partial-labeling _label based on attribute a for the nodes in g.

    Args:
        g: DataGraph, the input graph.
        a: str, an attribute of g using which _label will be created.
        p: An optional float indicating the proportion labeled.
            Neglected if unlabeled is provided.
        unlabeled: An optional list of specified unlabeled node ids.
            If not provided, the unlabeled nodes will be randomly selected.
        _label: str, the label attribute.
        :type p: float
    """
    n = g.vcount()
    if unlabeled is None:
        assert p is not None
        unlabeled = rand.choice(n, int((1 - p) * n))
    else:
        assert p is None

    g.vs[_label] = deepcopy(g.vs[a])
    for i in unlabeled:
        g.vs[i][_label] = None

    return


def accuracy(pred_labels, true_labels, test_vs):
    """
    Computes classification accuracy on test vertices.

    Args:
        pred_labels: A list/dict of predicted labels for each node.
        true_labels: A list/dict of ground-truth labels for each node.
        test_vs: A list of test node ids (int) over which the accuracy
            will be measured.

    Returns:
        acc: (int) classification accuracy on test_vs.
    """
    acc = sum(pred_labels[i] == true_labels[i] for i in test_vs) / len(test_vs)

    return acc


def rmse_probs_probs(pred_probs, true_probs, test_vs):
    """
    Computes classification accuracy on test vertices.

    Args:
        pred_probs: A dict of predicted probabilities for each node.
        true_probs: A dict of ground-truth probabilities for each node.
        test_vs: A list of test node ids (int) over which the accuracy
            will be measured.

    Returns:
        acc: (int) classification accuracy on test_vs.
    """
    temp = 0
    for i in test_vs:
        pred_dict = pred_probs[i]
        true_dict = true_probs[i]
        assert isinstance(pred_dict, dict)
        assert isinstance(true_dict, dict)
        temp += sum((pred_dict[key] - true_dict[key]) ** 2
                    for key in true_dict.keys())

    return np.sqrt(temp / len(test_vs))


def rmse_probs_labels(pred_probs, true_labels, test_vs):
    """
    Computes classification accuracy on test vertices.

    Args:
        pred_probs: A dict of predicted probabilities for each node.
        true_labels: A list/dict of ground-truth probabilities for each node.
        test_vs: A list of test node ids (int) over which the accuracy
            will be measured.

    Returns:
        acc: (int) classification accuracy on test_vs.
    """
    vals = set(true_labels.values()) if isinstance(true_labels, dict) \
        else set(true_labels)

    true_probs = {i: {val: 1 * (val == true_labels[i]) for val in vals}
                  for i in test_vs}

    return rmse_probs_probs(pred_probs, true_probs, test_vs)


# --------------------------------------------------------------------------- #
# Kernel functions
# --------------------------------------------------------------------------- #

# Reference: # http://www.cs.toronto.edu/~duvenaud/cookbook/index.html

def exponential_kernel(dt, tau, type='pdf'):
    """
    Exponential kernel.

    Args:
        dt: Input variable.
        tau: Length scale (square-root).
    """
    if type == 'pdf':
        return 1 / tau * np.exp(-dt / tau)
    elif type == 'cdf':
        return 1 - np.exp(-(dt / tau))
    else:
        raise NameError('Kernel type not understood')


def periodic_kernel(dt, p, tau):
    """
    Periodic kernel.

    Args:
        dt: Input variable.
        p: Period.
        tau: Length scale (square-root).
    """
    return np.exp(-2 * np.sin(np.pi * dt / p) ** 2 / tau)


def local_periodic_kernel(dt, p, tau):
    """
    Local periodic kernel with exponential decay.

    Args:
        dt: Input variable.
        p: Period.
        tau: Length scale (square-root).
    """
    return exponential_kernel(dt, tau) * periodic_kernel(dt, p, tau)


# --------------------------------------------------------------------------- #
# Plotting setup
# --------------------------------------------------------------------------- #

font = {'size': 15}
plt.rc('font', **font)
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('text', usetex=True)

_RED = '#F8766D'
_GREEN = '#7CAE00'
_BLUE = '#00BFC4'
_PURPLE = '#C77CFF'
_ORANGE = '#FFA500'

_red = '#D7191C'
_orange = '#FDAE61'
_green = '#ABDDA4'
_blue = '#2B83BA'
_brown = '#A6611A'
_gold = '#DFC27D'
_lblue = '#ABD9E9'
_lgreen = '#80CDC1'

# Color wheel
_COLORS = [_BLUE, _RED, _GREEN, _PURPLE, _ORANGE, _blue, _brown, _gold, _lblue,
           _lgreen, _red, _green, 'navy', 'green', 'blue', 'magenta', 'yellow']


# --------------------------------------------------------------------------- #
# Simple parallelism using Pool
# --------------------------------------------------------------------------- #


def parallel(f, sequence, processes):
    pool = Pool(processes)
    # pool = ThreadPool()
    result = pool.map(f, sequence)
    pool.close()
    pool.join()

    return [res for res in result if res is not None]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
class PickleUtil:
    @classmethod
    def write(cls, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    @classmethod
    def check_create_dir(cls, d):
        if not os.path.exists(d):
            os.makedirs(d)
            print_stats(f"Created d={d}")
        else:
            print_stats(f"Exists d={d}")

    @classmethod
    def read(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    @classmethod
    def read_older(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    @classmethod
    def write_text(cls, data, filename):
        with open(filename, 'w') as f:
            f.write(str(data))


class NumPickle:
    @staticmethod
    def save(filename, **kwargs):
        np.savez_compressed(filename, **kwargs)

    @staticmethod
    def load(filename, allow_pickle=True, npz=True):
        ext = "npz" if npz else "npl"
        with FileLock(f"{filename}.{ext}.lock"):
            return SimpleNamespace(**np.load(f"{filename}.{ext}", allow_pickle=allow_pickle))

    @staticmethod
    def exists(filename, npz=True):
        ext = "npz" if npz else "npl"
        return os.path.exists(f"{filename}.{ext}")


class RunningMean:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def m(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count

    def a(self, v):
        self.sum += v
        self.count += 1


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

class Plotter:
    @classmethod
    def SRQ_plots(cls, tour_length_map):
        df = defaultdict(list)
        hue = "Key"
        x = "i/n"
        y = "Ti/Tn"
        for (num_tours, typekey, level), tour_lengths in tour_length_map.items():
            n = len(tour_lengths)
            tn = sum(tour_lengths)
            ti = 0
            for i, t in enumerate(tour_lengths):
                df["numtours"].append(num_tours)
                df["typekey"].append(typekey)
                df["level"].append(level)
                df[x].append(1.0 * i / n)
                ti += t
                df[y].append(1.0 * ti / tn)

        df = DataFrame(df)
        print_stats(len(df))
        df[x] = list(map(float, df[x]))
        df[y] = list(map(float, df[y]))
        sns.factorplot(x=x, data=df, y=y, hue="level", row="numtours", col="typekey")
        plt.savefig("temp.png")


if __name__ == "__main__":
    print(GitInfo.is_git_directory())
    print(GitInfo.is_git_directory("/media/TI10716100D/"))
    print(GitInfo.is_git_directory(5.3))

    print(GitInfo.get_current_branch_name())

    print(GitInfo.get_current_commit_hash())

