import itertools
from enum import Enum
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from common.helper import Util


class MC(Enum):
    b = 1
    p = 2
    p_mh = 3
    kb = 4
    kb_mh = 5
    ke = 6
    ke_mh = 7
    gr = 8
    sp = 9
    sp_mh = 10

    def get(self):
        return {
            self.b: BaseMC,
            self.p: Permutahedron,
            self.p_mh: PermutahedronMHRW,
            self.kb: GenPermutahedron,
            self.kb_mh: GenPermutahedronMHRW,
            self.ke: RGenPermutahedron,
            self.ke_mh: RGenPermutahedronMHRW,
            self.gr: GroupMC,
            self.sp: SquarePermutahedron,
            self.sp_mh: SquarePermutahedronMHRW,
        }[self]

    @classmethod
    def mc_name(cls, x):
        try:
            pretty_name = {
                cls.b: "Ours",
                cls.p: 'Permutahedron',
                cls.p_mh: 'Permutahedron',
                cls.kb: '4-ary start',
                cls.kb_mh: '4-ary start',
                cls.ke: '4-ary end',
                cls.ke_mh: '4-ary end',
                cls.gr: '4-ary jump',
                cls.sp: 'Square Permutahedron',
                cls.sp_mh: 'Square Permutahedron',
            }[cls[x]]
            mh = x.endswith("_mh")
        except KeyError:
            pretty_name = 'NA'
            mh = False
        return pretty_name, mh


class BaseMC:
    # Base Markov Chain
    n_steps = 2

    _permutation_matrix = None

    @classmethod
    def get_permutation_generator(cls, sl):
        if cls._permutation_matrix is None:
            rng = np.arange(sl)
            permutation_matrix = []
            for i in range(sl):
                pm = (np.eye(sl, dtype=int) * 2 - 1)
                pm[pm == 1] = i
                pm[pm == -1] = np.tile(rng[rng != i], sl)
                permutation_matrix.append(pm)
            cls._permutation_matrix = np.array(permutation_matrix)
        return cls._permutation_matrix

    @classmethod
    def initialize(cls, x):
        return np.apply_along_axis(np.random.permutation, 1, x)  # uniformly sample a starting state

    @classmethod
    def ss_p(cls, x, y, grad_fn):
        """
        :param x: Sequences
        :param y: outputs
        :param grad_fn: gradient norm function
        :return:
        the steady state probabilities of eaxh state x
        """
        return grad_fn(x, y)

    @classmethod
    def _half_step(cls, x, t, y, grad_fn):
        """
        Given a set of permutations and a gradient norm generating function,
        samples a transition according to the MC with stationary distribution equivalent to the ISMC
        Is supposed to be used with either tours with restarts, frontier sampling or on its own
        :param x: Sequences
        :param y: outputs
        :param t: if =-1 it is a real node, else it is a virtual node
        :param grad_fn: gradient norm function
        :return:
        new_x: new frontier
        new_t: new vertex type
        d: the degree of the last state
        ss_p: steady state probability of new_x
        """
        bs, sl = x.shape
        d = np.zeros(bs)
        ss_p = np.zeros(bs)
        new_t = t.copy()
        new_x = x.copy()

        real_count = t[t == -1].shape[0]
        virt_count = bs - real_count
        if real_count > 0:  # real nodes
            new_t[t == -1] = np.random.choice(sl, real_count)
            n = grad_fn(x[t == -1], y[t == -1])
            d[t == -1] = n * sl
            ss_p[t == -1] = n
            # next state permutation remains unchanged

        if virt_count > 0:  # virtual nodes
            # Compute neighbors
            sampled_positions = t[t != -1]
            pm = cls.get_permutation_generator(sl)
            fx_perms = Util.alm(lambda i: x[t != -1][i][pm[sampled_positions[i]]], range(virt_count))
            fx_perms = fx_perms.reshape((-1, sl))

            # Get norms for gradients
            f_norms = grad_fn(fx_perms, np.repeat(y[t != -1], sl, 0))

            # Organize as norms per tour and sample
            fnrs = f_norms.reshape((-1, sl))
            choices = np.apply_along_axis(Util.safe_sample, 1, fnrs).squeeze()

            # store degree, edge weight and next state of the mc
            ss_p[t != -1] = f_norms[np.arange(0, fx_perms.shape[0], sl) + choices]
            d[t != -1] = fnrs.sum(axis=1)
            new_x[t != -1] = fx_perms[np.arange(0, fx_perms.shape[0], sl) + choices]
            new_t[t != -1] = -1
        return new_x, new_t, d, ss_p

    @classmethod
    def step(cls, x, y, grad_fn, t=None):
        states = []
        if t is None:
            t = np.full(x.shape[0], -1)
        for _ in range(cls.n_steps):
            x, t, d, ss_p = cls._half_step(x, t, y, grad_fn)  # MC step, x and t change with every step
            states.append(SimpleNamespace(x=x, t=t, d=d, ss_p=ss_p))
        return states


class GroupMC(BaseMC):
    # Niepert's MC
    n_steps = 2

    _permutation_matrix = None

    @classmethod
    def get_permutation_generator_new(cls, sl, k):
        if cls._permutation_matrix is None:
            rng = np.arange(sl)
            permutation_matrix = []
            for i in range(sl):
                pm = (np.eye(sl, dtype=int) * 2 - 1)
                pm[pm == 1] = i
                pm[pm == -1] = np.tile(rng[rng != i], sl)
                permutation_matrix.append(pm[k:])
            cls._permutation_matrix = np.array(permutation_matrix)
        return cls._permutation_matrix

    @classmethod
    def _half_step(cls, x, t, y, grad_fn):
        """
        Given a set of permutations and a gradient norm generating function,
        samples a transition according to the MC with stationary distribution equivalent to the ISMC
        Is supposed to be used with either tours with restarts, frontier sampling or on its own
        :param x: Sequences
        :param y: outputs
        :param t: if =-1 it is a real node, else it is a virtual node
        :param grad_fn: gradient norm function
        :return:
        new_x: new frontier
        new_t: new vertex type
        d: the degree of the last state
        ss_p: steady state probability of new_x
        """
        bs, sl = x.shape
        d = np.zeros(bs)
        ss_p = np.zeros(bs)
        new_t = t.copy()
        new_x = x.copy()

        k = int(sl / 2)  #

        real_count = t[t == -1].shape[0]
        virt_count = bs - real_count
        if real_count > 0:  # real nodes
            new_t[t == -1] = np.random.choice(sl - k, real_count) + k  #
            n = grad_fn(x[t == -1], y[t == -1])
            d[t == -1] = n * sl
            ss_p[t == -1] = n
            # next state permutation remains unchanged

        if virt_count > 0:  # virtual nodes
            # Compute neighbors
            sampled_positions = t[t != -1]

            #
            pm = cls.get_permutation_generator_new(sl, k)
            temp_x = x[t != -1]
            temp_x = np.concatenate([np.apply_along_axis(np.random.permutation, 1, temp_x[:, :-k]), temp_x[:, -k:]], 1)
            fx_perms = Util.alm(lambda i: temp_x[i][pm[sampled_positions[i]]], range(virt_count))
            fx_perms = fx_perms.reshape((-1, sl))
            #

            # Get norms for gradients
            f_norms = grad_fn(fx_perms, np.repeat(y[t != -1], sl - k, 0))  #

            # Organize as norms per tour and sample
            fnrs = f_norms.reshape((-1, sl - k))  #
            choices = np.apply_along_axis(Util.safe_sample, 1, fnrs).squeeze()

            # store degree, edge weight and next state of the mc
            ss_p[t != -1] = f_norms[np.arange(0, fx_perms.shape[0], sl - k) + choices]
            d[t != -1] = fnrs.sum(axis=1)
            new_x[t != -1] = fx_perms[np.arange(0, fx_perms.shape[0], sl - k) + choices]
            new_t[t != -1] = -1
        return new_x, new_t, d, ss_p

    @classmethod
    def step(cls, x, y, grad_fn, t=None):
        states = []
        if t is None:
            t = np.full(x.shape[0], -1)
        for _ in range(cls.n_steps):
            x, t, d, ss_p = cls._half_step(x, t, y, grad_fn)  # MC step, x and t change with every step
            states.append(SimpleNamespace(x=x, t=t, d=d, ss_p=ss_p))
        return states


class _Associatahedron(BaseMC):
    # Permutahedron
    n_steps = 1
    mhrw = False

    @classmethod
    def get_permutation_generator(cls, sl):
        raise NotImplementedError()

    @classmethod
    def ss_p(cls, x, y, grad_fn):
        """
        :param x: Sequences
        :param y: outputs
        :param grad_fn: gradient norm function
        :return:
        the steady state probabilities of eaxh state x
        """
        if cls.mhrw:
            return grad_fn(x, y)
        else:
            d, _, _ = cls._get_degree_neighborhood(x, y, grad_fn)
            return d

    @classmethod
    def _get_degree_neighborhood(cls, x, y, grad_fn, smooth=True):
        bs, sl = x.shape
        pm = cls.get_permutation_generator(sl)
        per_x_cnt = pm.shape[0]
        all_pms = Util.alm(lambda i: x[i][pm], range(bs))

        # Get norms for gradients
        all_norms = grad_fn(all_pms.reshape((-1, sl)), np.repeat(y, per_x_cnt, 0))
        all_norms = all_norms.reshape((-1, per_x_cnt))

        # Organize as norms per tour and sample

        neighbor_pms = all_pms[:, 1:, :]
        if smooth:
            neighbor_norms = (all_norms[:, 1:] + all_norms[:, 0, None]) / 2
            d = all_norms[:, 0] + all_norms[:, 1:].mean(axis=1)
        else:
            neighbor_norms = all_norms[:, 1:]
            d = all_norms[:, 1:].mean(axis=1)
        return d, neighbor_pms, neighbor_norms

    @classmethod
    def _half_step(cls, x, t, y, grad_fn):
        """
        Given a set of permutations and a gradient norm generating function,
        samples a transition according to the MC with stationary distribution equivalent to the ISMC
        Is supposed to be used with either tours with restarts, frontier sampling or on its own
        :param x: Sequences
        :param y: outputs
        :param t: if =-1 it is a real node, else it is a virtual node
        :param grad_fn: gradient norm function
        :return:
        new_x: new frontier
        new_t: new vertex type
        d: the degree of the last state
        ss_p: steady state probability of new_x
        """
        if cls.mhrw:
            d, neighbor_pms, neighbor_norms = cls._get_degree_neighborhood(x, y, grad_fn, False)
            choices = np.apply_along_axis(Util.safe_sample, 1, neighbor_norms).squeeze()

            sl = x.shape[1]
            bs, per_x_cnt = neighbor_norms.shape

            x_candidate = neighbor_pms.reshape(-1, sl)[np.arange(0, bs * per_x_cnt, per_x_cnt) + choices]
            x_candidate_degree, _, _ = cls._get_degree_neighborhood(x_candidate, y, grad_fn, False)
            ap = d / x_candidate_degree
            a = (np.random.uniform(size=ap.shape[0]) < ap).astype(int)[:, None]
            x = a * x_candidate + (1 - a) * x
            return x, t, d, cls.ss_p(x, y, grad_fn)
        else:
            d, neighbor_pms, neighbor_norms = cls._get_degree_neighborhood(x, y, grad_fn)
            choices = np.apply_along_axis(Util.safe_sample, 1, neighbor_norms).squeeze()
            sl = x.shape[1]
            bs, per_x_cnt = neighbor_norms.shape
            x = neighbor_pms.reshape(-1, sl)[np.arange(0, bs * per_x_cnt, per_x_cnt) + choices]
            ss_p, _, _ = cls._get_degree_neighborhood(x, y, grad_fn)
            return x, t, d, ss_p


class Permutahedron(_Associatahedron):
    @classmethod
    def get_permutation_generator(cls, sl):
        if cls._permutation_matrix is None:
            e1 = np.eye(sl, dtype=int)
            e2 = np.concatenate((e1[:, 1:], e1[:, 0, None]), 1)
            pm = np.repeat(np.arange(sl)[None, :], sl, 0) - e1 + e2
            pm[0] = np.arange(sl)
            cls._permutation_matrix = pm
        return cls._permutation_matrix


class PermutahedronMHRW(Permutahedron):
    mhrw = True


class SquarePermutahedron(_Associatahedron):

    @classmethod
    def get_permutation_generator(cls, sl):
        if cls._permutation_matrix is None:
            bmpg = BaseMC.get_permutation_generator(sl)
            pg = []
            for i in range(sl):
                if i == 0:
                    pg.append(bmpg[i])
                else:
                    pg.append(bmpg[i][np.arange(sl) != i])
            cls._permutation_matrix = np.concatenate(pg, 0)
        return cls._permutation_matrix


class SquarePermutahedronMHRW(SquarePermutahedron):
    mhrw = True


class GenPermutahedron(_Associatahedron):
    depth_k = 4

    @classmethod
    def get_permutation_generator(cls, sl):
        if cls._permutation_matrix is None:
            pm = np.array(list(itertools.permutations(np.arange(cls.depth_k), cls.depth_k)))
            pm = np.concatenate([pm, np.repeat(np.arange(cls.depth_k, sl)[None, :], pm.shape[0], 0)], 1)
            rev = np.arange(sl - 1, -1, -1)[None, :]
            pm = np.concatenate([np.arange(sl)[None, :], pm, rev])
            cls._permutation_matrix = pm
        return cls._permutation_matrix


class GenPermutahedronMHRW(GenPermutahedron):
    mhrw = True


class RGenPermutahedron(_Associatahedron):
    depth_k = 4

    @classmethod
    def get_permutation_generator(cls, sl):
        if cls._permutation_matrix is None:
            pm = np.array(list(itertools.permutations(np.arange(sl - cls.depth_k, sl), cls.depth_k)))
            pm = np.concatenate([np.repeat(np.arange(sl - cls.depth_k)[None, :], pm.shape[0], 0), pm], 1)
            rev = np.arange(sl - 1, -1, -1)[None, :]
            pm = np.concatenate([np.arange(sl)[None, :], pm, rev])
            cls._permutation_matrix = pm
        return cls._permutation_matrix


class RGenPermutahedronMHRW(RGenPermutahedron):
    mhrw = True


class SampleDict:
    def __init__(self):
        self.samples = dict()

    def update(self, *args):
        for k, wt, y, tid in zip(*args):
            if k not in self.samples:
                self.samples[k] = (wt, y, tid)
            else:
                self.samples[k] = (self.samples[k][0] + wt, y, tid)

    def get(self):
        b_x = np.array(list(self.samples.keys()))
        b_y = Util.alm(lambda t: t[1], self.samples.values())[:, None]
        wt = Util.alm(lambda t: t[0], self.samples.values())
        tid = Util.alm(lambda t: t[2], self.samples.values())
        return b_x, b_y, wt, tid

    def get_hrv_thm_estimator(self, bs):
        b_x, b_y, wt, tid = self.get()
        ohe_tid = np.eye(bs)[tid]
        wt_norm = np.matmul(ohe_tid.transpose(), wt)
        wt /= wt_norm[tid]
        wt /= bs
        return b_x, b_y, wt


class SamplerUtil:
    @classmethod
    def importance_sample_permutation(cls, mb_x, mb_y, n_l, n_s, mb_wt, splits=1):
        """

        Each permutation or example has dimension d
        The large batch size is n_l and the task is to sample n_s samples out of the large batch
        The importance sampling is done using weights mb_wt
        Additionally you have spl splits, and the importance sampling happens per split
        This is so that importance sampling can happen per unique sequence
        Note that the weight includes the correction for the number of splits

        :param mb_x: np.array of input x of dimension (splits*n_l, d)
        :param mb_y: np.array of input y of dimension (splits*n_l, 1)
        :param n_l: int
        :param n_s: int
        :param mb_wt: np.array
        :param splits: int
        :return: b_x, b_y, b_wt
        """
        _rs = lambda nda: nda.reshape((splits, n_l, -1))
        mb_x, mb_y, mb_wt = Util.lm(_rs, (mb_x, mb_y, mb_wt))

        b_x, b_y, b_wt = [], [], []
        for i in range(splits):
            wts = mb_wt[i].squeeze()
            if wts.sum() == 0:
                wts[:] = 1
            wts /= wts.sum()
            sampled_idx = np.random.choice(wts.shape[0], n_s, p=wts)
            b_x.append(mb_x[i][sampled_idx])
            b_y.append(mb_y[i][sampled_idx])
            b_wt.append(1 / wts[sampled_idx] / n_l / n_s / splits)
        b_x, b_y, b_wt = Util.lm(lambda nda: np.concatenate(nda, 0), (b_x, b_y, b_wt))
        return b_x, b_y, b_wt

    @classmethod
    def simple_estimate_norm(cls, b_x, b_y, grad_fn, n_samp=5):
        b_x = np.repeat(b_x, n_samp, 0)
        b_y = np.repeat(b_y, n_samp, 0)
        return grad_fn(b_x, b_y).reshape((-1, n_samp)).mean(1)

    @classmethod
    def restart_sample_permutations(cls, b_x, b_y, alpha, nt, grad_fn, mc=BaseMC):
        # alpha is not sensitive to the degree of the permutahedron and that needs to be clear to the user
        b_x = np.repeat(b_x, nt, 0)  # for easy tour management
        b_y = np.repeat(b_y, nt, 0)  # for easy tour management
        total_tours, sl = b_x.shape

        b_id = np.arange(total_tours)
        b_x = mc.initialize(b_x)
        normalizer = cls.simple_estimate_norm(b_x, b_y, grad_fn)  # n_x always has norms that match b_id
        # The normalizer is only required to sample restarts

        sd = SampleDict()
        active = np.full(b_x.shape[0], 1)
        f_x, f_y, f_id = b_x, b_y, b_id
        with tqdm(desc=f"{active.sum()} Tours remaining") as pbar:
            while active.sum() > 0:
                pbar.update(1)
                pbar.set_description(f"{active.sum()} Tours remaining")

                # steady state probability /degree of current position
                pi_x = mc.ss_p(f_x, f_y, grad_fn) / normalizer[f_id]

                # Append samples with biases
                sd.update(Util.lm(tuple, f_x), 1 / (alpha + pi_x), f_y.squeeze(1).tolist(), f_id)

                # alpha = alpha_0*n
                restart = np.random.uniform(size=pi_x.shape[0]) < (alpha / (alpha + pi_x))
                if restart.any():
                    active -= restart.astype(int)
                    if active.sum() == 0:
                        break

                # filter out restarted tours
                f_x, f_y, f_id = f_x[active == 1], f_y[active == 1], f_id[active == 1]
                active = active[active == 1]
                f_x = mc.step(f_x, f_y, grad_fn)[-1].x
        b_x, b_y, wt, _ = sd.get()
        wt *= alpha / total_tours
        return b_x, b_y, wt

    @classmethod
    def mcmc_sample_permutations(cls, b_x, b_y, w_l, grad_fn, mc=BaseMC):
        b_x = mc.initialize(b_x)
        bs, sl = b_x.shape
        b_id = np.arange(bs)

        sd = SampleDict()
        bias = 1 / mc.ss_p(b_x, b_y, grad_fn)
        sd.update(Util.lm(tuple, b_x), bias, b_y.squeeze(1).tolist(), b_id)
        for _ in tqdm(range(w_l - 1), desc="rw_len"):
            next_state = mc.step(b_x, b_y, grad_fn)[-1]
            b_x, b_x_norms = next_state.x, next_state.ss_p
            # Append samples with biases
            bias = 1 / b_x_norms
            bias[bias != bias] = np.finfo(float).eps
            sd.update(Util.lm(tuple, b_x), bias, b_y.squeeze(1).tolist(), b_id)
        return sd.get_hrv_thm_estimator(bs)

    @classmethod
    def frontier_sample_permutations_vertices(cls, b_x, b_y, f_s, w_l, grad_fn, mc=BaseMC):
        bs, sl = b_x.shape
        b_x = np.repeat(b_x, f_s, 0)
        b_y = np.repeat(b_y, f_s, 0)
        b_id = np.repeat(np.arange(bs), f_s, 0)

        b_x = mc.initialize(b_x)
        b_n = mc.ss_p(b_x, b_y, grad_fn)  # n_x always has normalized ssp that match b_x
        normalizer = b_n.reshape((-1, f_s)).mean(1)
        b_n /= normalizer[b_id]

        sd = SampleDict()
        active = np.full_like(b_id, 1)  # during the first iter, everybody samples a neighbor
        budget = np.zeros(b_id.shape[0])

        for _ in tqdm(range(w_l), desc="rw_len"):
            f_x, f_y, f_id = b_x[active == 1], b_y[active == 1], b_id[active == 1]

            next_states = mc.step(f_x, f_y, grad_fn)
            for state in next_states:  # add budgets for batch sampling
                d = state.d / normalizer[f_id]  # normalized degree
                budget[active == 1] += np.random.exponential(1 / d)

            # Update norms and states that changed
            b_x[active == 1] = next_states[-1].x
            b_n[active == 1] = next_states[-1].ss_p / normalizer[b_id[active == 1]]

            # mark actives
            active[:] = 0
            budget_per_frontier = budget.reshape((-1, f_s))
            # Set the minimum per frontier as active (these will always be within budget)
            active[budget_per_frontier.argmin(1) + np.arange(0, budget.shape[0], f_s)] = 1

            bias = 1 / b_n[active == 1]
            bias[bias != bias] = np.finfo(float).eps
            sd.update(Util.lm(tuple, b_x[active == 1]), bias, b_y[active == 1].squeeze(1).tolist(), b_id[active == 1])
        return sd.get_hrv_thm_estimator(bs)

    @classmethod
    def hybrid_frontier_sample_permutations_vertices(cls, b_x, b_y, f_s, w_l, grad_fn, mc=BaseMC):
        bs, sl = b_x.shape
        b_x = np.repeat(b_x, f_s, 0)
        b_y = np.repeat(b_y, f_s, 0)
        b_id = np.repeat(np.arange(bs), f_s, 0)

        b_x = mc.initialize(b_x)
        b_n = mc.ss_p(b_x, b_y, grad_fn)  # n_x always has normalized ssp that match b_x
        normalizer = b_n.reshape((-1, f_s)).mean(1)
        b_n /= normalizer[b_id]
        b_n_record = [b_n.copy()]

        sd = SampleDict()
        active = np.full_like(b_id, 1)  # during the first iter, everybody samples a neighbor
        budget = np.zeros(b_id.shape[0])

        for _ in tqdm(range(w_l), desc="rw_len"):
            f_x, f_y, f_id = b_x[active == 1], b_y[active == 1], b_id[active == 1]

            next_states = mc.step(f_x, f_y, grad_fn)
            for state in next_states:  # add budgets for batch sampling
                d = state.d / normalizer[f_id]  # normalized degree
                budget[active == 1] += np.random.exponential(1 / d)

            # Update norms and states that changed
            b_x[active == 1] = next_states[-1].x
            b_n[active == 1] = next_states[-1].ss_p / normalizer[b_id[active == 1]]

            # mark actives
            active[:] = 0
            budget_per_frontier = budget.reshape((-1, f_s))
            # Set the minimum per frontier as active (these will always be within budget)
            active[budget_per_frontier.argmin(1) + np.arange(0, budget.shape[0], f_s)] = 1

            bias = 1 / b_n[active == 1]
            bias[bias != bias] = np.finfo(float).eps

            b_n_record.append(b_n_record[-1].copy())
            b_n_record[-1][active == 1] = b_n[active == 1]
            sd.update(Util.lm(tuple, b_x[active == 1]), bias, b_y[active == 1].squeeze(1).tolist(), b_id[active == 1])
        b_n_record = np.vstack(b_n_record[:-1]).reshape(w_l, bs, f_s)
        wt_norm = (1 / b_n_record.mean(axis=2)).sum(axis=0)
        b_x, b_y, wt, tid = sd.get()
        wt /= wt_norm[tid]
        wt /= bs
        return b_x, b_y, wt

