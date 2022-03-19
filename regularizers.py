"""Birkhoff regularization towards permutation invariance"""

from collections import OrderedDict

import numpy as np
import torch

from common.torch_util import TorchUtil as tu
from util.constants import Constants, InputType, Regularization


class FiniteDifferences:
    @staticmethod
    def make_d_step_center(eps, seqlen):
        """
        Create a doubly stochastic matrix by a
        convex combination between identity matrix
        and the matrix 1/n * ones() (this matrix computes a mean)

        :param eps: The desired norm between output and identity matrix
        :param seqlen: The dimension of the matrix
        """
        divisor = float(np.sqrt(seqlen - 1))
        assert divisor > 0 and eps < divisor  
        c = 1 - eps / divisor
        centroid = torch.full((seqlen, seqlen), 1. / seqlen)
        return c * torch.eye(seqlen) + (1 - c) * centroid

    @staticmethod
    def make_d_step_edge(eps, seqlen, permute_range=None):
        """
        Create a doubly stochastic matrix by a
        convex combination between identity matrix
        and the matrix with two consecutive rows swapped (a neighbor to the identity on the Birkhoff polytope)

        :param eps: The desired norm between output and identity matrix
        :param seqlen: The dimension of the matrix
        :param permute_range: The range of the rows to be swapped
        """
        assert 0. < eps < 2., "Epsilon must be between 0 and 2"
        c = 1. - eps / 2.
        if permute_range is None:
            permute_range = seqlen
        else:
            assert 1 < permute_range <= seqlen

        first_idx = torch.randint(permute_range - 1, (1,)).item()
        second_idx = (first_idx + 1)

        D = torch.eye(seqlen)
        D[first_idx, second_idx] = 1. - c
        D[second_idx, first_idx] = 1. - c
        D[first_idx, first_idx] = c
        D[second_idx, second_idx] = c

        return D


class JpRegularizer:
    def __init__(self, regularization, input_type, total_permutations=None, strength=None, eps=None, penalize_prediction=False,
                 tangent_prop=False, tp_config=None, fixed_edge=False):
        """
        :param total_permutations: Number of permutations to use for
        computing the penalty, >= number used for training

        :param: regularization strength (if any)
        :param: eps -- size of finite difference for the step-center method
        :param: penalize_prediction -- If true, we are regularizing permutation sensitivity of the output not the latent
                                       representation
        :param: tangent prop -- If true, we are using Tangent Prop style penalty that first estimate tangent vectors by
                                taking diff step on x and then calculate the Jacobian - tangent vector product
         :param: tp_config -- configurations of TangentProp, including
                             tp_power and tp_normed -- The default values correspond to the Tangent Prop paper, squared L2 norm.
                             tp_num_tangent_vectors -- Number of tangent vectors used for TangentProp (default: 1)
        :param: fixed_edge -- Fix the edge of DS matrices for diff_step_edge regularization (to the one between 1st and 2nd perms)
        """
        assert regularization in Regularization
        assert isinstance(input_type, InputType)
        self.input_type = input_type

        for bool_var in [penalize_prediction, tangent_prop, fixed_edge]:
            assert isinstance(bool_var, bool), \
                f"JpRegularizer expected a bool but got type {type(bool_var)} for some input"

        if total_permutations is None:
            assert regularization in (Regularization.none, Regularization.diff_step_center,
                                      Regularization.diff_step_edge, Regularization.diff_step_basis)
        else:
            assert isinstance(total_permutations, int)
            if regularization == Regularization.pw_diff:
                assert total_permutations > 1, "pw_diff regularizer needs at least 2 permutations"
            else:
                assert total_permutations > 0

        if strength is None:
            assert regularization == Regularization.none
        else:
            assert isinstance(strength, float) and strength >= 0.0

        if regularization in (Regularization.diff_step_center, Regularization.diff_step_edge, Regularization.diff_step_basis):
            assert eps is not None and isinstance(eps, float) and eps > 0.0
        else:
            assert tangent_prop is False, "tangent_prop can only be true for a step-type regularization"

        if fixed_edge:
            assert regularization in (Regularization.diff_step_edge, Regularization.diff_step_basis), \
                "Fixed edge was specified, but edge-step regularization is not in effect."

        if tangent_prop:
            assert isinstance(tp_config, OrderedDict)
            assert all(config in tp_config for config in ['tp_power', 'tp_normed', 'tp_num_tangent_vectors'])
            assert isinstance(tp_config['tp_power'], int) and tp_config['tp_power'] > 0
            assert isinstance(tp_config['tp_normed'], bool)
            assert isinstance(tp_config['tp_num_tangent_vectors'], int) and tp_config['tp_num_tangent_vectors'] > 0
            if tp_config['tp_num_tangent_vectors'] > 1:
                assert not fixed_edge, "Edge must not be fixed for More-tangent-vector TangentProp."
                assert regularization in (Regularization.diff_step_edge, Regularization.diff_step_basis), \
                    "More-tangent-vector TangentProp was specified, but edge-step regularization is not in effect."
            self.tp_power = tp_config['tp_power']
            self.tp_normed = tp_config['tp_normed']
            self.tp_num_tangent_vectors = tp_config['tp_num_tangent_vectors']

        self.regularization = regularization
        self.total_permutations = total_permutations
        self.strength = strength
        self.eps = eps
        self.penalize_prediction = penalize_prediction
        self.tangent_prop = tangent_prop

        self.D = None  # D is fixed in center step; retained to save time
        # permute range of edge-step D
        self.edge_permute_range = 2 if fixed_edge else None # specified for fixed edge or BA experiments or variable-size graphs

    def __call__(self, representations, model, disable_gradient, adjmats=None, sets=None, vertex_features=None):
        """
        :param representations: tensor of shape (num_sequences, num_permutations, dimension)
        """
        assert representations.dim() == 2
        assert isinstance(disable_gradient, bool)

        if self.input_type == InputType.graph:
            assert adjmats is not None
            seq_len = adjmats[0].shape[0]  # Currently assume same sizes (simplicity)
            if not self.tangent_prop:  # both adjmats and features are required to compute finite diff
                assert vertex_features is not None
        else:
            assert sets is not None
            seq_len = sets[0].shape[0]  # Currently assume same sizes (simplicity)
            seq_len = sets[0].shape[0]  # Currently assume same sizes (simplicity)

        previous_grad_state = torch.is_grad_enabled()
        if disable_gradient:
            torch.set_grad_enabled(False)

        if self.regularization in (Regularization.diff_step_center,
                                     Regularization.diff_step_edge,
                                     Regularization.diff_step_basis):

            # If computing the "basis" penalty, compute both the center-step and edge-step penalties
            # o.w. just compute one
            if self.regularization in (Regularization.diff_step_center, Regularization.diff_step_basis):
                # Finite diff regularization, move input to the very center of permuto
                # (we construct the relevant matrix here which is needed for the center step)
                if self.D is None:
                    self.D = tu.move(FiniteDifferences.make_d_step_center(self.eps, seq_len))
                if self.tangent_prop:
                    center_penalty = self.diff_step_tangent_prop(representations, self.D,
                                                                 adjmats, sets, vertex_features)
                else:
                    center_penalty = self.diff_step(representations, model, self.D, adjmats, sets, vertex_features)

            if self.regularization in (Regularization.diff_step_edge, Regularization.diff_step_basis):
                # Finite diff regularization, move input along an edge
                # Note, if there are duplicates in the input sequence, the edge penalty is zero,
                # but it will be unlikely and it's not worth checking here.
                if self.tangent_prop:
                    # To prevent numerical overflow:
                    # Don't sum then divide, divide as we go
                    divisor = 1. / self.tp_num_tangent_vectors
                    edge_penalty = tu.move(torch.zeros(representations.shape[0]))
                    for j in range(self.tp_num_tangent_vectors):
                        D = tu.move(FiniteDifferences.make_d_step_edge(self.eps, seq_len, self.edge_permute_range))
                        edge_penalty += divisor * self.diff_step_tangent_prop(representations, D,
                                                                              adjmats, sets, vertex_features)
                else:
                    D = tu.move(FiniteDifferences.make_d_step_edge(self.eps, seq_len, self.edge_permute_range))
                    edge_penalty = self.diff_step(representations, model, D, adjmats, sets, vertex_features)

            if self.regularization == Regularization.diff_step_basis:
                # Assign equal weights to center_penalty and edge_penalty
                result = self.strength * 0.5 * (center_penalty + edge_penalty)
            elif self.regularization == Regularization.diff_step_center:
                result = self.strength * center_penalty
            elif self.regularization == Regularization.diff_step_edge:
                result = self.strength * edge_penalty

        # Implementing "no regularization" for completeness and flexibility
        # Not recommended to use this way as gradient will propagate through this layer,
        # better to use logic to avoid computing the penalty in the caller
        elif self.regularization == Regularization.none:
            result = torch.zeros_like(representations.shape)
        else:
            raise NotImplementedError(f"Regularization not implemented in __call__: {self.regularization}")

        torch.set_grad_enabled(previous_grad_state)
        return result

    def __unify_shapes(self, in_tensor):
        """
        Unify the shapes to be two-dimensional
        Shape depends on whether we're regularizing the prediction or the latent
        Sometimes, representations will have an additional dimension
        because other regularizers require it...the unsqueeze() is in loss.
        """
        if in_tensor.dim() == 2:
            return in_tensor
        elif in_tensor.dim() == 3 and in_tensor.shape[1] == 1:
            return in_tensor.squeeze(1)
        elif in_tensor.dim() == 1:
            return in_tensor.unsqueeze(1)
        else:
            raise RuntimeError("Unexpected dimension based on implementation")

    def diff_step(self, representations, model, D, adjmats, sets, vertex_features):
        """
        Finite difference gradient estimator
        Uses D that move input along permutohedron
        """

        # Send sequence along direction of its permutohedron
        if self.input_type == InputType.graph:
            adjmats_inside = D @ adjmats @ D.transpose(0, 1)
            predictions_inside, latent_inside = model(adjmats_inside, vertex_features)
        else:
            sets_inside = D @ sets
            predictions_inside, latent_inside = model(sets_inside)

        if self.penalize_prediction:
            representation_inside = predictions_inside
        else:
            representation_inside = latent_inside

        representations = self.__unify_shapes(representations)
        representation_inside = self.__unify_shapes(representation_inside)

        if representations.shape != representation_inside.shape:
            print(f"representations.shape is {representations.shape}\nrepresentation_inside.shape is {representation_inside.shape}")
            raise RuntimeError(f"shape mismatch")

        penalty = (1. / self.eps) * torch.norm(representations - representation_inside,
                                               dim=1,
                                               p=Constants.NORM_TYPE)
        return penalty

    # Focus on this method
    def diff_step_tangent_prop(self, representations, D, adjmats, sets, vertex_features):
        """
        Finite difference gradient estimator (tangent prop)
        Uses D that move input along permutohedron
        """
        input_obj = adjmats if self.input_type == InputType.graph else sets

        assert input_obj.requires_grad, "Tangent prop regularizer requires grad for (one of) the input(s), see code."
        assert input_obj.dim() == 3
        assert vertex_features is None or vertex_features.requires_grad is False, \
            "Failed sanity check: adjmats should not require grad in permute-onehot tangent prop."

        # Assert grads are being tracked, write general message in case we port this to sequences (one input) later

        representations = self.__unify_shapes(representations)

        # Send sequence along direction of its permutohedron
        if self.input_type == InputType.graph:
            input_inside = D @ adjmats @ D.transpose(0, 1)
        else:
            input_inside = D @ sets

        # Compute Jacobian-vector product with tangent vector
        tangent_vec = (input_obj - input_inside) / self.eps
        jvp = self.get_jvp(representations, input_obj, tangent_vec)  # shape of jvp: (batch_size, reps_size)

        # compute penalty
        if self.tp_normed:
            penalty = torch.norm(jvp, p=self.tp_power, dim=1)
        elif self.tp_power == 1:
            penalty = torch.sum(torch.abs(jvp), dim=1)
        else:
            if self.tp_power % 2 != 0:
                jvp = torch.abs(jvp)
            penalty = torch.sum(torch.pow(jvp, self.tp_power), dim=1)

        return penalty

    @staticmethod
    def get_jvp(y, x, vec):
        """
        Construct a differentiable Jacobian-vector product for a function
        Adapted from: https://gist.github.com/ybj14/7738b119768af2fe765a2d63688f5496
        Trick from: https://j-towns.github.io/2017/06/12/A-new-trick.html

        :param y: output of a function
        :param x: input of a function
        :param vec: vector
        """
        u = torch.zeros_like(y, requires_grad=True)  # u is an auxiliary variable and could be arbitary
        ujp = torch.autograd.grad(y, x, grad_outputs=u, create_graph=True)[0]
        jvp = torch.autograd.grad(ujp, u,
                                  grad_outputs=vec,
                                  create_graph=True)[0]
        return jvp

    def perm_grad_penalty(self):
        """Gradient of f-arrow at a particular function"""
        raise NotImplementedError

