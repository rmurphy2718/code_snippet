"""
A class that calls set/graph models and passes them to the appropriate regularization
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from common.helper import Util
from common.torch_util import TorchUtil as tu
from models.GIN_utils import normalize_to_unit_norm
from my_common.my_helper import is_tensor
from regularizers import JpRegularizer
from scalings import Scalings
from util.constants import InputType, Regularization, TaskType


class ModelGS(ABC, nn.Module):
    """
    Parent class that implements various invariant-training strategies of any permutation-sensitive model of a set/graph
    We forward, compute "standard" loss (e.g. MSE) and other penalty terms.

    Permutation-sensitive models should inherit and implement "embed" and "predict" functions,
    which are necessary (at least) for penalize-latent and penalize-prediction functionalities.
    """
    def __init__(self):
        # Initialize nn.Module
        # Don't initialize other aspects of ModelGS!
        # Doing so requires us to pass ModelGS things
        super(ModelGS, self).__init__()

    def init_model_gs(self, permute_for_pi_sgd, regularizer, scaling, task_type):
        assert isinstance(regularizer, JpRegularizer)
        assert isinstance(permute_for_pi_sgd, bool)
        assert isinstance(scaling, Scalings)
        self.regularizer = regularizer
        self.input_type = regularizer.input_type
        self.permute_data = permute_for_pi_sgd
        self.scaling = scaling

        if task_type == TaskType.binary_classification:
            self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        elif task_type == TaskType.multi_classification:
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
        elif task_type == TaskType.regression:
            self.loss_func = nn.MSELoss(reduction='none')

    # ===================================================
    # Functions for forwarding, which is always done,
    # regardless of invariance strategies.
    # ===================================================
    @abstractmethod
    def embed(self, *args):
        """
        Layers that return a latent representation of graph or set

        Inputs:
            set or adjmats, vertex_features


        Return:
            2-dimensional representation (batch size, latent_representation)
        """
        pass

    @abstractmethod
    def predict(self, latent):
        """
        Layers that map a latent representation to a `prediction` (pre-softmax if applicable)
        Inputs:
            Exactly one input, a two-dimensional tensor (batch, embedding_dim)
        """
        pass

    def forward(self, *input_tuple):
        latent = self.embed(*input_tuple)

        if latent.ndim not in (2, 3):  # (batch, embeddings) or (batch, set length, embeddings)
            msg = "Latent embedding (output of embed function) should have exactly 2 dims: (batch_size, latent_dim)\n"
            msg += "Predict function should accept two- or three-dimensional arguments"
            raise RuntimeError(msg)

        if self.scaling.quantize and self.training:
            noise = torch.randn_like(latent)
            latent = torch.add(latent, noise)

        prediction = self.predict(latent)
        if prediction.ndim != 2:
            raise RuntimeError("Prediction should return a 2D tensor (batch, preds)")

        return prediction, latent

    # ===================================================
    # Permuting for data augmentation
    # regardless of invariance strategies.
    # ===================================================
    def permute_one(self, mat):
        assert mat.ndim == 2
        assert is_tensor(mat)

        # Permute rows and columns of adjacency matrix
        perm = torch.randperm(mat.size(0))
        mat = mat[perm, :]
        if self.input_type == InputType.graph:
            mat = mat[:, perm]

        return mat


    def permute(self, data):
        """ Permute a list or tensor of matrices """
        mats = data[0]
        permuted_list = Util.lm(self.permute_one, mats)
        permuted_tensor = torch.stack(permuted_list)
        return permuted_tensor, data[1]


    def process_data_input(self, data):
        """
        Move to GPU, Process set versus graph input.
        Abstracted b/c we call in mc_variance.
        """
        if self.input_type == InputType.graph:  # Data is a tuple: (adjmat, vertex_features)
            assert isinstance(data, tuple) and len(data) == 2  # holds adjmat, vertex_features
        else:  # For sets, make a tuple of length one
            data = (data, )

        assert all([is_tensor(dd) for dd in data])
        assert all([dd.ndim == 3 for dd in data])

        data = tu.move(data)  # Can move a tuple as well
        return data

    def main(self, data, targets, no_reduce=False, sum_reduce=False, wt=None):
        """
        Perform these steps, only standard forwarding and loss computation are required
        1. Optional permute for pi-SGD data augmentation
        2. Forwards model (child class must implement it)
        3. Optional addition of noise to put all representations on a less-ambiguous scale (mostly for Birkhoff reg)
        4. Compute "standard" loss
        5. Optional computation of Birkhoff-style regularization (JpRegularizer)

        Input:
            For sets: Tensor of shape (batch, set_length, set_elements)
            For graphs: Tuple with (adjmats, vertex_features)

        returns:
            prediction, loss, penalty, latent_norm

        """
        assert is_tensor(targets)
        targets = tu.move(targets)
        data = self.process_data_input(data)

        # ------- Permute
        if self.permute_data:
            data = self.permute(data)

        # ------- Forward (and possibly add random noise)

        if self.regularizer.tangent_prop:
            data[0].requires_grad = True

        prediction, latent = self.forward(*data)

        # ------- Standard loss
        loss = self.get_loss(prediction, targets)

        # ------- Penalties
        if self.regularizer.input_type == InputType.graph:
            overall_loss, penalty, latent_norm = self.penalty(prediction, latent, loss,
                                                              adjmats=data[0],
                                                              vertex_features=data[1])
        else:
            overall_loss, penalty, latent_norm = self.penalty(prediction, latent, loss,
                                                              sets=data[0])

        overall_loss, penalty, latent_norm = self.loss_reduction(overall_loss, penalty, latent_norm,
                                                                 no_reduce, sum_reduce, wt)

        return prediction, overall_loss, penalty, latent_norm

    def get_loss(self, pred, y):
        """
        Returns loss without reduction.
        Checks for appropriate dimensions depending on the loss function being used
        If dimensions are mismatched, return easier-to-read error messages than PyTorch does by default
        """
        if isinstance(self.loss_func, nn.CrossEntropyLoss):
            if pred.ndim == 2 and y.ndim == 1:
                return self.loss_func(pred, y)
            else:
                raise RuntimeError(f"""Error in `get_loss`: expected 2-dim prediction and 1-dim target:
                Got {pred.shape} {y.shape}
                Either user intended for multiple-target classification, which is not implemented,
                or unexpected dimensions are encountered at runtime""")
        elif isinstance(self.loss_func, nn.BCEWithLogitsLoss) or \
                isinstance(self.loss_func, nn.MSELoss):
            if pred.ndim == 2 and y.ndim == 1:
                pred = pred.squeeze(dim=1)
            if pred.shape == y.shape:
                return self.loss_func(pred, y)
            else:
                raise RuntimeError(f"""Unexpected shape for {self.loss_func}: 
                {pred.shape}, {y.shape}
                While some nn loss functions try to broadcast, we are more stringent and enforce same shape""")
        else:
            raise NotImplementedError(f"Haven't implemented loss {self.loss_func}")

    def penalty(self, pred, latent, loss, **input_kwargs):
        """
        Birkhoff regularization and latent norm embedding
        kwargs can pass adjmats, sets, and possibly vertex features.
        """
        overall_loss = loss
        permutation_penalty = torch.zeros_like(loss)  # Even when not regularizing, we pass variable to reductions later
        latent_penalty = torch.zeros_like(loss)

        if self.training:
            assert pred.dim() == 2 and latent.dim() == 2, \
                "Unexpected number of dimensions in representation passed to regularizer."
            # ------------------------------------
            # Penalize permutation sensitivity
            # ------------------------------------
            if self.regularizer.regularization != Regularization.none:
                if self.regularizer.penalize_prediction:
                    representation = pred
                else:
                    representation = latent
                permutation_penalty = self.regularizer(representation,
                                                       self,
                                                       disable_gradient=False,
                                                       **input_kwargs
                                                       )

                if permutation_penalty.shape != loss.shape:
                    raise RuntimeError("Differing shapes in penalty and loss")

                overall_loss = overall_loss + permutation_penalty

            # ------------------------------------
            # Penalize large values of the latent
            # ------------------------------------
            if self.scaling.penalize_embedding:
                # strength * || latent ||
                #
                # Note: Assertion above guarantees that the latent is two dimensional
                latent_penalty = torch.mul(torch.norm(latent, dim=1),
                                           self.scaling.embedding_penalty)

                # Add to overall loss
                overall_loss = overall_loss + latent_penalty

        return overall_loss, permutation_penalty, latent_penalty

    def loss_reduction(self, overall_loss, penalty, latent_norm, no_reduce=False, sum_reduce=False, wt=None):
        """
        Appropriate reduction of loss and penalty (or lack thereof):
          > summation, simple mean, weighted average, or no reduction
        """
        if no_reduce:
            return overall_loss, penalty, latent_norm
        if sum_reduce:
            return overall_loss.sum(), penalty.sum(), latent_norm.sum()
        if wt is None:
            return overall_loss.mean(), penalty.mean(), latent_norm.mean()
        else:
            return (overall_loss * wt).sum(), (penalty * wt).sum(), (latent_norm * wt).sum()

    def mc_variance(self, data, num_pairs = 15):
        """
        Quanitify the variability of f-arrow.
        Conceptually, for a given graph,
        Repeat:
            (1) Sample pairs of permutations pi and phi,
            (2) compute ||f(x_pi) - f(x_phi)||^2
        Then:
            Take the mean.
        This estimates 2*tr(Sigma), where Sigma is the population variance of the output of f-arrow, for a given input
        We don't care about the factor of two.  It's just a constant, and we care about relative variability
        We return the max and means of this estimate for the entire batch
        """
        self.process_data_input(data)
        assert num_pairs > 0
        assert data[0].ndim == 3
        batch_size = data[0].shape[0]
        divisor = 1.0 / num_pairs

        pred_diffs = tu.move(torch.zeros(batch_size))
        latent_diffs = tu.move(torch.zeros(batch_size))

        for ii in range(num_pairs):
            # Sample a pair of permutations, and compute f(x_pi) - f(x_phi)
            data = self.permute(data)
            pred_1, latent_1 = self.forward(*data)
            data = self.permute(data)
            pred_2, latent_2 = self.forward(*data)

            # Scale to have unit-norm rows to facilitate interpretation across models
            if pred_1.shape[1] > 1:
                pred_1 = normalize_to_unit_norm(pred_1, dim=1)
                pred_2 = normalize_to_unit_norm(pred_2, dim=1)

            if latent_1.shape[1] > 1:
                latent_1 = normalize_to_unit_norm(latent_1, dim=1)
                latent_2 = normalize_to_unit_norm(latent_2, dim=1)

            # Compute ||f(x_pi) - f(x_phi)||^2 for each vector in the batch,
            # which computes an estimate of 2*tr(Sigma) for each sequence/graph in batch
            pred_diffs += divisor * torch.sum(torch.pow(pred_1 - pred_2, 2), dim=1)
            latent_diffs += divisor * torch.sum(torch.pow(latent_1 - latent_2, 2), dim=1)

        # raise NotImplementedError("Check again")  # Check MC variance again.
        return pred_diffs.mean().item(), pred_diffs.max().item(), \
               latent_diffs.mean().item(), latent_diffs.max().item()
