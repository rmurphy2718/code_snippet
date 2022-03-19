import torch
import numpy as np
from common.helper import Util
from my_common.my_helper import concat_list
from util.constants import *


class JpUtil:
    @staticmethod
    def _reshape_for_inference(predictions, n_permutations):
        """
        Input: list describing a vector of shape (num_seq * num_perm, 1)
        Output: matrix of shape (num_seq, num_perm)
        """
        assert isinstance(predictions, list)
        assert predictions[0].dim() == 1 or \
               predictions[0].dim() == 2 and predictions[0].shape[1] == 1

        # from list to array-like
        output = concat_list(predictions, 0)
        return output.reshape((-1, n_permutations))

    @staticmethod
    def _reshape_for_reprs(representations, n_permutations):
        """
        Input: list of matrices of shapes (num_seq * num_perms, hidden_dim)
        Output: 3-rank tensor of shape (num_seq, num_perm, hidden_dim)
        """
        # (keep this method separate from above in case of future changes

        assert isinstance(representations, list)
        assert representations[0].dim() == 2

        output = concat_list(representations, 0)
        hdim = output.shape[1]

        return output.reshape(-1, n_permutations, hdim)

    @classmethod
    def diff_permutation(cls, input_sequence):
        """
        Generate a permutation of the input sequence uniformly except for the original one
        :param input_sequence: expected input sequence to be a 1-dimensional ndarray, e.g., [1,2,3]
        :return: permuted sequence, e.g., [2,1,3],[3,1,2],... (except [1,2,3])
        """
        if not (isinstance(input_sequence, np.ndarray)):
            raise NotImplementedError("Expected input_sequence to be numpy arrays")
        assert len(input_sequence.shape)==1 and input_sequence.shape[0] > 0

        permuted_sequence = np.random.permutation(input_sequence)
        while (permuted_sequence == input_sequence).all():
            permuted_sequence = np.random.permutation(input_sequence)

        return permuted_sequence

    @classmethod
    def forward_random_permutations(cls, images, input_sequence, model, n_permutations, inference, reprs, disable_grads, allow_identity=True):
        """
        Permute the input sequences several times and forward them through the model
        Uses cases: (Inference) Obtaining predictions (yhat) for multiple permutations, for JP inference
                    (Representations) Obtaining latent representations at multiple permutations to better evaluate
                     the permutation sensitivity

        :params inference and reprs: bool, permutations for (Inference) and/or (Representations) case
        :param: disable_grads will turn off gradients if True, otherwise, it uses
                              gradients based on whether they are already enabled or not
        :return: Output for either inference and/or representations case, appropriately reshaped for downstream
        """

        assert isinstance(inference, bool) and isinstance(reprs, bool)
        assert inference or reprs
        assert isinstance(disable_grads, bool)

        if not (isinstance(input_sequence, np.ndarray) and isinstance(images, np.ndarray)):
            raise NotImplementedError("Expected input_sequence and images to be numpy arrays")

        # Toggle gradient computation
        previous_grad_state = torch.is_grad_enabled()
        if disable_grads:
            torch.set_grad_enabled(False)

        input_sequence = input_sequence.repeat(n_permutations, 0)

        if allow_identity:
            input_sequence = np.apply_along_axis(np.random.permutation, 1, input_sequence)
        else:
            input_sequence = np.apply_along_axis(cls.diff_permutation, 1, input_sequence)

        predictions = list()
        representations = list()
        for vbatch in Util.make_batches(input_sequence.shape[0], Constants.get_batch_limit(nograd= not torch.is_grad_enabled()),
                                        use_tqdm=False):
            valid_batch_seq = input_sequence[vbatch]
            model_output, representation = model(images, valid_batch_seq)[:2]

            if inference:
                predictions.append(model_output)
            if reprs:
                representations.append(representation)

        # Reshape the outputs in the appropriate way for the context
        if inference and reprs:
            result = cls._reshape_for_inference(predictions, n_permutations), \
                     cls._reshape_for_reprs(representations, n_permutations)
        elif inference and not reprs:
            result = cls._reshape_for_inference(predictions, n_permutations)
        elif not inference and reprs:
            result = cls._reshape_for_reprs(representations, n_permutations)

        torch.set_grad_enabled(previous_grad_state)
        return result

    @staticmethod
    def get_jvp_transpose(y, x, vec):
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
        ujpT = ujp.transpose(1, 0) if ujp.dim() > 1 else ujp
        jvpT = torch.autograd.grad(ujpT, u,
                                   grad_outputs=torch.transpose(vec, 1, 0) if vec.dim() > 1 else vec,
                                   create_graph=True)[0]

        return jvpT

if __name__ == "__main__":
    from janossy_models import JanossyMathModel

    device = torch.device('cuda:0')

    np.random.seed(1)
    n_seq = 3000
    seq_size = 3
    img_size = 16

    n_img = n_seq * seq_size

    images = np.random.randint(-3, 8, (n_img, img_size))
    images = images.reshape(n_img, img_size)
    input_sequence = np.arange(n_img).reshape(n_seq, seq_size)

    from regularizers import JpRegularizer
    model = JanossyMathModel(input_dim=images.shape[1],
                             model_type=Model.lstm,
                             num_layers=1,
                             num_neurons=8,
                             aggregation=Aggregation.attention,
                             loss_function=LossFunction.mse,
                             regularizer=JpRegularizer(Regularization.none)).to(device)

    #preds, reprs = model(images, input_sequence)

    result = JpUtil.forward_random_permutations(images, input_sequence, model, 2, False, True, True)
    JpUtil.forward_random_permutations(images, input_sequence, model, 2, False, True, True)
