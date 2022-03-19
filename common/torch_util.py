import numpy as np
import torch

from common import NORM_TYPE


class TorchUtil:
    cuda = torch.cuda.is_available()
    device = 0

    @classmethod
    def set_seed(cls, seed):
        if seed > 0:
            torch.manual_seed(seed=seed)
            if cls.cuda:
                torch.cuda.manual_seed(seed=seed)

    @classmethod
    def Tensor(cls):
        tensor = torch.Tensor()
        if cls.cuda:
            tensor = tensor.cuda()
        return tensor

    @classmethod
    def multicat(cls, list_1, list_2):
        return [torch.cat([t1, t2]) for t1, t2 in zip(list_1, list_2)]

    @classmethod
    def cat(cls, args, dim):
        return torch.cat(args, dim=dim)

    @classmethod
    def from_numpy(cls, param, cuda=True):
        """

        :rtype: torch.Tensor
        """
        param = torch.from_numpy(param)
        if cuda and cls.cuda:
            param = param.cuda()
        return param

    @classmethod
    def move(cls, *params):
        rt = tuple(map(lambda p: p if p is None or not TorchUtil.cuda else p.cuda(), params))
        return rt[0] if len(params) == 1 else rt

    @classmethod
    def ensure2d(cls, tens):
        if len(tens) > 0 and tens.dim() == 1:
            tens = tens.unsqueeze(0)
        return tens

    @classmethod
    def get_model_norm(cls, model, grad=False):
        return float(np.linalg.norm(np.array(list(map(
            lambda x: torch.norm(x.grad.detach() if grad else x.detach(), p=NORM_TYPE).cpu().numpy(), model.parameters()))),
            ord=NORM_TYPE))

    @classmethod
    def from_sparse(cls, sdf):
        coo = sdf.to_coo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    @classmethod
    def lt(cls, t) -> torch.LongTensor:
        return torch.LongTensor(t)

    @classmethod
    def ft(cls, t) -> torch.FloatTensor:
        return torch.FloatTensor(t)
