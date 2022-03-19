# A utility for inspecting neural network models (i.e. querying their properties)

import torch
import torch.nn as nn

class NetworkInspect:
    @staticmethod
    def summarize_tensor(tt):
        """ Print summary statistics for a tensor """
        assert torch.is_tensor(tt), "Must provide a PyTorch tensor object"
        sumstats = dict()
        sumstats['shape'] = tt.shape
        sumstats['type'] = tt.type()
        sumstats['min'] = round(tt.min().item(), 4)
        sumstats['mean'] = round(tt.mean().item(), 4)
        sumstats['max'] = round(tt.max().item(), 4)

        for kk, vv in sumstats.items():
            print(kk, "\t", vv)

    @staticmethod
    def summarize_model_weights(model):
        """Loops over all the layers and prints summaries of the weight matrices"""
        assert isinstance(model, nn.Module)

        print("Summarizing model weight matrices")
        for kk, vv in model.state_dict().items():
            print(f"-----\nLayer: {kk}")
            NetworkInspect.summarize_tensor(vv)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Test tensor
    x = torch.randn(10000, 2).to(device)
    NetworkInspect.summarize_tensor(x)

    # Test model summary
    mod = nn.Sequential(nn.Linear(5, 16),
                  nn.Tanh(),
                  nn.Conv2d(1, 3, 5),
                  nn.Tanh())

    NetworkInspect.summarize_model_weights(mod)