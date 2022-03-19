import torch
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    """Define a multilayer perceptron
       assume that all intermediate hidden layers have the same dimension (number of neurons)
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers, act, other_mlp_parameters={}):
        """ :param: other_mlp_parameters: dictionary with keys of dropout and/or batchnorm.  values are dropout prob"""
        super(MLP, self).__init__()
        assert num_hidden_layers > 0, "MLP should have at least one hidden layer"
        assert isinstance(other_mlp_parameters, dict), 'other_mlp_parameters should be dict or none.'

        # Check that the other mlp parameters are valid
        for key_ in other_mlp_parameters.keys():
            if key_ not in ['dropout', 'batchnorm']:
                raise ValueError(
                    "The key entered into other_mlp_parameters is invalid.  Must be in ['dropout', 'batchnorm'].  Entered: " + str(
                        key_))

        self.do_dropout = False
        if 'dropout' in other_mlp_parameters:
            assert isinstance(other_mlp_parameters['dropout'], float), "dropout prob should be a float"
            assert 0.0 <= other_mlp_parameters['dropout'] < 1.0, "dropout prob needs to be in half-open interval [0, 1)"
            dropout_prob = other_mlp_parameters['dropout']

            if dropout_prob > 0:
                self.do_dropout = True
                self.dropout_layer = nn.Dropout(p=dropout_prob)

        # Set batchnorm flags, add layers later
        # If batchnorm is a key and its value is true:
        if 'batchnorm' in other_mlp_parameters and \
                isinstance(other_mlp_parameters['batchnorm'], bool) and \
                other_mlp_parameters['batchnorm']:
            self.do_batchnorm = True
        else:
            # If (batchnorm is not a key) OR (it has value False)
            self.do_batchnorm = False

        self.act = act
        self.num_hidden_layers = num_hidden_layers
        self.layers = []

        for ii in range(num_hidden_layers + 1):
            # Input to hidden
            if ii == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dim))
            # Hidden to output
            elif ii == num_hidden_layers:
                self.layers.append(nn.Linear(hidden_dim, out_dim))
            # Hidden to hidden
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

            #
            # Init weights with Xavier Glorot and set biases to zero
            #
            init.xavier_uniform_(self.layers[-1].weight)
            init.zeros_(self.layers[-1].bias)

            self.add_module("layer_{}".format(ii), self.layers[-1])
            #
            # Batchnorm
            #
            if self.do_batchnorm and ii < num_hidden_layers:
                # Get out_features in a robust way by calling getattr
                lin = getattr(self, "layer_{}".format(ii))
                self.layers.append(nn.BatchNorm1d(lin.out_features))
                self.add_module("batchnorm_{}".format(ii), self.layers[-1])

    def forward(self, x):
        for jj in range(self.num_hidden_layers + 1):
            layer = getattr(self, "layer_{}".format(jj))
            x = layer(x)
            if jj < self.num_hidden_layers:
                x = self.act(x)

                # Batchnorm and/or dropout
                if self.do_batchnorm:
                    bn = getattr(self, "batchnorm_{}".format(jj))
                    if x.dim() == 3:
                        batch_size = x.size(0)
                        x = bn(x.view(-1, x.size(2)))
                        x = x.view(batch_size, -1, x.size(1))
                    else:
                        x = bn(x)

                if self.do_dropout:
                    x = self.dropout_layer(x)
        return x