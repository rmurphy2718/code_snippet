import torch
import torch.nn as nn

from models.mlp import MLP
from model_gs import ModelGS
from models.model_utils import get_act_func
from util.constants import Activation
from common.helper import log_stats
from my_common.my_helper import is_tensor, diags_all_one


# ================================================================
#
# Parent class generates MLPs for the vertex embedding
#   for both the whole-graph and one-graph classes
# (carry-over from previous implementations)
# ================================================================
class GinParent(ModelGS):
    def __init__(self, input_data_dim, num_agg_steps, vertex_embed_dim, mlp_num_hidden, mlp_hidden_dim, mlp_act,
                 vertices_are_onehot, logger, epsilon_tunable, other_mlp_parameters={}):
        """
        :param input_data_dim: Dimension of the vertex attributes
        :param num_agg_steps: K, the number of WL iterations.  The number of neighborhood aggregations
        :param vertex_embed_dim: Dimension of the `hidden' vertex attributes iteration to iteration
        :param mlp_num_hidden: Number of layers.  1 layer is sigmoid(Wx). 2 layers is Theta sigmoid(Wx)
        :param mlp_hidden_dim: Number of neurons in each layer
        :param vertices_are_onehot: Are the vertex features one-hot-encoded?  Boolean
        :param vertex_embed_only: We are only interested in the vertex embeddings at layer K.
                   ..not forming a graph-wide embedding
                   ..note: this is helpful for debug
        :param epsilon_tunable: If True, the epsilons in the GIN formulation will
                                be trainable parameters and learned w/ backprop
                                Otherwise, they are untrained and fixed at zero.
        """
        # Input check
        for int_var in [input_data_dim, num_agg_steps, vertex_embed_dim, mlp_num_hidden, mlp_hidden_dim]:
            assert isinstance(int_var, int) and int_var > 0, \
                f"GinParent expected nonnegative integer but got {int_var} for some input"
        for bool_var in [vertices_are_onehot, epsilon_tunable]:
            assert isinstance(bool_var, bool), \
                f"GinParent expected a bool but got type {type(bool_var)} for some input"
        assert isinstance(mlp_act, Activation)

        self.vertices_are_onehot = vertices_are_onehot
        self.input_data_dim = input_data_dim
        self.num_agg_steps = num_agg_steps
        self.vertex_embed_dim = vertex_embed_dim
        self.logger = logger
        self.epsilon_tunable = epsilon_tunable
        self.act = get_act_func(mlp_act)
        log_stats(self.logger, "mlp info", msg=f"{mlp_act.name} will be used in aggregation MLP.")
        #
        # Info about dropout and batchnorm
        #
        if 'dropout' in other_mlp_parameters and other_mlp_parameters['dropout'] > 0:
            log_stats(self.logger, "mlp info",
                      msg=f"Dropout will be used in aggregation MLP.  Probability: {other_mlp_parameters['dropout']}")
            do_dropout = True
        else:
            log_stats(self.logger, "mlp info",
                      msg=f"Dropout will NOT be used in aggregation MLP.")
            do_dropout = False

        if 'batchnorm' in other_mlp_parameters and \
                isinstance(other_mlp_parameters['batchnorm'], bool) and \
                other_mlp_parameters['batchnorm']:
            log_stats(self.logger, "mlp info",
                      msg=f"Batchnorm will be used in aggregation MLP.")
            do_batchnorm = True
        else:
            log_stats(self.logger, "mlp info",
                      msg=f"Batchnorm will NOT be used in aggregation MLP.")
            do_batchnorm = False

        # Info again if both dropout and batchnorm are used
        # This structure: ReLU -> IC layer (batchnorm + dropout) -> Weights
        # is recommended by https://arxiv.org/abs/1905.05928v1
        if do_dropout and do_batchnorm:
            log_stats(self.logger, "mlp info",
                      msg="Both dropout and batchnorm are selected in GIN.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Init model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        super(GinParent, self).__init__()

        if epsilon_tunable:
            log_stats(self.logger, "GIN data info",
                      msg="User indicated: epsilon is tunable.\nEpsilons will be learned from backprop,  init to zero")

            self.epsilons = nn.ParameterList()
            for ll in range(num_agg_steps):
                epsilon_k = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.epsilons.append(epsilon_k)
        else:
            log_stats(self.logger, "GIN data info",
                      msg="User indicated: epsilon is NOT tunable.\nEpsilon won't be learned with backprop, fixed to 0")

        # Layers for embedding.
        self.gin_layers = []
        for itr in range(num_agg_steps):
            #
            # In the first step,
            # if vertex attributes are not one-hot,
            # we need an extra MLP in the first layer
            #
            if itr == 0:
                self.gin_layers.append(MLP(in_dim=self.input_data_dim,
                                           hidden_dim=mlp_hidden_dim,
                                           out_dim=vertex_embed_dim,
                                           num_hidden_layers=mlp_num_hidden,
                                           act=self.act,
                                           other_mlp_parameters=other_mlp_parameters))
                if vertices_are_onehot:
                    log_stats(self.logger, "GIN data info",
                              msg=f"User indicated that vertices are one-hot.")

                    # Add the first aggregator layer and move on to next iteration
                    self.add_module("agg_{}".format(itr), self.gin_layers[-1])
                    continue
                else:
                    log_stats(self.logger, "GIN data info",
                              msg=f"User indicated that vertices are NOT one-hot, will have extra layer.")
                    self.add_module("raw_embedding_layer", self.gin_layers[-1])
                    # ... Goes on to next line of code to add agg_0

            # Assume all 'hidden' vertex features are of the same dim
            self.gin_layers.append(MLP(in_dim=vertex_embed_dim,
                                       hidden_dim=mlp_hidden_dim,
                                       out_dim=vertex_embed_dim,
                                       num_hidden_layers=mlp_num_hidden,
                                       act=self.act,
                                       other_mlp_parameters=other_mlp_parameters))

            self.add_module("agg_{}".format(itr), self.gin_layers[-1])

        #
        # Compute graph embedding dim (note it won't be used if
        #   we only want vertex embeds, but that's fine)
        self.graph_embed_dim = self.input_data_dim + vertex_embed_dim * num_agg_steps


# ========================================
class GinMultiGraph(GinParent):
    """
    Designed for graph classification
    """

    def __init__(self, adjmat_list, input_data_dim, num_agg_steps, vertex_embed_dim, mlp_num_hidden, mlp_hidden_dim,
                 mlp_act, vertices_are_onehot, target_dim, logger, epsilon_tunable=True, other_mlp_parameters={}):
        """
        Most parameters defined in the parent class
        :param adjmat_list: List of all adjmats to be considered
        Purpose: force input validation, but not saved to any variable.
        The user will enter the graphs in the dataset.  In principle, the graphs passed to
        initialize could be different than those used in the forward method; it is up
        to the user to properly do input validation on all desired graphs
        This is NOT stored as a self object; rest easy we're not wasting memory
        :param target_dim: Dimension of the response variable (the target)
        :param epsilon_tunable: Do we make epsilon in equation 4.1 tunable
        """
        assert isinstance(target_dim, int) and target_dim > 0
        self.target_dim = target_dim

        assert all(list(map(is_tensor, adjmat_list))), "All adjacency matrices must be tensors"
        assert all(list(map(diags_all_one, adjmat_list))), "All adjacency matrices must have ones on the diag"

        super(GinMultiGraph, self).__init__(input_data_dim=input_data_dim,
                                            num_agg_steps=num_agg_steps,
                                            vertex_embed_dim=vertex_embed_dim,
                                            mlp_num_hidden=mlp_num_hidden,
                                            mlp_hidden_dim=mlp_hidden_dim,
                                            mlp_act=mlp_act,
                                            vertices_are_onehot=vertices_are_onehot,
                                            logger=logger,
                                            epsilon_tunable=epsilon_tunable,
                                            other_mlp_parameters=other_mlp_parameters
                                            )

        self.add_module("last_linear", nn.Linear(self.graph_embed_dim, self.target_dim))

    def embed(self, input_adjmats, X):
        """
        Get a graph-level prediction for a list of graphs
        :param X: Vertex attributes for every vertex in every batch
        :param input_adjmats: Adjacency matrices in batch
        """
        # Check dimension and type of inputs.
        assert is_tensor(input_adjmats) and input_adjmats.dim() == 3
        assert is_tensor(X) and X.dim() == 3
        assert X.shape[0:2] == input_adjmats.shape[0:2]

        # Get embedding from X
        graph_embedding = torch.sum(X, dim=1)

        if not self.vertices_are_onehot:
            embedding = getattr(self, "raw_embedding_layer")
            H = embedding(X)
        else:
            H = X

        for kk in range(self.num_agg_steps):
            # Sum self and neighbor
            if not self.epsilon_tunable:
                # Aggregation in matrix form: (A + I)H
                agg_pre_mlp = torch.matmul(input_adjmats, H)
            else:
                #
                # Add epsilon to h_v, as in equation 4.1
                # Note that the proper matrix multiplication is
                # (A + (1+epsilon)I)H = (A+I)H + epsilon H
                #
                # Our implementation avoids making epsilon interact with the
                #  adjacency matrix, which would make PyTorch want to
                #  track gradients through the adjmat by default
                #
                epsilon_k = self.epsilons[kk]
                agg_pre_mlp = torch.matmul(input_adjmats, H) + epsilon_k * H

            mlp = getattr(self, "agg_{}".format(kk))
            H = mlp(agg_pre_mlp)  # (num_graphs, num_vertices, vertex_feature_dimension)
            #
            layer_k_embed = torch.sum(H, dim=1)

            graph_embedding = torch.cat((graph_embedding, layer_k_embed), dim=1)

        return graph_embedding

    def predict(self, graph_embedding):
        # Map to a prediction
        last_layer = getattr(self, "last_linear")
        final = last_layer(graph_embedding)
        return final

