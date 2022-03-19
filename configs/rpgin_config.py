from argparse import ArgumentParser
from util.constants import Activation, GinModelType
from my_common.my_helper import enum_sprint
from configs.model_config import ModelArgParser

class RpGinArgParser(ModelArgParser):

    @classmethod
    def add_args(cls, parser):
        ##########
        # PARAMETERS THAT CONTROL GIN MODELS
        ##########
        assert isinstance(parser, ArgumentParser)

        group = parser.add_argument_group('RpGin', 'HyperParameters for RpGin')

        group.add_argument('-bn', '--use_batchnorm', default=False, action="store_const", const=True,
                            help='Boolean flag, should batch normalization be implemented?')
        group.add_argument('-ddp', '--dense_dropout_prob', default=0.0, type=float,
                            help='Dropout probability for the dense layer')
        group.add_argument('-nh', '--num_mlp_hidden', default=2, type=int,
                            help='Number of hidden layers in the MLP')
        group.add_argument('-ngl', '--num_gnn_layers', default=5, type=int,
                            help='Number of iterations of WL-like aggregation')
        group.add_argument('-gmt', '--gin_model_type', default=GinModelType.rpGin.name, type=str,
                            help='Either regularGin or dataAugGin or rpGin. Note: the model choice influences how the data is loaded/used')

        # TODO: Make activation function global?
        group.add_argument('-act', '--activation_function', default=Activation.ReLU.name, type=str,
                            help=f'Activation functions used in the aggregator MLP. One of {enum_sprint(Activation)}')
        group.add_argument('-ez', '--set_epsilon_zero', default=False, action="store_const", const=True,
                            help='Boolean flag, should epsilon be set to zero?  By default, we train epsilon via backprop')
        group.add_argument('-ed', '--vertex_embed_dim', default=16, type=int,
                            help='Dimension of each vertex embedding')
        group.add_argument('-nm', '--mlp_hidden_dim', default=16, type=int,
                            help='Number of hidden units in the aggregator MLP')
        parser.add_argument('-ohd', '--onehot_id_dim', default=30, type=int,
                            help='For use with rpGin.  Dimension of the one-hot ID.')

    def assign_parsed(self, args):
        #
        # Set params that control model
        #
        self.use_batchnorm = args.use_batchnorm
        self.dense_dropout_prob = args.dense_dropout_prob
        assert 0 <= self.dense_dropout_prob < 1, "Invalid range for dense_dropout_prob"
        self.num_mlp_hidden = args.num_mlp_hidden
        self.num_gnn_layers = args.num_gnn_layers
        self.gin_model_type = GinModelType[args.gin_model_type]
        self.act = Activation[args.activation_function]
        self.set_epsilon_zero = args.set_epsilon_zero
        self.vertex_embed_dim = args.vertex_embed_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.onehot_id_dim = args.onehot_id_dim

    def get_model_id_list(self):
        return [self.act.name, self.gin_model_type, self.dense_dropout_prob, self.set_epsilon_zero, self.vertex_embed_dim,
                self.mlp_hidden_dim, self.num_gnn_layers, self.num_mlp_hidden, self.onehot_id_dim, self.use_batchnorm]

if __name__ == '__main__':
    args = RpGinArgParser(['-bn -ngl 3 '])  # Should get not instantiatiable error
    print(args.__dict__)