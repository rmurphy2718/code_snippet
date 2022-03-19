from argparse import ArgumentParser
from util.constants import Activation, GinModelType
from my_common.my_helper import enum_sprint, is_positive_int
from configs.model_config import ModelArgParser

class TransformerArgParser(ModelArgParser):

    @classmethod
    def add_args(cls, parser):
        ##########
        # PARAMETERS THAT CONTROL GIN MODELS
        ##########
        assert isinstance(parser, ArgumentParser)
        # TODO for the max-regression task from set transformer,
        #  we should first implement L1 loss (in config.py)

        group = parser.add_argument_group('Transformer', 'HyperParameters for Transformer')

        # Positional encoding
        group.add_argument('-trans_npe', '--trans_no_positional_encoding',
                           action='store_true', default=False,
                           help="Disable learned position encoding.")
        # Dictionary dim
        group.add_argument('--dictionary_embedding_dim', type=int,
                           default=None,
                           help="Dimension of the fixed embedding, required if set of scalars.")
        # Num inducing points (using default from Set Transformer's code)
        group.add_argument('-trans_nind', '--trans_num_inds', type=int,
                           default=32,
                           help="Number of inducing points for set transformer")
        # Hidden dimension (using default from Set Transformer's code)
        group.add_argument('-trans_dh', '--trans_dim_hidden', type=int,
                           default=128,
                           help="Dimension of the hidden layer in transformer")
        # Attention heads (using default from Set Transformer's code)
        group.add_argument('-trans_nhd', '--trans_num_heads', type=int,
                           default=4,
                           help="Number of attention heads in transformer")
        # Layer Normalization (using default from Set Transformer's code)
        group.add_argument('--layer_norm', default=False, action='store_true',
                           help='User layer normalization')

    def assign_parsed(self, args):
        """ Set params that control model """
        self.trans_positional_encoding = not args.trans_no_positional_encoding
        self.dictionary_embedding_dim = args.dictionary_embedding_dim
        self.trans_num_inds = args.trans_num_inds
        self.trans_dim_hidden = args.trans_dim_hidden
        self.trans_num_heads = args.trans_num_heads
        self.layer_norm = args.layer_norm

        if self.dictionary_embedding_dim is not None:
            assert is_positive_int(self.dictionary_embedding_dim)

    def get_model_id_list(self):
        _lst = [self.trans_positional_encoding, self.dictionary_embedding_dim,
                self.trans_num_inds, self.trans_dim_hidden, self.trans_num_heads,
                self.layer_norm]
        return _lst

if __name__ == '__main__':
    pass
    # args = TransformerArgParser(['-bn -ngl 3 '])  # Should get not instantiatiable error
    # print(args.__dict__)
