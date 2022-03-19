import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

# ===========================================================
# Modules for the Set Transformer
# https://github.com/juho-lee/set_transformer
# ===========================================================
from model_gs import ModelGS
from my_common.my_helper import is_positive_int


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# ===========================================================
# Set Transformer with Positional Encoding
# Extended from https://github.com/juho-lee/set_transformer
# ============================================================
class Transformer(ModelGS):
    def __init__(self, dim_input, length_input, dim_output, use_positional_encoding, num_outputs=1,
                 embedding_vocab_size=None, dictionary_embedding_dim=None,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        """
        Params from train.py (after data loaded): dim_input, length_input, dim_output, embedding_vocab_size
        Params from config.py: use_positional_encoding, dictionary_embedding_dim, num_inds, dim_hidden, num_heads, ln

        num_outputs is in the original code, but we'll stick to 1 for our purposes.

        """
        # ------------------------------------------------------------------------------------
        # Checks
        # ------------------------------------------------------------------------------------
        for int_var in [dim_input, num_outputs, dim_output, num_inds, dim_hidden, num_heads, length_input]:
            assert is_positive_int(int_var), f"Expected positive integer but got {int_var} in SetTransformer"
        assert isinstance(ln, bool) and isinstance(use_positional_encoding, bool)

        if num_outputs != 1:
            raise NotImplementedError("Only implemented for one-output case. (One output, which can be a vector)")

        if dim_input == 1:  # Must use a fixed nn.Embedding
            if (not is_positive_int(embedding_vocab_size)) or (not  is_positive_int(dictionary_embedding_dim)):
                raise ValueError(f"""Transformer got dim_input == 1,
                 Expects positive integers embedding_vocab_size and dictionary_embedding_dim but got
                 {embedding_vocab_size} {dictionary_embedding_dim} respectively """)
        # ------------------------------------------------------------------------------------
        # Model components: (1) Map to latent space with Positional Encoding (2) Set Transformer
        # ------------------------------------------------------------------------------------
        super(Transformer, self).__init__()
        # Dictionary lookup for scalar integer inputs
        if dim_input == 1:
            self.embedding = nn.Embedding(embedding_vocab_size, dictionary_embedding_dim)
            init.uniform_(self.embedding.weight, a=-0.5, b=0.5)
            self.embedding.weight.requires_grad = False

            transformer_input_dim = dictionary_embedding_dim
        else:
            self.embedding = None
            transformer_input_dim = dim_input

        # Positional encoding vectors (assumes fixed size inputs)
        if use_positional_encoding:
            self.positional_vectors = nn.Parameter(torch.randn(1, length_input, transformer_input_dim))
        else:
            self.positional_vectors = None

        # (2) Set transformer modules, directly from https://github.com/juho-lee/set_transformer
        self.enc = nn.Sequential(
            ISAB(transformer_input_dim, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))

    def embed(self, x):
        """
        Position embeddings described in https://arxiv.org/pdf/1705.03122.pdf,
        which is cited by the original transformer as a learnable positional embedding
        """
        assert x.ndim == 3, "Transformer embed expects 3D input: (batch, set size, element dimension)"
        # Use a dictionary embedding for scalar integer inputs.
        if self.embedding is not None:
            x = self.embedding(x)
            # Squeeze out the useless dimension that embeddding causes
            x = x.squeeze(2)
        # Positional Encoding
        if self.positional_vectors is not None:
            assert x.shape[1:3] == self.positional_vectors.shape[1:3]
            x = torch.add(x, self.positional_vectors)

        return x

    def predict(self, x):
        out = self.dec(self.enc(x))
        # Squeeze from (batch, num outputs, output dimension) to (batch, output dimension)
        out = out.squeeze(1)
        return out

    def __str__(self):
        """ Add extra information about the model for logging. """
        _str = "Transformer information:\n"
        if self.embedding is None:
            _str += "A fixed embedding has NOT been included in the model.\n"
        if self.positional_vectors is None:
            _str += "Positional encoding is NOT active.\n"
        else:
            _str += f"Additive positional encoding, init as random normal, shape: {self.positional_vectors.shape}.\n"

        _str += self.__repr__()

        return _str


if __name__ == "__main__":
    import sys
    from config import Config
    from util.constants import TaskType, Regularization, InputType
    from regularizers import JpRegularizer
    from scalings import Scalings

    # Test 1 ---------- Using config for Transformer seems syntactically OK
    # >> This tests some invalid models and sees that they raise errors
    for dim_input in [2, 1]:
        for _ded in [None, 8]:
            for _ln in [True, False]:
                    args = "-ix 0 -mt transformer -data rp_paper -it graph"  # data and it just for testing's sake.
                    if _ded is not None:
                        args += f" --dictionary_embedding_dim {_ded}"
                    if _ln:
                        args += f" --layer_norm"

                    try:
                        cfg = Config(args.strip().split(" "))
                        print(cfg.stats_file_name())
                    except:
                        print("Error in creating config")

                    ln = cfg.model_cfg.layer_norm
                    num_head = cfg.model_cfg.trans_num_heads
                    dim_hid = cfg.model_cfg.trans_dim_hidden
                    num_ind = cfg.model_cfg.trans_num_inds
                    ded = cfg.model_cfg.dictionary_embedding_dim
                    pe = cfg.model_cfg.trans_positional_encoding

                    try:  # Making a transformer
                        model = Transformer(dim_input=dim_input,  # Determined from train.py (arbitrary here)
                                            length_input=5,  # Determined from train.py (arbitrary here)
                                            dim_output=1,  # Determined from train.py (arbitrary here)
                                            embedding_vocab_size=5,  # Determined from train.py (arbitrary here)
                                            use_positional_encoding=pe,  # Determined from config
                                            dictionary_embedding_dim=ded,  # Determined from config
                                            num_inds=num_ind,  # Determined from config
                                            dim_hidden=dim_hid,  # Determined from config
                                            num_heads=num_head,  # Determined from config
                                            ln=ln  # Determined from config
                                            )
                        regularizer = JpRegularizer(regularization=Regularization.none, input_type=InputType.set)
                        scalings = Scalings(False, False)
                        model.init_model_gs(permute_for_pi_sgd=True,
                                            regularizer=regularizer,
                                            scaling=scalings,
                                            task_type=TaskType.regression)
                        print(model)
                    except ValueError as ve:
                        print("\n\n\n\n Value error making Transformer")
                        print(ve)
                    finally:
                        print("No errors for this config")
                        print(f"dim_input={dim_input}\tdictionary={_ded}\tlayer_norm={_ln}")

    #
    # Simply test the config (ran interactively for different choices)
    #
    args = "-ix 0 -mt transformer -data rp_paper -it graph"  # data and it just for testing's sake.
    args += ' -trans_nind 222 --dictionary_embedding_dim 23  '
    cfg = Config(args.strip().split(" "))
    cfg.stats_file_name()

    # -----------------
    # Test 2 :Using modelGS with Transformer seems syntactically OK
    # -----------------
    seq_len = 5
    vocab_size = 10
    batch_size = 2
    dictionary_embedding_dim = 8
    dim_input = 2
    model = Transformer(dim_input=dim_input,  # Determined from train.py (arbitrary here)
                        length_input=seq_len,  # Determined from train.py (arbitrary here)
                        dim_output=1,  # Determined from train.py (arbitrary here)
                        embedding_vocab_size=vocab_size,  # Determined from train.py (arbitrary here)
                        use_positional_encoding=True,  # Determined from config
                        dictionary_embedding_dim=dictionary_embedding_dim,  # Determined from config
                        num_inds=2,  # Determined from config
                        )

    model.init_model_gs(permute_for_pi_sgd=False,
                        regularizer=JpRegularizer(regularization=Regularization.none, input_type=InputType.set),
                        scaling=Scalings(False, False),
                        task_type=TaskType.regression)


    x = torch.randint(vocab_size, (batch_size, seq_len, dim_input))
    predictions, latent = model.forward(x)
    print(predictions.shape)
    print(latent.shape)

    if dim_input == 1:  # PE is used.
        assert latent.shape == (batch_size, seq_len, dictionary_embedding_dim)
    else:
        assert latent.shape == x.shape
