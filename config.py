import argparse
import os
from collections import OrderedDict

from configs.transformer_config import TransformerArgParser
from scalings import Scalings
from common.helper import PickleUtil, Util, print_stats
from util.constants import ExperimentData, ReplicatePermType, DEFAULT_RANDOM_GENERATION_SEED, Regularization, \
    TaskType, Constants, ModelType, InputType, GinModelType
from my_common.my_helper import enum_sprint
from configs.data_config import DataArgParser
from configs.model_config import ModelArgParser
from configs.graph_config import GraphDataArgParser
from configs.rpgin_config import RpGinArgParser

RESULT_DIR = "../results/"
MODEL_DIR = "../models/"
PickleUtil.check_create_dir(RESULT_DIR)
PickleUtil.check_create_dir(MODEL_DIR)

COMMON_ARG_PARSERS = [DataArgParser, ModelArgParser]
DATA_ARG_PARSERS = [GraphDataArgParser]
MODEL_ARG_PARSERS = [RpGinArgParser, TransformerArgParser]


class Config:
    def __init__(self, args=None):

        parser = argparse.ArgumentParser("Experiments with Regularization")
        # Gather all data and model arguments and add them to the parser
        for argparser in COMMON_ARG_PARSERS:
            self.add_common_args(parser, argparser)
        for argparser in DATA_ARG_PARSERS + MODEL_ARG_PARSERS:
            self.add_args(parser, argparser)

        ##########
        # PARAMETERS FOR DATA LOADING
        ##########
        parser.add_argument('-it', '--input_type', type=str, required=True,
                            help=f'Type of input. One of: {enum_sprint(InputType)}')
        parser.add_argument('-ix', '--cv_fold', type=int, help='Which fold in cross-validation (e.g., 0 thru 5)',
                            required=True)

        ##########
        # PARAMETERS FOR MODEL
        ##########
        parser.add_argument('-mt', '--model_type', type=str, required=True,
                            help=f'Type of model. One of: {enum_sprint(ModelType)}')

        ##########
        # PARAMETERS FOR OPTIMIZATION and COMPUTATION
        ##########

        ##########
        # PARAMETERS THAT CONTROL TRAINING AND EVALUATION
        ##########
        parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate for Adam Optimizer')
        # TODO: Implement L1 loss for maximum tasks, which makes more sense than MSE
        parser.add_argument('-ne', '--num_epochs', default=100, type=int,
                            help='Number of epochs for training at a minimum')
        parser.add_argument('-pi', '--patience_increase', default=100, type=int,
                            help='Number of epochs to increase patience')
        parser.add_argument('-ni', '--num_inf_perm', default=5, type=int, help='Number of inference-time permutations')
        parser.add_argument('-ei', '--eval_interval', default=50, type=int, help='Time interval of more-permutation inference')
        parser.add_argument('-et', '--eval_train', default=False, action='store_true',
                            help='Evaluate train loss/acc after backprop at each epoch')
        parser.add_argument('-sv', '--seed_val', default=-1, type=int,
                            help='Seed value, to get different random inits and variability')
        parser.add_argument('-umb', '--use_mini_batching', default=False, action='store_true',
                            help='Use mini-batching for training')
        parser.add_argument('-bs', '--batch_size', default=64, type=int,
                            help='Size of mini-batches when mini-batching is used')
        parser.add_argument('-dl', '--drop_last', default=False, action='store_true',
                            help='Drop the last incomplete batch, if the dataset size is not divisible by the batch size.')
        parser.add_argument('-bl', '--batch_limit', default=1024, type=int,
                            help='Maximum size of mini-batches or the full-batch.')
        parser.add_argument('-dsb', '--dont_shuffle_batches', default=False, action='store_true',
                            help="Don't shuffle batches for every epoch")
        parser.add_argument('--dont_permute_for_pi_sgd', default=False, action='store_true',
                            help='If set, pi-SGD training will NOT be used: data will not be permuted')

        ##########
        # PARAMETERS THAT CONTROL VARIANCE EVALUATION
        ##########
        # Test variability: every "eval_interval", assess the variability in farrow to permutations
        # This will track it during training, as well as at the end of training. We can always load a pre-trained
        # model and simply evaluate variability in the output file.
        parser.add_argument('--test_variability', action='store_true',
                            help="Run experiment to track variability of farrow through training every `eval_interval`")

        ##########
        # PARAMETERS FOR SCALING
        # Hypothesis: intermediate layers can arbitrarily scale and unscale their outputs, thwarting attempts to
        #             regularize permutation sensitivity according to the distance between vectors.
        ##########
        parser.add_argument('-sc_p', '--scaling_penalize_embedding', action='store_true', default=False,
                            help="Scaling Option to penalize the L2 norm of the embedding")
        parser.add_argument('-sc_ep', '--scaling_embedding_penalty', default=None,
                            help="Scaling Option: strength of L2 penalty. Must be set if --scaling_penalize_embedding")
        parser.add_argument('-sc_q', '--scaling_quantize', action='store_true', default=False,
                            help="Scaling Option: Add random normal noise to the latent representations")

        ##########
        # PARAMETERS THAT CONTROL REGULARIZATION
        ##########
        parser.add_argument('-r', '--regularization', type=str,
                            help='Type of regularization to apply', default='none')
        parser.add_argument('-r_strength', '--regularization_strength', default=None,
                            help='Regularization strength')
        # User be aware! r_eps effectively becomes multiplied by the regularization
        # penalty in finite difference penalties.  So it interacts with reg strength
        parser.add_argument('-r_eps', '--regularization_epsilon', default=None,
                            help='Size of finite difference for regularization')
        parser.add_argument('--fixed_edge', action='store_true',
                            help='Fix the edge of DS matrices for diff_step_edge regularization')
        # TODO: test '-r_np' later
        parser.add_argument('-r_np', '--regularization_num_perms', default=None,
                            help='Number of extra permutations for regularization')
        parser.add_argument('-ppred', '--penalize_prediction', action='store_true',
                            help="Penalize permutation sensitivity in the prediction yhat (default: latent rep)")
        parser.add_argument('-tpp', '--tangent_prop', action='store_true',
                            help='Use TangentProp regularization (default: False)')
        parser.add_argument('--tp_power', default=2, type=int,
                            help='The power in the penalty term for TangentProp (default: 2)')
        parser.add_argument('--tp_normed', action='store_true',
                            help='Compute norm, not only a power, of TangentProp loss by taking root (default:False)')
        parser.add_argument('-tp_ntv', '--tp_num_tangent_vectors', default=1, type=int,
                            help='Number of tangent vectors for TangentProp (default:1)')

        args = parser.parse_args(args)

        # ===================================================
        # Parameters for data and model types
        # ===================================================
        self.input_type = InputType[args.input_type.lower()]
        self.model_type = ModelType[args.model_type.lower()]

        # assign parsed hyperparameters to attributes of config containers
        if self.input_type == InputType.set:
            assert self.model_type != ModelType.rpgin, "Set data cannot be handled by RpGin."
            raise NotImplementedError
        elif self.input_type == InputType.graph:
            self.data_cfg = self.parsed_to_cfg(args, GraphDataArgParser)
        else:
            raise NotImplementedError(f"Unable to handle {self.input_type} yet.")

        if self.model_type == ModelType.rpgin:
            self.model_cfg = self.parsed_to_cfg(args, RpGinArgParser)
        elif self.model_type == ModelType.lstm:
            raise NotImplementedError
        elif self.model_type == ModelType.transformer:
            self.model_cfg = self.parsed_to_cfg(args, TransformerArgParser)
        else:
            raise NotImplementedError(f"Haven't implemented {self.model_type} yet.")

        # ===================================================
        # Parameters for cross-validation
        # ===================================================
        self.cv_fold = args.cv_fold
        assert 0 <= self.cv_fold
        if self.data_cfg.experiment_data == ExperimentData.rp_paper:
            assert self.cv_fold < 5  # While we load the rp_paper data, user may enter n_splits other than 5
        else:
            assert self.cv_fold < self.data_cfg.n_splits

        # ===================================================
        # Parameters for optimization and computation
        # ===================================================

        # ===================================================
        # Parameters for experiment data
        # ===================================================
        if self.data_cfg.experiment_data in (ExperimentData.rp_paper, ExperimentData.customized):
            self.task_type = TaskType.multi_classification
            if self.data_cfg.experiment_data == ExperimentData.rp_paper:
                self.data_cfg.sparse = True
                self.data_id = ""
                self.sample_random_state = None
            elif self.data_cfg.experiment_data == ExperimentData.customized:
                self.data_id = self.data_cfg.get_graph_id_string()
                self.sample_random_state = self.data_cfg.sample_random_state
        elif self.data_cfg.experiment_data in (ExperimentData.smp, ExperimentData.er_edges):
            self.data_id = self.data_cfg.get_graph_id_string()
            self.sample_random_state = None
        elif self.data_cfg.experiment_data in (ExperimentData.ba, ExperimentData.ger):
            self.data_id = self.data_cfg.get_graph_id_string()
            self.sample_random_state = self.data_cfg.sample_random_state
        else:
            raise ValueError(f"Invalid ExperimentData given: {self.data_cfg.experiment_data}")

        self.load_sparse = self.data_cfg.sparse
        self.data_path = self.data_cfg.get_data_path()
        #
        # Check if data exists
        #
        if os.path.exists(self.data_path):
            print(f"Will load {self.data_path}")
        else:
            raise FileNotFoundError(f"{self.data_path} does not exist! Please create data using \'create_data.py\'.")

        # ===================================================
        # Parameters that control training, validation and/or testing
        # ===================================================
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.patience_increase = args.patience_increase
        assert self.patience_increase >= 0, "Patience is a non-negative int"
        self.num_inf_perm = args.num_inf_perm
        self.eval_interval = args.eval_interval
        assert self.eval_interval > 0, "eval_interval must be a positive integer"
        self.eval_train = args.eval_train
        self.seed_val = args.seed_val

        self.use_mini_batching = args.use_mini_batching  # will be checked in Train
        # If mini-batching options are specified, we must be using mini-batching
        assert args.batch_size > 0 and args.batch_limit > 0, "batch_size and batch_limit must be strictly positive."
        if args.batch_size != parser.get_default('batch_size') or args.drop_last != parser.get_default('drop_last'):
            assert self.use_mini_batching, \
                "Non-default values of mini-batching options were specified, but --use_mini_batching is False"
        self.batch_size = args.batch_size # will be checked in Train
        self.drop_last = args.drop_last
        self.batch_limit = args.batch_limit
        self.shuffle_batches = not args.dont_shuffle_batches

        # ===================================================
        # Parameters that control Pi-SGD
        # ===================================================
        # TODO: Can we assert args.dont_permute_for_pi_sgd and args.num_inf_perm == 1 ?
        if args.dont_permute_for_pi_sgd and args.num_inf_perm != parser.get_default('num_inf_perm'):
            raise ValueError("Cannot specify `--dont_permute_for_pi_sgd` and non-default num_inf_perm")
        if not args.dont_permute_for_pi_sgd and self.model_type == ModelType.rpgin:
            assert self.model_cfg.gin_model_type == GinModelType.rpGin, "Pi-SGD cannot be used with Gin models other than rpGin"
        self.permute_for_pi_sgd = not args.dont_permute_for_pi_sgd

        # ===================================================
        # Parameters that control variance evaluation
        # ===================================================
        self.test_variability = args.test_variability

        # ===================================================
        # Parameters that control scaling
        # (Scaling is relevant for both baseline and regularization, since we may want to
        #  do a fair comparison)
        # ===================================================
        # Build a scaling struct, and let the class do the input validation
        self.scaling = Scalings(penalize_embedding=args.scaling_penalize_embedding,
                                quantize=args.scaling_quantize,
                                embedding_penalty=args.scaling_embedding_penalty)

        # ===================================================
        # Parameters that control regularization
        # ===================================================

        self.regularization = Regularization[args.regularization.lower()]
        if self.regularization != Regularization.none:
            if self.model_type == ModelType.rpgin:
                assert self.model_cfg.gin_model_type == GinModelType.rpGin, "Regularizers cannot be used with Gin models other than rpGin"
        self.regularization_strength = args.regularization_strength
        self.regularization_eps = args.regularization_epsilon
        self.num_reg_permutations = args.regularization_num_perms
        self.fixed_edge = args.fixed_edge

        if self.regularization_strength is not None:
            self.regularization_strength = float(self.regularization_strength)
            assert self.regularization_strength >= 0.0  # Allow 0.0 for debug and easy hyperparameter search

        if self.regularization != Regularization.none:
            assert self.regularization_eps is not None
            self.regularization_eps = float(self.regularization_eps)
            assert self.regularization_eps > 0.0

        if self.num_reg_permutations is not None:
            self.num_reg_permutations = int(self.num_reg_permutations)
            assert self.num_reg_permutations > 0

        self.penalize_prediction = args.penalize_prediction
        self.tangent_prop = args.tangent_prop

        # If tangent prop options are specified, we must be using tangent prop
        if args.tp_normed or args.tp_power != parser.get_default('tp_power') \
                or args.tp_num_tangent_vectors != parser.get_default('tp_num_tangent_vectors'):
            assert self.tangent_prop, \
                "Non-default values of tp_normed or tp_power were specified, but --tangent_prop is False"

        assert args.tp_power > 0
        assert args.tp_num_tangent_vectors > 0
        self.tp_config = OrderedDict()
        self.tp_config['tp_power'] = args.tp_power
        self.tp_config['tp_normed'] = args.tp_normed
        self.tp_config['tp_num_tangent_vectors'] = args.tp_num_tangent_vectors

    def add_common_args(self, parser, subparser):
        """
        Wrapper for adding common arguments from other config files
        """
        return subparser.add_common_args(parser)

    def add_args(self, parser, subparser):
        """
        Wrapper for adding arguments from other config files
        """
        return subparser.add_args(parser)

    def parsed_to_cfg(self, parsed, subparser):
        """
        Wrapper for initializing active data/model config containers
        """
        # parsed arguments must not be None
        assert parsed is not None
        return subparser(None, parsed)

    @staticmethod
    def params_to_str(params):
        params = map(lambda x: x if not isinstance(x, bool) else ("T" if x else "F"), params)
        return "_".join(map(str, params))

    def _data_params(self):
        """
        Return Data Information
        """
        data_params = [self.input_type, self.data_cfg.experiment_data.name, self.cv_fold]

        if self.data_cfg.experiment_data == ExperimentData.customized:
            sps_str = "sparse_in" if self.load_sparse else "dense_in"
            sample_random_state_str = ""  # Fill in only if non-default
            if self.data_cfg.permutation_strategy == ReplicatePermType.deterministic:
                perm_type_str = 'det'
            elif self.data_cfg.permutation_strategy == ReplicatePermType.sampled:
                perm_type_str = 'rand'
                if self.sample_random_state != DEFAULT_RANDOM_GENERATION_SEED:
                    sample_random_state_str = f"permseed_{self.sample_random_state}"

            data_params += [self.data_cfg.num_vertices, self.data_cfg.num_skip_lengths, self.data_cfg.num_permutations,
                            self.data_cfg.n_splits, self.data_cfg.l_type.name, sps_str, perm_type_str]
            if len(sample_random_state_str) > 0:
                data_params += [sample_random_state_str]
        else:
            data_params += [self.data_id]

        return data_params

    def _model_params(self):
        """
        Return Model Information
        """
        model_params = [self.model_type.name]
        model_params += self.model_cfg.get_model_id_list()
        if hasattr(self.model_cfg, 'use_batchnorm') and self.model_cfg.use_batchnorm:
            model_params += [self.batch_limit]  # batch limit impacts batch sizes used by batchnorm

        return model_params

    def _train_params(self):
        """
        Return Training Information
        """
        train_params = [self.permute_for_pi_sgd, self.learning_rate, self.patience_increase, self.seed_val,
                         self.num_epochs, self.num_inf_perm, self.eval_interval, self.eval_train, self.shuffle_batches]
        train_params += ["mini_batches", self.batch_size, self.drop_last] if self.use_mini_batching else ["full_batch"]
        #
        # Add Scaling information
        #
        if self.scaling.scaling_active:
            train_params += ["s"]
            train_params += [self.scaling.penalize_embedding]
            if self.scaling.penalize_embedding:
                train_params += [self.scaling.embedding_penalty]
            train_params += [self.scaling.quantize]
        #
        # Add Regularization Information
        #
        if self.regularization != Regularization.none:
            train_params += [self.regularization.name]
            if self.regularization in (Regularization.diff_step_edge, Regularization.diff_step_basis):
                train_params += [self.fixed_edge]
            train_params += [self.num_reg_permutations, self.regularization_strength, self.regularization_eps,
                             self.penalize_prediction, self.tangent_prop]
            if self.tangent_prop:
                train_params += list(self.tp_config.values())
        #
        # Add General Information
        #

        return train_params

    def _model_id(self):
        #
        # Add dataset information
        #
        data_params = ["DAT"] + self._data_params()
        #
        # Add Model Information
        #
        model_params = ["MOD"] + self._model_params() + ["TRA"] + self._train_params()

        model_id = self.params_to_str(data_params + model_params)
        # Last resort: hash the filenames, at the cost of human readability
        if len(model_id) > Constants.MODEL_ID_MAX:
            print_stats(f"Model ID too long! MAX: {Constants.MODEL_ID_MAX}; Length of '{model_id}': {len(model_id)}.")
            print_stats("Hashing model_id by md5...")
            model_id = Util.md5(model_id)
            print_stats("Hashing done.")
        return model_id

    def output_file_name(self):
        return f"{RESULT_DIR}{self._model_id()}.output.txt"

    def stats_file_name(self):
        return f"{RESULT_DIR}{self._model_id()}.stats.txt"

    def log_file_name(self):
        return f"{RESULT_DIR}{self._model_id()}.log"

    def checkpoint_file_name(self):
        return f"{MODEL_DIR}{self._model_id()}.checkpoint.pth.tar"


if __name__ == "__main__":
    cfg = Config()
    print("\n", cfg.stats_file_name(), "\n")
    # print(cfg.scaling)

