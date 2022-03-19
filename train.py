import copy
import os
import time
from operator import itemgetter

from tqdm import tqdm

from common.helper import (DataFrameBuilder, GitInfo, PickleUtil, Util,
                           log_stats, print_stats)
from common.json_dump import JsonDump
from common.torch_util import TorchUtil as tu
from config import Config, GinModelType
from configs.graph_config import SYNTHETIC_X_EYE, SYNTHETIC_X_UNITY
from models.GIN_model import *
from models.GIN_utils import construct_onehot_ids
from my_common.my_helper import p_freq_table
from regularizers import *
from util.constants import InputType, ModelType, TaskType
from util.training_utils import *

DEBUG = int(os.environ.get("pydebug", 0)) > 0
if DEBUG:
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=27189, stdoutToServer=True, stderrToServer=True, suspend=False)

PARALLELISM = 8

_parallelism = torch.get_num_threads()
if PARALLELISM < _parallelism:
    torch.set_num_threads(PARALLELISM)
    print(f"Parallelism changed from {_parallelism} to {torch.get_num_threads()}")
else:
    print(f"Parallelism was not changed from original num threads, and is {torch.get_num_threads()}")


def accuracy(yhat, y):
    """ Compute accuracy """
    assert y.ndim == 1
    assert yhat.ndim == 2
    if yhat.shape[1] > 1:
        scores = torch.argmax(yhat, dim=1)
        num_correct = torch.sum(scores == y).item()
    else:
        y_pred = yhat.squeeze() > 0
        num_correct = torch.sum(y_pred.to(y.dtype) == y).item()
    return num_correct / float(len(y))


class Train:

    @classmethod
    def load_data(cls, args):
        print_stats("---Loading data...---", path=args.data_path)
        dataset = PickleUtil.read(args.data_path)

        #
        # Load cv splits
        #
        fold_to_indices = dataset['cv_indices']
        train_idx = fold_to_indices[args.cv_fold]['train']
        val_idx = fold_to_indices[args.cv_fold]['val']

        #
        # Load general stats
        #
        target_dim = dataset['target_dim']
        #
        # Handle the task type, as well as some legacies
        # >> Legacy datasets did not have 'task_type', so we define them in config.py, so they are in args.
        # >> Thus, we must put them in args, even though it makes the code ugly...
        if 'task_type' in dataset:
            args.task_type = dataset['task_type']
        elif hasattr(args, 'task_type'):
            assert isinstance(args.task_type, TaskType)
        else:
            raise ValueError("Must indicate Task Type so the loss function can be determined")

        #
        # Load targets
        #
        y = dataset['y']
        # sanity check of target
        if y.ndim == 1:
            assert target_dim == 1
        elif y.ndim == 2:
            assert target_dim == y.shape[1]
        else:
            raise RuntimeError("Dimensions of y does not match target_dim.")
        y_tr = y[torch.tensor(train_idx)]
        y_vl = y[torch.tensor(val_idx)]

        #
        # Load data and info for graphs/sets
        #
        data_info = {}

        if args.input_type == InputType.graph:
            #
            # Load graph-specific info
            #
            data_info['are_onehot'] = dataset['are_onehot']

            #
            # Load adjacency matrices
            #
            loaded_adjmats = dataset['adjmats']

            # Convert to dense matrices
            if args.load_sparse:
                adjmats = [aa.todense() for aa in loaded_adjmats]
            else:
                adjmats = loaded_adjmats

            #
            # Load X
            # Standard WL-approach: featureless implies use a constant vertex attribute, for every vertex
            # (such data could be generated here rather than loaded, but this coding structure easily
            # lends itself to future extensions)
            #
            if args.model_type == ModelType.rpgin and args.model_cfg.gin_model_type == GinModelType.regularGin:
                # X_all = torch.load(os.path.join(base_dir, 'Synthetic_Data', 'X_unity_Kary_Deterministic_Graphs.pt'))
                x_list = PickleUtil.read(SYNTHETIC_X_UNITY)
            elif args.model_type == ModelType.rpgin and args.model_cfg.gin_model_type == GinModelType.dataAugGin:
                x_list = PickleUtil.read(SYNTHETIC_X_EYE)
            elif isinstance(args.model_type, ModelType):
                assert data_info['are_onehot'], "Always use onehot-id with rpGin."
                #
                # Set the dimension of the one hot id
                #   (redefine it if the user makes it too big)
                largest_adjmat = np.max([adjmat.shape[0] for adjmat in adjmats])
                if args.model_cfg.onehot_id_dim > largest_adjmat:
                    onehot_id_dim = largest_adjmat
                else:
                    onehot_id_dim = args.model_cfg.onehot_id_dim
                #
                # Construct onehot ids
                #
                x_list = []
                for mat in adjmats:
                    x_list.append(construct_onehot_ids(mat.shape[0], onehot_id_dim))
            else:
                raise RuntimeError("Unrecognized ModelType!")

            # Split the data into training and validation,
            # with formats depending on the model class we will use
            adj_tr = np.array(itemgetter(*train_idx)(adjmats))
            adj_vl = np.array(itemgetter(*val_idx)(adjmats))
            x_tr = torch.stack(itemgetter(*train_idx)(x_list))
            x_vl = torch.stack(itemgetter(*val_idx)(x_list))

            # always use tensor adjacency matrices
            adj_tr = torch.from_numpy(adj_tr).to(torch.float32)
            adj_vl = torch.from_numpy(adj_vl).to(torch.float32)

            # sanity check of adjmats and X
            assert adj_tr is not None and adj_vl is not None
            assert adj_tr.ndim == 3 and adj_vl.ndim == 3
            assert x_tr.shape[0] == adj_tr.shape[0]

            # Wrap adjacency matrices and onehot-ids into data tuples
            data_tr = (adj_tr, x_tr)
            data_vl = (adj_vl, x_vl)

        elif args.input_type == InputType.set:
            raise NotImplementedError
            data_tr = (None, )
            data_vl = (None, )

        else:
            raise RuntimeError("Unrecognized InputType!")

        # infer real batch size used in training
        train_size = data_tr[0].shape[0]
        if not args.use_mini_batching or args.batch_size >= train_size:
            batch_size = train_size
            use_mini_batching = False
        else:
            batch_size = args.batch_size
            use_mini_batching = args.use_mini_batching

        return data_tr, y_tr, data_vl, y_vl, data_info, target_dim, use_mini_batching, batch_size

    @classmethod
    def get_model(cls, args: Config, data_tr, data_info, target_dim, logger_stats):

        log_stats(logger_stats, "Building model...")

        #
        # Initialize the regularizer
        #
        regularizer = JpRegularizer(args.regularization, args.input_type, total_permutations=args.num_reg_permutations,
                                    strength=args.regularization_strength, eps=args.regularization_eps,
                                    penalize_prediction=args.penalize_prediction, tangent_prop=args.tangent_prop,
                                    tp_config=args.tp_config, fixed_edge=args.fixed_edge)

        log_stats(logger_stats, "Regularization Info",
                  r=args.regularization.name, input_type=args.input_type,
                  strength=regularizer.strength, eps=regularizer.eps,
                  penalize_prediction=regularizer.penalize_prediction,
                  tangent_prop=regularizer.tangent_prop, tp_config=args.tp_config, fixed_edge=args.fixed_edge)

        #
        # Initialize the model
        #
        isinstance(args.model_type, ModelType), "Unrecognized ModelType!"

        if args.model_type == ModelType.rpgin:

            if args.model_cfg.gin_model_type == GinModelType.rpGin:
                other_mlp_params = {'batchnorm': args.model_cfg.use_batchnorm}
                if args.model_cfg.dense_dropout_prob > 0.:
                    other_mlp_params['dropout'] = args.model_cfg.dense_dropout_prob
                eps_tunable = not args.model_cfg.set_epsilon_zero

                model = GinMultiGraph(adjmat_list=data_tr[0], input_data_dim=data_tr[1].shape[-1],
                              num_agg_steps=args.model_cfg.num_gnn_layers, vertex_embed_dim=args.model_cfg.vertex_embed_dim,
                              mlp_num_hidden=args.model_cfg.num_mlp_hidden, mlp_hidden_dim=args.model_cfg.mlp_hidden_dim,
                              mlp_act=args.model_cfg.act, vertices_are_onehot=data_info['are_onehot'], target_dim=target_dim,
                              logger=logger_stats, epsilon_tunable=eps_tunable, other_mlp_parameters=other_mlp_params)
                model.init_model_gs(permute_for_pi_sgd=args.permute_for_pi_sgd, regularizer=regularizer,
                                    scaling=args.scaling, task_type=args.task_type)
            else:
                raise NotImplementedError("We need to revisit non rpGIN models.")

        elif args.model_type == ModelType.transformer:
            raise NotImplementedError
            model = None

        elif args.model_type == ModelType.lstm:
            raise NotImplementedError
            model = None

        else:
            raise NotImplementedError(f"Haven't implement model {args.model_type.name}")

        return tu.move(model)

    @classmethod
    def load_state(cls, cfg: Config, model, best_model, optimizer, logger_stats):
        checkpoint = cfg.checkpoint_file_name()
        if os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint)
            model.load_state_dict(state_dict['model'])
            best_model.load_state_dict(state_dict['best_model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            best_val_metric = state_dict['best_val_metric']
            patience = state_dict['patience']
            patience_increase = state_dict['patience_increase']
            epoch = state_dict['epoch']
            best_epoch = state_dict['best_epoch']
            training_time = state_dict['training_time']
            log_stats(logger_stats, "restarting_optimization",
                        best_val_metric=best_val_metric, patience=patience,
                        patience_increase=patience_increase, epoch=epoch, training_time=training_time)
        else:
            best_val_metric = np.inf
            patience = cfg.num_epochs  # look as these many epochs
            patience_increase = cfg.patience_increase  # wait these many epochs longer once validation error stops reducing
            epoch = -1
            best_epoch = -1
            training_time = 0
            log_stats(logger_stats, "starting_optimization",
                        best_val_metric=best_val_metric, patience=patience,
                        patience_increase=patience_increase, epoch=epoch, training_time=0)
        return model, best_model, optimizer, best_val_metric, patience, patience_increase, epoch, best_epoch, training_time

    @classmethod
    def update_batch_stats_backward(cls, batch_stats, loss, penalty, latent_norm, mb_bs, epoch, backward):
        cls.check_nan(loss, epoch)
        assert mb_bs > 0
        batch_stats['batch_loss'] += torch.div(loss.detach(), mb_bs).item()
        batch_stats['batch_penalty'] += torch.div(penalty.detach(), mb_bs).item()
        batch_stats['batch_latent_norm'] += torch.div(latent_norm.detach(), mb_bs).item()
        if backward:
            torch.div(loss, mb_bs).backward()

    @classmethod
    def forward_batch(cls, cfg: Config, model: ModelGS, data, y, epoch=None, return_pred=False):
        """
        Forward a mini-batch / full batch. Chop into small batches when necessary wrt batch limit.
        Get predictions, loss, stats, and optionally backpropagate for the model.
        When data are graphs, this model permutes the adjacency matrix by the definition of regularization.
        """
        batch_size = data[0].shape[0]
        batch_stats = Util.dictize(batch_loss=0., batch_penalty=0., batch_acc=0., batch_latent_norm=0., nfp=-1)
        batch_preds = []

        # Computational batches: split current minibatch into smaller batches for the sake of memory
        # We do NOT update parameters for each computational batch, so it's not the same as minibatch SGD
        for mb_sid in Util.make_batches(batch_size, cfg.batch_limit, use_tqdm=False):
            pred, loss_train, penalty_train, latent_norm = model.main(tuple(dd[mb_sid] for dd in data),
                                                                      y[mb_sid], sum_reduce=True)
            cls.update_batch_stats_backward(batch_stats, loss_train, penalty_train, latent_norm,
                                            batch_size, epoch, model.training)
            batch_preds.append(pred)

        batch_pred = torch.cat(batch_preds)
        # Accuracy
        if cfg.task_type in (TaskType.multi_classification, TaskType.binary_classification):
            batch_stats['batch_acc'] = accuracy(batch_pred.detach(), tu.move(y))
        else:
            batch_stats['batch_acc'] = -1
        # Return predictions for proper inference.
        if return_pred:
            batch_stats['batch_pred'] = batch_pred
        return batch_stats

    @classmethod
    def log_data_info(cls, cfg, logger_stats, data_tr, y_tr, data_vl, y_vl, data_info, use_mini_batching, batch_size):

        log_stats(logger_stats, "Data loaded", path=cfg.data_path)

        log_stats(logger_stats, "Basic info", model_type=cfg.model_type.name, data_type=cfg.input_type.name,
                  ExperimentData=cfg.data_cfg.experiment_data.name, permutation_sampled_random_seed=cfg.sample_random_state)

        # Log information about the onehot-id dimension, if applicable
        if cfg.input_type == InputType.graph:
            assert 'are_onehot' in data_info, "Onehot-id info must be included in data_info of graph data."
            if data_info['are_onehot']:
                onehot_id_dim = data_tr[1].shape[-1]
                if cfg.model_cfg.onehot_id_dim > onehot_id_dim:
                    log_stats(logger_stats, f"Selected value of onehot-id-dim, {cfg.onehot_id_dim}, is larger than the largest graph")
                    log_stats(logger_stats, f"onehot-id-dim has been reset to {onehot_id_dim}, the largest adjmat")
                else:
                    log_stats(logger_stats, f"onehot-id-dim is {onehot_id_dim}")

        # Log information about target distributions
        if cfg.task_type in (TaskType.multi_classification, TaskType.binary_classification):
            log_stats(logger_stats, "Class distributions",
                      train=p_freq_table(y_tr),
                      test=p_freq_table(y_vl))

        # Log information about data dimensionality
        log_stats(logger_stats, "data_shape",
                  data_tr_shape=[dat.shape for dat in data_tr],
                  data_vl_shape=[dat.shape for dat in data_vl],
                  y_tr_shape=y_tr.shape,
                  y_vl_shape=y_vl.shape)

        # Log information about batches
        if cfg.use_mini_batching and not use_mini_batching:
            log_stats(logger_stats, f"Selected value of batch_size, {cfg.batch_size}, is larger than or equal to the size of training set")
            log_stats(logger_stats, f"use_mini_batching has been reset to False")
            log_stats(logger_stats, f"batch_size has been reset to {batch_size}, the size of training set")

        batching_scheme = "Mini-batches" if use_mini_batching else "Full batch"
        log_stats(logger_stats, f"{batching_scheme} will be used", batch_size=batch_size,
                  drop_last=cfg.drop_last, shuffle_batches=cfg.shuffle_batches)
        if cfg.batch_limit < batch_size:
            log_stats(logger_stats, f"Training batches will be {'further ' if use_mini_batching else ''}split for computation.",
                      batch_limit=cfg.batch_limit)
        if cfg.batch_limit < data_vl[0].shape[0]:
            log_stats(logger_stats, "Validation data will be split for computation.", batch_limit=cfg.batch_limit)

    @classmethod
    def log_setup_info(cls, cfg, logger_stats):
        # Log information about the running session.
        session_info = {"start time": time.strftime("%Y-%m-%d %H:%M"), "cuda is_available()": torch.cuda.is_available(),
                        "Host": os.environ.get("SLURM_JOB_NODELIST"), 'Number of CPU Threads': torch.get_num_threads()}
        log_stats(logger_stats, "Training Session Info", **session_info)

        # Log information about the current branch and commit (i.e., the current HEAD).
        if GitInfo.is_git_directory():
            git_info = {"current branch name": GitInfo.get_current_branch_name(),
                        "current commit hash": GitInfo.get_current_commit_hash()}
            log_stats(logger_stats, "Git Info", **git_info)
        else:
            log_stats(logger_stats, "Git Info: not available")

        # Log information about our strategy to make code faster
        log_stats(logger_stats, "Code optimization and data type info", load_sparse=cfg.load_sparse)

        # Log information about general training strategies: pi-sgd, scaling, etc.
        if cfg.model_type != ModelType.rpgin or cfg.model_cfg.gin_model_type == GinModelType.rpGin:
            log_stats(logger_stats, "Permuting for pi-sgd:", permute_for_pi_sgd=cfg.permute_for_pi_sgd)

        log_stats(logger_stats, cfg.scaling)

    @classmethod
    def main(cls):
        cfg = Config()

        # Load data and construct cross-validation splits
        data_tr, y_tr, data_vl, y_vl, data_info, target_dim, use_mini_batching, batch_size = cls.load_data(cfg)

        full_seq = np.arange(data_tr[0].shape[0])  # indices used for shuffling

        # Initialize loggers
        logger_stats = JsonDump(cfg.log_file_name())
        epoch_stats = JsonDump(cfg.stats_file_name())
        output_stats = JsonDump(cfg.output_file_name())

        # Log information about training setups
        cls.log_setup_info(cfg, logger_stats)

        # Log information about dataset and batching
        cls.log_data_info(cfg, logger_stats, data_tr, y_tr, data_vl, y_vl, data_info, use_mini_batching, batch_size)

        # Set seeds
        if cfg.seed_val != -1:
            torch.manual_seed(cfg.seed_val)
            np.random.seed(cfg.seed_val)  # random permutations may be generated with numpy or scipy
            log_stats(logger_stats, "Random seed info", seed=cfg.seed_val)
        else:
            log_stats(logger_stats, "Random seed info: random seeds have NOT been set.")

        # Initialize model
        model = cls.get_model(cfg, data_tr, data_info, target_dim, logger_stats)
        best_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        model, best_model, optimizer, best_val_metric, patience, patience_increase \
            , epoch, best_epoch, training_time = cls.load_state(cfg, model, best_model, optimizer, logger_stats)

        # If models are Pi-SGD compatible
        pi_sgd_compatible = cfg.model_type != ModelType.rpgin or cfg.model_cfg.gin_model_type == GinModelType.rpGin

        _es = {}
        log_stats(logger_stats, "---------Training Model---------", model=model, optimizer=optimizer)
        while patience - epoch > 1:
            # start from current value of epoch and run till patience
            # If during this period, patience increases, the outer loop takes care of it

            # Update epoch here, or else the last value of epoch from the previous for loop will be duplicated.
            epoch = epoch + 1
            for epoch in tqdm(range(epoch, patience), desc="Epochs"):
                start_time = time.time()
                model.train()

                # Random shuffle of the input
                if cfg.shuffle_batches:
                    np.random.shuffle(full_seq)

                # Initialize train stats (running average of mini-batches)
                epoch_avg_train_stats = Util.dictize(loss_train=WeightedRunningAverage(),
                                                     penalty_train=WeightedRunningAverage(),
                                                     acc_train=WeightedRunningAverage(),
                                                     latent_norm_train=WeightedRunningAverage(),
                                                     forward_passes=0)

                # TODO: permute all graphs before forward() for efficiency
                # Prediction and loss
                if pi_sgd_compatible:
                    for batch_seq_id in Util.make_batches(full_seq, batch_size, cut=cfg.drop_last, use_tqdm=False):
                        optimizer.zero_grad()
                        # Forward, loss, backward()
                        #   The next function does the usual three steps, but we have different classes optimized for
                        #   importance sampling (IS) and the generic RP
                        batch_train_stats = cls.forward_batch(cfg, model, tuple(dd[batch_seq_id] for dd in data_tr),
                                                              y_tr[batch_seq_id], epoch)
                        cls.update_train_stats(epoch_avg_train_stats, batch_train_stats, len(batch_seq_id))
                        optimizer.step()
                else:
                    # TODO: Revisit non RP-GIN models.  Probably just need to do some passing onto GPU, reshaping
                    #       then call model directly.
                    raise NotImplementedError("We need to revisit non rpGIN models.")

                epoch_time = time.time() - start_time
                training_time += epoch_time

                # Evaluate model.
                model.eval()
                with torch.no_grad():
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # get accuracy and print predictions
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # This computes training/validation accuracy for the model AFTER it's updated
                    acc_val, loss_val = cls.eval(cfg, data_vl, y_vl, model, epoch)
                    if cfg.eval_train:
                        train_acc_eval_time, loss_train_eval_time = cls.eval(cfg, data_tr, y_tr, model, epoch)
                    #
                    # Evaluate RP-GIN thoroughly every cfg.eval_interval
                    #
                    inf_train_acc, inf_train_loss, inf_val_acc, inf_val_loss = None, None, None, None
                    var_pred_mean, var_pred_max, var_latent_mean, var_latent_max = None, None, None, None
                    if pi_sgd_compatible and (epoch + 1) % cfg.eval_interval == 0:
                        # Proper inference using more permutations
                        # TODO: Shall we enforce num_perm to be 1 when not permuting for pi-sgd?
                        if cfg.permute_for_pi_sgd:
                            inf_train_acc, inf_train_loss = cls.eval(cfg, data_tr, y_tr, model, epoch,
                                                                     n_perm=cfg.num_inf_perm)
                            inf_val_acc, inf_val_loss = cls.eval(cfg, data_vl, y_vl, model, epoch,
                                                                 n_perm=cfg.num_inf_perm)

                        # Test variability over epochs.
                        if cfg.test_variability:
                            # It should be sufficient to look at the validation set
                            var_pred_mean, var_pred_max, var_latent_mean, var_latent_max = \
                                model.mc_variance(tuple(dd[:cfg.batch_limit] for dd in data_vl))

                #
                # Early stopping using error
                #
                if cfg.task_type in (TaskType.multi_classification, TaskType.binary_classification):
                    val_metric = 1 - acc_val
                else:
                    val_metric = loss_val
                # can val_metric based on k-permutation inference?
                if val_metric < best_val_metric:
                    patience = max(patience, epoch + patience_increase + 1)
                    best_val_metric = val_metric
                    best_epoch = epoch

                    # Save Weights
                    best_model = copy.deepcopy(model)
                    log_stats(logger_stats, "saving_model_checkpoint", epoch=epoch)

                _es = Util.dictize(epoch=epoch
                                   , loss_val=loss_val
                                   , acc_val=acc_val
                                   , best_val_metric=best_val_metric
                                   , training_time=training_time
                                   , inf_train_acc=inf_train_acc
                                   , inf_val_acc = inf_val_acc
                                   , inf_train_loss = inf_train_loss
                                   , inf_val_loss = inf_val_loss
                                   , best_epoch = best_epoch)

                # Update train stats at eval time, if any
                if cfg.eval_train:
                    _es['loss_train_eval_time'] = loss_train_eval_time
                    _es['train_acc_eval_time'] = train_acc_eval_time

                # Update train stats (running average of mini-batches)
                for k, v in epoch_avg_train_stats.items():
                    if isinstance(v, WeightedRunningAverage):
                        epoch_avg_train_stats[k] = v()
                # loss_train for regularized models includes penalty_train
                # loss_train should be intrepreted as the avg batch loss_train during an epoch, if mini-batching is used
                epoch_avg_train_stats['loss_train'] -= epoch_avg_train_stats['penalty_train']
                epoch_avg_train_stats['loss_train'] -= epoch_avg_train_stats['latent_norm_train']
                _es.update(epoch_avg_train_stats)

                if cfg.test_variability:
                    _es.update(Util.dictize(var_pred_mean=var_pred_mean,
                                            var_pred_max=var_pred_max,
                                            var_latent_mean=var_latent_mean,
                                            var_latent_max=var_latent_max))

                epoch_stats.add(**_es)

                # Save state at the end of epoch, save with epoch+1 to avoid redoing an epoch
                torch.save(Util.dictize(
                    model=model.state_dict(), best_model=best_model.state_dict(), optimizer=optimizer.state_dict(),
                    best_val_metric=best_val_metric, patience=patience, patience_increase=patience_increase,
                    epoch=epoch, best_epoch=best_epoch, training_time=training_time
                ), cfg.checkpoint_file_name())

        # Final evaluation by more-permutation inference
        model.eval()
        best_model.eval()
        with torch.no_grad():
            # Set number of permutations for inference
            if pi_sgd_compatible:
                if cfg.permute_for_pi_sgd:
                    num_inf_perms = cfg.num_inf_perm
                else:
                    num_inf_perms = 1  # use one-perm evaluation when not permuting for pi-sgd
            else:
                raise NotImplementedError("Need to revisit models other than rpGin")
            #
            inf_train_acc, inf_train_loss = cls.eval(cfg, data_tr, y_tr, model, epoch, n_perm=num_inf_perms)
            inf_val_acc, inf_val_loss = cls.eval(cfg, data_vl, y_vl, model, epoch, n_perm=num_inf_perms)
            best_inf_train_acc, best_inf_train_loss = cls.eval(cfg, data_tr, y_tr, best_model, epoch, n_perm=num_inf_perms)
            best_inf_val_acc, best_inf_val_loss = cls.eval(cfg, data_vl, y_vl, best_model, epoch, n_perm=num_inf_perms)

            _es_update = Util.dictize(inf_train_acc=inf_train_acc
                         , inf_val_acc = inf_val_acc
                         , inf_train_loss = inf_train_loss
                         , inf_val_loss = inf_val_loss
                         , best_inf_train_acc = best_inf_train_acc
                         , best_inf_val_acc = best_inf_val_acc
                         , best_inf_train_loss = best_inf_train_loss
                         , best_inf_val_loss = best_inf_val_loss
                         , final_accuracy = best_inf_val_acc
                         , epoch = epoch
                         , best_epoch=best_epoch)
            _es.update(_es_update)

            # Quantify variability in f-arrow
            if cfg.test_variability:
                # It should be sufficient to look at the validation set
                var_pred_mean, var_pred_max, var_latent_mean, var_latent_max = \
                    model.mc_variance(tuple(dd[:cfg.batch_limit] for dd in data_vl))
                _es.update(Util.dictize(var_pred_mean=var_pred_mean,
                                        var_pred_max=var_pred_max,
                                        var_latent_mean=var_latent_mean,
                                        var_latent_max=var_latent_max))

            output_stats.add(**_es)
            log_stats(logger_stats, "final_training_status", **_es)
        log_stats(logger_stats, "Done")

    @classmethod
    def eval(cls, cfg, data, y, model, epoch=None, n_perm=1, force_eval_style=False):
        """
        Evaluate the model using either multi-permutations (proper inference) or just one permutation
        :param force_eval_style: If true, return loss by loss_func(pred, y) directly, which could be different from
                                 the loss returned by the model when training on penalty only and evaluating with one perm
        """
        assert isinstance(n_perm, int) and n_perm >= 1
        # To prevent numerical overflow:
        # Don't sum then divide, divide as we go
        if n_perm > 1 or force_eval_style:
            divisor = 1. / n_perm
            for iii in range(n_perm):
                eval_stats = cls.forward_batch(cfg, model, data, y, epoch, return_pred=True)
                pred = eval_stats['batch_pred']
                if iii == 0:
                    final_pred = divisor * pred
                else:
                    final_pred += divisor * pred

            loss = model.get_loss(final_pred, tu.move(y))
            loss = loss.mean().item()

            if cfg.task_type in (TaskType.multi_classification, TaskType.binary_classification):
                acc_val = accuracy(final_pred.detach(), y)
            else:
                acc_val = -1
        else:
            eval_stats = cls.forward_batch(cfg, model, data, y, epoch)
            acc_val, loss = eval_stats['batch_acc'], eval_stats['batch_loss']

        return acc_val, loss

    @staticmethod
    def check_nan(scalar_loss_tensor, epo):
        if torch.isnan(scalar_loss_tensor):
            raise RuntimeError("nan loss encountered" + ("" if epo is None else f" at epoch {epo}"))

    @staticmethod
    def update_train_stats(epoch_stats, batch_stats, batch_size):
        epoch_stats['loss_train'].update(batch_stats['batch_loss'], batch_size)
        epoch_stats['penalty_train'].update(batch_stats['batch_penalty'], batch_size)
        epoch_stats['acc_train'].update(batch_stats['batch_acc'], batch_size)
        epoch_stats['latent_norm_train'].update(batch_stats['batch_latent_norm'], batch_size)
        epoch_stats['forward_passes'] += batch_stats['nfp']


if __name__ == '__main__':
    Train.main()
