"""The script for training offline cost models"""
import time
import logging
import random
import argparse
import multiprocessing

import numpy as np
import xgboost as xgb

import tvm
from tvm import ansor
from tvm.ansor.cost_model.xgb_model import custom_callback, pack_sum_square_error, \
    pack_sum_rmse, pack_sum_average_recall_score, pack_sum_xgbmatrix, pack_sum_average_peak_score, \
    dmatrix_context, max_curve

from common import measure_schedule
import tune_op_subgraph

def top_k(arr, k):
    """Return top k elements in an array"""
    return arr[np.argpartition(arr, -k)[-k:]]

def get_normalized_feature(filename, n_lines):
    tic = time.time()
    features, normalized_throughputs, task_ids = \
        ansor.feature.get_per_stmt_features_from_file(filename, n_lines)
    print("Feature extraction time: %.2f" % (time.time() - tic))

    return np.array(features), normalized_throughputs, task_ids

def train_standalone_xgb(log_file, n_lines, split_scheme, load_model, model_name):
    log_interval = 30
    split_train_ratio = 0.8
    plan_size = 64

    # Extract features from file
    xs, ys, gids = get_normalized_feature(log_file, n_lines)

    if split_scheme == 'random':
        # Randomly split the data set into training set and test set
        idx = np.random.permutation(len(xs))
        split = int(split_train_ratio * len(idx))
        x_train, x_test = xs[idx[:split]], xs[idx[split:]]
        y_train, y_test = ys[idx[:split]], ys[idx[split:]]
        gid_train, gid_test = gids[idx[:split]], gids[idx[split:]]
    elif split_scheme == 'wkl':
        # Split the data set into training set and test set according to workloads
        n_wkls = np.max(gids) + 1
        perm = np.random.permutation(n_wkls)
        gids = perm[gids]
        split = int(split_train_ratio * n_wkls)
        train_idx, test_idx = gids <= split, gids > split
        x_train, x_test = xs[train_idx], xs[test_idx]
        y_train, y_test = ys[train_idx], ys[test_idx]
        gid_train, gid_test = gids[train_idx], gids[test_idx] - (split + 1)
        print("#Train wkl: %d\t#Test wkl: %d" % (split, n_wkls - split))
    else:
        raise ValueError("Invalid split scheme: " + split_scheme)

    # Process input data to xgb format
    dtrain = pack_sum_xgbmatrix(x_train, y_train, gids=gid_train, weights=y_train)
    dtest = pack_sum_xgbmatrix(x_test, y_test, gids=gid_test, weights=y_test)

    print("x_train shape", x_train.shape)
    print("x_test shape", x_test.shape)
    print("dtrain shape", (dtrain.num_row(), dtrain.num_col()))
    print("dtest shape", (dtest.num_row(), dtest.num_col()))

    # Set hyper parameters
    xgb_params = {
	'max_depth': 10,
	'gamma': 0.001,
	'min_child_weight': 0,
	'eta': 0.2,
	# todo(lmzheng): automatically decrease learning rate when the loss is too large

	'n_gpus': 0,
	'nthread': multiprocessing.cpu_count() // 2,
	'verbosity': 0,
	'seed': 43,
	'disable_default_eval_metric': 1,
    }

    if load_model:
        # load pre-trained model
        bst = xgb.Booster(xgb_params)
        bst.load_model(model_name)
        scores = list(bst.get_fscore().items())
        scores.sort(key=lambda x: x[1], reverse=True)
        feature_names = ansor.feature.get_per_stmt_feature_names()
        scores = [[feature_names[int(x[1:])], y] for x, y in scores]
        print("Feature scores:")
        print(scores)
    else:
        # train a new model
        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=10000,
            obj=pack_sum_square_error,
            callbacks=[custom_callback(
                stopping_rounds=100,
                metric='tr-p-rmse',
                fevals=[pack_sum_rmse, pack_sum_average_peak_score(plan_size)],
                evals=[(dtrain, 'tr'), (dtest, 'te')],
                maximize=False,
                verbose_eval=log_interval)])
        bst.save_model(model_name)

    eval_res = bst.eval_set([(dtrain, 'train'), (dtest, 'test')], 0,
                            pack_sum_average_peak_score(plan_size))
    eval_res = eval_res.replace('[0]', '')
    print("Eval results: %s" % eval_res)


def train_xgb_model(log_file, n_lines, load_model, model_name):
    # Extract features from file
    inputs, results = ansor.LogReader(log_file).read_lines(n_lines)

    # Train the model
    model = ansor.XGBModel()
    if load_model:
        model.load(model_name)
        print("Loaded.")
    else:
        print("========== Train a New Model ==========")
        model.update(inputs, results)
        model.save(model_name)
        print("Saved.")

    # Run evaluation for one test workload
    print("========== Run Evaluation ==========")
    test_wkl_key = inputs[0].task.workload_key
    print("Run evaluation for workload %s" % test_wkl_key)

    # Construct test set from the log file
    plan_size = model.plan_size
    test_inputs = []
    test_results = []
    test_ys = []
    for inp, res in zip(inputs, results):
        if inp.task.workload_key == test_wkl_key:
            test_inputs.append(inp)
            test_results.append(res)
            test_ys.append(ansor.utils.array_mean(res.costs))

    # Rebuild the test states and test search task
    task = test_inputs[0].task
    task = ansor.SearchTask(ansor.workload_key_to_dag(task.workload_key),
            task.workload_key, task.target, task.target_host)
    test_states = ansor.serialization.get_states_from_measure_inputs(test_inputs, task)

    # Make prediction
    preds = model.predict(task, test_states)
    test_ys = np.min(test_ys) / test_ys

    # Compute average peak score
    trials = np.argsort(preds)[::-1][:model.plan_size]
    trial_scores = test_ys[trials]
    curve = max_curve(trial_scores) / np.max(test_ys)
    score = np.mean(curve)
    print("Evaluation: a-peak@%d = %.4f" % (model.plan_size, score))

    # Remeasure to check against the saved records
    print("========== Check against Saved Log File ==========")
    indices = np.argsort(-preds)[:5]
    flop = task.compute_dag.flop_ct
    min_cost = np.min(test_ys)
    for idx in indices:
        inp, res = test_inputs[idx], test_results[idx]
        s, bufs = task.compute_dag.apply_steps_from_state(inp.state)
        costs = measure_schedule(s, bufs, task.target)
        print("No: %6d\tPredicted score: %.2f\tPredicted GFLOPS: %.2f\tRecorded GFLOPS: %.2f\tRemeasured GFLOPS: %.2f" %
              (idx, preds[idx], preds[idx] * (flop / min_cost) / 1e9,
               flop / ansor.utils.array_mean(res.costs) / 1e9, flop / np.mean(costs) / 1e9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str)
    parser.add_argument("--n-lines", type=int, default=-1,
        help='Only use the first `n-lines` lines of the log file`. (-1 means use all lines)')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-test", action='store_true',
        help="If true, split the dataset into training set and test set")
    parser.add_argument("--load-model", action='store_true')
    parser.add_argument("--model-name", type=str, default='saved_model.xgb')
    parser.add_argument("--split-scheme", type=str, choices=['random', 'wkl'], default='random')
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger('ansor').setLevel(logging.DEBUG)

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.eval_test:
        train_standalone_xgb(args.log_file, n_lines=args.n_lines, split_scheme=args.split_scheme,
                             load_model=args.load_model, model_name=args.model_name)
    else:
        train_xgb_model(args.log_file, n_lines=args.n_lines,
                        load_model=args.load_model, model_name=args.model_name)

