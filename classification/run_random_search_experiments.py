#!/usr/bin/env python


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
import os
import copy
import time
import experiment_tools
import regularization_search_experiments
from experiment_params import ExperimentParams, get_categories



def main(argv):
    loaded_embedding = experiment_tools.preload_embed(os.path.join(argv.base_dir, argv.dataset))

    training_args = {
        'pattern': experiment_tools.select_param_value('PATTERN', argv.pattern),
        'd_out': experiment_tools.select_param_value('D_OUT', argv.d_out),
        'seed': int(experiment_tools.select_param_value('SEED', argv.seed)),
        'learned_structure': experiment_tools.select_param_value('LEARNED_STRUCTURE', argv.learned_structure),
        'semiring': experiment_tools.select_param_value('SEMIRING', argv.semiring),
        "depth": argv.depth,
        "filename_prefix": argv.filename_prefix,
        "dataset": argv.dataset, "use_rho": False,
        "gpu": argv.gpu,
        "max_epoch": argv.max_epoch, "patience": argv.patience,
        "batch_size": argv.batch_size, "use_last_cs": argv.use_last_cs,
        "logging_dir": argv.logging_dir,
        "reg_strength": argv.reg_strength,
        "base_data_dir": argv.base_dir, "output_dir": argv.model_save_dir
    }

    reg_goal_params = experiment_tools.select_param_value('REG_GOAL_PARAMS', argv.reg_goal_params)

    rand_search_args = {
        "k": argv.k,
        "l": argv.l,
        "m": argv.m,
        "n": argv.n,
        "sparsity_type": argv.sparsity_type,
        "reg_goal_params_list": [int(x) for x in reg_goal_params.split(",")]
    }

    print(training_args)
    print(rand_search_args)

    training_args["loaded_embedding"] = loaded_embedding

    run_grid_search(training_args, rand_search_args)


def run_grid_search(training_args, rand_search_args):
    start_time = time.time()
    counter = [0]
    categories = get_categories()

    total_evals = len(categories) * \
                  (rand_search_args["m"] + rand_search_args["n"] + \
                   rand_search_args["k"] + rand_search_args["l"]) * \
                  len(rand_search_args["reg_goal_params_list"])

    all_reg_search_counters = []

    for reg_goal_params in rand_search_args["reg_goal_params_list"]:
        best, reg_search_counters = regularization_search_experiments.train_k_then_l_models(
            k=rand_search_args["k"], l=rand_search_args["l"],
            counter=counter, total_evals=total_evals, start_time=start_time,
            reg_goal_params=reg_goal_params,
            sparsity_type=rand_search_args["sparsity_type"],
            **training_args)

        all_reg_search_counters.append(reg_search_counters)
        training_args_copy = copy.deepcopy(training_args)

        training_args_copy["pattern"]=best['learned_pattern']
        training_args_copy["d_out"] = best['learned_d_out']
        training_args_copy["learned_structure"] = 'l1-states-learned'

        args = regularization_search_experiments.train_m_then_n_models(
            m=rand_search_args["m"], n=rand_search_args["n"], counter=counter,
            total_evals=total_evals, start_time=total_evals,
            **training_args_copy)

    print("search counters:")
    for search_counter in all_reg_search_counters:
        print(search_counter)



def training_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("--learned_structure", help="Learned structure", type=str, default="l1-states-learned")
    p.add_argument('--reg_goal_params', type=str, default="80,60,40,20")
    p.add_argument('--filename_prefix', help='logging file prefix?', type=str,
                   default="all_cs_and_equal_rho/saving_model_for_interpretability/")
    p.add_argument("-m", "--model_save_dir", help="where to save the trained model", type=str)
    p.add_argument("--logging_dir", help="Logging directory", type=str, required=True)
    p.add_argument("--max_epoch", help="Number of iterations", type=int, default=500)
    p.add_argument("--patience", help="Patience parameter (for early stopping)", type=int, default=30)
    p.add_argument("--sparsity_type", help="Type of sparsity (wfsa, edges, states, rho_entropy or none)",
                   type=str, default="states")
    p.add_argument("--reg_strength", help="Regularization strength", type=float, default=8 * 10 ** -6)
    p.add_argument("--semiring", help="Type of semiring (plus_times, max_times, max_plus)",
                   type=str, default="plus_times")
    p.add_argument("--k", help="K argument for random search", type=int, default=20)
    p.add_argument("--l", help="L argument for random search", type=int, default=5)
    p.add_argument("--m", help="M argument for random search", type=int, default=20)
    p.add_argument("--n", help="N argument for random search", type=int, default=5)

    return p


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[experiment_tools.general_arg_parser(), training_arg_parser()])
    sys.exit(main(parser.parse_args()))

