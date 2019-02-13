from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
import os
from experiment_params import ExperimentParams, get_categories
import train_classifier
import numpy as np
import time
import regularization_search_experiments
import experiment_tools


def main(argv):
    loaded_embedding = experiment_tools.preload_embed(os.path.join(argv.base_dir,argv.dataset))
    
    exp_num = 0

    start_time = time.time()
    counter = [0]
    categories = get_categories()
    
    
    # a basic experiment
    if exp_num == 0:
        if argv.random_selection:
            hyper_parameters_assignments = hparam_sample()
        else:
            hyper_parameters_assignments = {
                "clip_grad": argv.clip,
                "dropout": argv.dropout,
                "rnn_dropout": argv.rnn_dropout,
                "embed_dropout": argv.embed_dropout,
                "lr": argv.lr, "weight_decay": argv.weight_decay,
                "depth": argv.depth
            }

        kwargs = {"pattern": argv.pattern, "d_out": argv.d_out,
                                "learned_structure": argv.learned_structure,
                                "reg_goal_params": argv.reg_goal_params,
                                "filename_prefix": argv.filename_prefix,
                                "seed": argv.seed, "loaded_embedding": loaded_embedding,
                                "dataset": argv.dataset, "use_rho": False,
                                "gpu": argv.gpu,
                                "max_epoch": argv.max_epoch, "patience": argv.patience,
                                "batch_size": argv.batch_size, "use_last_cs": argv.use_last_cs,
                                "logging_dir": argv.logging_dir,
                                "base_data_dir": argv.base_dir, "output_dir": argv.model_save_dir,
                                "reg_strength": argv.reg_strength, "sparsity_type": argv.sparsity_type,
                                "semiring": argv.semiring}

        args = ExperimentParams(**kwargs, **hyper_parameters_assignments)

        cur_valid_err = train_classifier.main(args)



# hparams to search over (from paper):
# clip_grad, dropout, learning rate, rnn_dropout, embed_dropout, l2 regularization (actually weight decay)
def hparam_sample(lr_bounds = [1.5, 10**-3]):
    assignments = {
        "clip_grad" : np.random.uniform(1.0, 5.0),
        "dropout" : np.random.uniform(0.0, 0.5),
        "rnn_dropout" : np.random.uniform(0.0, 0.5),
        "embed_dropout" : np.random.uniform(0.0, 0.5),
        "lr" : np.exp(np.random.uniform(np.log(lr_bounds[0]), np.log(lr_bounds[1]))),
        "weight_decay" : np.exp(np.random.uniform(np.log(10**-5), np.log(10**-7))),
    }

    return assignments


def training_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("--learned_structure", help="Learned structure", type=str, default="l1-states-learned")
    p.add_argument('--reg_goal_params', type=int, default = 20)
    p.add_argument('--filename_prefix', help='logging file prefix?', type=str, default="all_cs_and_equal_rho/saving_model_for_interpretability/")
    p.add_argument("-t", "--dropout", help="Use dropout", type=float, default=0.1943)
    p.add_argument("--rnn_dropout", help="Use RNN dropout", type=float, default=0.0805)
    p.add_argument("--embed_dropout", help="Use RNN dropout", type=float, default=0.3489)
    p.add_argument("-l", "--lr", help="Learning rate", type=float, default=2.553E-02)
    p.add_argument("--clip", help="Gradient clipping", type=float, default=1.09)
    p.add_argument('-w', "--weight_decay", help="Weight decay", type=float, default=1.64E-06)
    p.add_argument("-m", "--model_save_dir", help="where to save the trained model", type=str)
    p.add_argument("--logging_dir", help="Logging directory", type=str)
    p.add_argument("--max_epoch", help="Number of iterations", type=int, default=500)
    p.add_argument("--patience", help="Patience parameter (for early stopping)", type=int, default=30)
    p.add_argument("--sparsity_type", help="Type of sparsity (wfsa, edges, states, rho_entropy or none)",
                   type=str, default="none")
    p.add_argument("--reg_strength", help="Regularization strength", type=float, default=0.0)
    p.add_argument("--semiring", help="Type of semiring (plus_times, max_times, max_plus)",
                   type=str, default="plus_times")
    p.add_argument("--random_selection", help="Randomly select hyperparameters",
                   action='store_true')
    # p.add_argument("-r", "--scheduler", help="Use reduce learning rate on plateau schedule", action='store_true')
    # p.add_argument("--debug", help="Debug", type=int, default=0)
    return p


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[experiment_tools.general_arg_parser(), training_arg_parser()])
    sys.exit(main(parser.parse_args()))
