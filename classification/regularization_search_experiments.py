import load_learned_structure
from run_current_experiment import get_k_sorted_hparams
from experiment_params import ExperimentParams
import train_classifier
import numpy as np
import time


def search_reg_str_entropy(cur_assignments, kwargs):
    starting_reg_str = kwargs["reg_strength"]
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/" + kwargs["dataset"]    
    found_small_enough_reg_str = False
    # first search by checking that after 5 epochs, more than half aren't above .9
    kwargs["max_epoch"] = 1
    counter = 0
    rho_bound = .99
    while not found_small_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_structure.entropy_rhos(
            file_base + args.filename() + ".txt", rho_bound)
        print("fraction under {}: {}".format(rho_bound,frac_under_pointnine))
        print("")
        if frac_under_pointnine < .25:
            kwargs["reg_strength"] = kwargs["reg_strength"] / 2.0
            if kwargs["reg_strength"] < 10**-7:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_big_lr"
        else:
            found_small_enough_reg_str = True

    found_large_enough_reg_str = False
    kwargs["max_epoch"] = 5
    rho_bound = .9
    while not found_large_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_structure.entropy_rhos(
            file_base + args.filename() + ".txt", rho_bound)
        print("fraction under {}: {}".format(rho_bound,frac_under_pointnine))
        print("")
        if frac_under_pointnine > .25:
            kwargs["reg_strength"] = kwargs["reg_strength"] * 2.0
            if kwargs["reg_strength"] > 10**4:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_small_lr"
        else:
            found_large_enough_reg_str = True
    # to set this back to the default
    kwargs["max_epoch"] = 500
    return counter, "okay_lr"

# ways this can fail:
# too small learning rate
# too large learning rate
# too large step size for reg strength, so it's too big then too small
def search_reg_str_l1(cur_assignments, kwargs):
    # the final number of params is within this amount of target
    smallest_reg_str = 10**-9
    largest_reg_str = 10**2
    distance_from_target = 10
    starting_reg_str = kwargs["reg_strength"]
    found_good_reg_str = False
    too_small = False
    too_large = False
    counter = 0
    reg_str_growth_rate = 2.0

    while not found_good_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err = train_classifier.main(args)
        learned_d_out, num_params = load_learned_structure.l1_group_norms(args=args, prox=kwargs["prox_step"])
        
        if num_params < kwargs["reg_goal_params"] - distance_from_target:
            if too_large:
                # reduce size of steps for reg strength
                reg_str_growth_rate = (reg_str_growth_rate + 1)/2.0
                too_large = False
            too_small = True
            kwargs["reg_strength"] = kwargs["reg_strength"] / reg_str_growth_rate
            if kwargs["reg_strength"] < smallest_reg_str:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_small_lr", cur_valid_err, learned_d_out
        elif num_params > kwargs["reg_goal_params"] + distance_from_target:
            if too_small:
                # reduce size of steps for reg strength
                reg_str_growth_rate = (reg_str_growth_rate + 1)/2.0
                too_small = False
            too_large = True
            kwargs["reg_strength"] = kwargs["reg_strength"] * reg_str_growth_rate

            if kwargs["reg_strength"] > largest_reg_str:
                kwargs["reg_strength"] = starting_reg_str
                
                # it diverged, and for some reason the weights didn't drop
                if num_params == int(args.d_out) * 4 and cur_assignments["lr"] > .25 and cur_valid_err > .3:
                    return counter, "too_big_lr", cur_valid_err, learned_d_out
                else:
                    return counter, "too_small_lr", cur_valid_err, learned_d_out            
        else:
            found_good_reg_str = True
    return counter, "okay_lr", cur_valid_err, learned_d_out

def train_k_then_l_models(k,l,counter,total_evals,start_time, logging_dir, **kwargs):
    assert "reg_strength" in kwargs
    if "prox_step" not in kwargs:
        kwargs["prox_step"] = False
    elif kwargs["prox_step"]:
        assert False, "It's too unstable. books/all_cs_and_equal_rho/hparam_opt/structure_search/proximal_gradient too big then too small"
    file_base = logging_dir + kwargs["dataset"]    
    best = {
        "assignment" : None,
        "valid_err" : 1,
        "learned_pattern" : None,
        "learned_d_out" : None,
        "reg_strength": None
        }

    reg_search_counters = []
    lr_lower_bound = 7*10**-3
    lr_upper_bound = .5
    all_assignments = get_k_sorted_hparams(k, lr_lower_bound, lr_upper_bound)
    for i in range(len(all_assignments)):
        
        valid_assignment = False
        while not valid_assignment:
            cur_assignments = all_assignments[i]
            if kwargs["sparsity_type"] == "rho_entropy":
                one_search_counter, lr_judgement = search_reg_str_entropy(cur_assignments, kwargs)
            elif kwargs["sparsity_type"] == "states":
                one_search_counter, lr_judgement, cur_valid_err, learned_d_out = search_reg_str_l1(
                    cur_assignments, kwargs)
                learned_pattern = "1-gram,2-gram,3-gram,4-gram"
                
            reg_search_counters.append(one_search_counter)
            if lr_judgement == "okay_lr":
                valid_assignment = True
            else:
                if lr_judgement == "too_big_lr":
                    # lower the upper bound
                    lr_upper_bound = cur_assignments['lr']
                    reverse = True
                elif lr_judgement == "too_small_lr":
                    # rase lower bound
                    lr_lower_bound = cur_assignments['lr']
                    reverse = False
                else:
                    assert False, "shouldn't be here."
                new_assignments = get_k_sorted_hparams(k-i, lr_lower_bound, lr_upper_bound)
                if reverse:
                    new_assignments.reverse()
                all_assignments[i:len(all_assignments)] = new_assignments
                
        if kwargs["sparsity_type"] == "rho_entropy":
            args = ExperimentParams(**kwargs, **cur_assignments)
            cur_valid_err = train_classifier.main(args)
        
            learned_pattern, learned_d_out, frac_under_pointnine = load_learned_structure.l1_group_norms(
                file_base + args.filename() + ".txt", .9)
        
        if cur_valid_err < best["valid_err"]:
            best = {
                "assignment" : cur_assignments,
                "valid_err" : cur_valid_err,
                "learned_pattern" : learned_pattern,
                "learned_d_out" : learned_d_out,
                "reg_strength": kwargs["reg_strength"]
            }

        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))

    kwargs["reg_strength"] = best["reg_strength"]
    for i in range(l):
        args = ExperimentParams(filename_suffix="_{}".format(i),**kwargs, **best["assignment"])
        cur_valid_err = train_classifier.main(args)
        counter[0] = counter[0] + 1
        
    
    return best, reg_search_counters
