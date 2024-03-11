# !usr/bin/env python
# Created by "Alexander Zubov", 26/09/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from .custom_selector import MhaSelector_custom, MhaSelector_custom_fitfun


def meta_fs_wrapper(
        X_train, y_train, X_test, y_test, 
        ml_algorithm, optim_algorithm, ml_problem="regression", 
        verbose=True, save_results=True, out_path=None, mha="default",
        **kwargs):
    if "selector" in kwargs:
        selector_pars = kwargs["selector"]
    else:
        selector_pars = {}
    if "ml_algorithm" in kwargs:
        ml_algo_pars = kwargs["ml_algorithm"]
    else:
        ml_algo_pars = None
    if "optimizer" in kwargs:
        optim_pars = kwargs["optimizer"]
    else:
        optim_pars = None
    if "fitting" in kwargs:
        fitting_pars = kwargs["fitting"]
    else:
        fitting_pars = {}
    
    if mha == "default":
        mha_selector = MhaSelector_custom
    elif mha == "fitfun":
        mha_selector = MhaSelector_custom_fitfun

    selector = mha_selector(
        problem=ml_problem, estimator=ml_algorithm, estimator_paras=ml_algo_pars,
        optimizer=optim_algorithm, optimizer_paras=optim_pars,
        **selector_pars
    )
    selector.fit(X_train, y_train, X_test, y_test, verbose=verbose, **fitting_pars)

    if save_results == True:
        save_optim_results(selector.optimizer, ml_algorithm, out_path)

    return selector, selector.get_best_obj_and_fit(), selector.selected_feature_solution

def save_optim_results(fitted_optimizer, ml_algorithm, out_path):
    if not out_path:
        out_path = "./"

    optim_name = fitted_optimizer.name
    ml_algo_name = type(ml_algorithm).__name__
    timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())

    file_id = "_".join([timestamp, ml_algo_name, "FS", optim_name])

    try:
        current_best_fit = np.array(fitted_optimizer.history.list_current_best_fit)
        current_best_pos = np.array([
            pos[0] for pos in fitted_optimizer.history.list_current_best
        ])
        current_best_num_feat = current_best_pos.sum(axis=1)

        global_best_fit = np.array(fitted_optimizer.history.list_global_best_fit)
        global_best_pos = np.array([
            pos[0] for pos in fitted_optimizer.history.list_global_best
        ])
        global_best_num_feat = global_best_pos.sum(axis=1)

        epoch_time = np.array(fitted_optimizer.history.list_epoch_time)

        np.savetxt(
            out_path + file_id + "_current_best_fit.txt", current_best_fit
        )
        np.savetxt(
            out_path + file_id + "_current_best_pos.txt", current_best_pos
        )
        np.savetxt(
            out_path + file_id + "_current_best_num_feat.txt", current_best_num_feat, fmt="%d"
        )
        np.savetxt(
            out_path + file_id + "_global_best_fit.txt", global_best_fit
        )
        np.savetxt(
            out_path + file_id + "_global_best_pos.txt", global_best_pos
        )
        np.savetxt(
            out_path + file_id + "_global_best_num_feat.txt", global_best_num_feat, fmt="%d"
        )
        np.savetxt(
            out_path + file_id + "_epoch_time.txt", epoch_time
        )
    except:
        with open(out_path + file_id + "_current_best_fit.pkl", "wb") as file:
            pickle.dump(fitted_optimizer.history.list_current_best_fit, file)
        
        with open(out_path + file_id + "_current_best_pos.pkl", "wb") as file:
            pickle.dump(fitted_optimizer.history.list_current_best, file)
        
        with open(out_path + file_id + "_global_best_fit.pkl", "wb") as file:
            pickle.dump(fitted_optimizer.history.list_global_best_fit, file)
        
        with open(out_path + file_id + "_global_best_pos.pkl", "wb") as file:
            pickle.dump(fitted_optimizer.history.list_global_best, file)
        
        with open(out_path + file_id + "_epoch_time.pkl", "wb") as file:
            pickle.dump(fitted_optimizer.history.list_epoch_time, file)