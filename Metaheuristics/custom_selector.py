# !usr/bin/env python
# Created by "Alexander Zubov", 01/10/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import numpy as np
from mafese.utils.mealpy_util import FeatureSelectionProblem
from mafese.utils.data_loader import Data
from mafese.wrapper.mha import MhaSelector
from permetrics.regression import RegressionMetric
from permetrics.classification import ClassificationMetric

from .custom_fitness_fun import FeatureSelectionProblem_custom


class MhaSelector_custom(MhaSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self, X_train, y_train, X_test, y_test, fit_weights=(0.9, 0.1), verbose=True, mode='single', n_workers=None, termination=None):
        """
        A customized version to incorporate manual train-test splitting.

        Parameters
        ----------
        X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples, for training the estimator.

        y_train : array-like of shape (n_samples,)
            The training target values, for training the estimator.
        
        X_test : {array-like, sparse matrix} of shape (n_samples, n_features)
            The testing input samples, for evaluating the fitness of the estimator.

        y_test : array-like of shape (n_samples,)
            The testing target values, for evaluating the fitness of the estimator.

        fit_weights : list, tuple or np.ndarray, default = (0.9, 0.1)
            The first weight is for objective value and the second weight is for the number of features

        verbose : int, default = True
            Controls verbosity of output.

        mode : str, default = 'single'
            The mode used in Optimizer belongs to Mealpy library. Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                - 'process': The parallel mode with multiple cores run the tasks
                - 'thread': The parallel mode with multiple threads run the tasks
                - 'swarm': The sequential mode that no effect on updating phase of other agents
                - 'single': The sequential mode that effect on updating phase of other agents, default

        n_workers : int or None, default = None
            The number of workers (cores or threads) to do the tasks (effect only on parallel mode)

        termination : dict or None, default = None
            The termination dictionary or an instance of Termination class. It is for Optimizer belongs to Mealpy library.
        """
        self.data = Data(X_train.copy(), y_train.copy())
        self.data.set_train_test(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
        lb = [-8, ] * X_train.shape[1]
        ub = [8, ] * X_train.shape[1]
        if self.problem == "classification":
            if len(np.unique(y_train)) == 2:
                self.obj_paras = {"average": "micro"}
            else:
                self.obj_paras = {"average": "weighted"}
            if self.obj_name is None:
                self.obj_name = "AS"
            else:
                self.obj_name = self._set_metric(self.obj_name, self.SUPPORT["classification_objective"])
            minmax = self.SUPPORT["classification_objective"][self.obj_name]
            metric_class = ClassificationMetric
        else:
            self.obj_paras = {"decimal": 4}
            if self.obj_name is None:
                self.obj_name = "MSE"
            else:
                self.obj_name = self._set_metric(self.obj_name, self.SUPPORT["regression_objective"])
            minmax = self.SUPPORT["regression_objective"][self.obj_name]
            metric_class = RegressionMetric
        fit_sign = -1 if minmax == "max" else 1
        log_to = "console" if verbose else "None"
        prob = FeatureSelectionProblem(lb, ub, minmax, data=self.data,
                                       estimator=self.estimator, transfer_func=self.transfer_func_, obj_name=self.obj_name,
                                       metric_class=metric_class, fit_weights=fit_weights, fit_sign=fit_sign, log_to=log_to,
                                       obj_weights=(1.0, 0.), obj_paras=self.obj_paras)
        best_position, best_fitness = self.optimizer.solve(prob, mode=mode, n_workers=n_workers, termination=termination)
        self.selected_feature_solution = np.array(best_position, dtype=int)
        self.selected_feature_masks = np.where(self.selected_feature_solution == 0, False, True)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]

class MhaSelector_custom_fitfun(MhaSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self, X_train, y_train, X_test, y_test, fit_weights=(0.9, 0.1), verbose=True, mode='single', n_workers=None, termination=None):
        """
        A customized version to incorporate manual train-test splitting and a
        special fitness function.

        Parameters
        ----------
        X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples, for training the estimator.

        y_train : array-like of shape (n_samples,)
            The training target values, for training the estimator.
        
        X_test : {array-like, sparse matrix} of shape (n_samples, n_features)
            The testing input samples, for evaluating the fitness of the estimator.

        y_test : array-like of shape (n_samples,)
            The testing target values, for evaluating the fitness of the estimator.

        fit_weights : list, tuple or np.ndarray, default = (0.9, 0.1)
            The first weight is for objective value and the second weight is for the number of features

        verbose : int, default = True
            Controls verbosity of output.

        mode : str, default = 'single'
            The mode used in Optimizer belongs to Mealpy library. Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                - 'process': The parallel mode with multiple cores run the tasks
                - 'thread': The parallel mode with multiple threads run the tasks
                - 'swarm': The sequential mode that no effect on updating phase of other agents
                - 'single': The sequential mode that effect on updating phase of other agents, default

        n_workers : int or None, default = None
            The number of workers (cores or threads) to do the tasks (effect only on parallel mode)

        termination : dict or None, default = None
            The termination dictionary or an instance of Termination class. It is for Optimizer belongs to Mealpy library.
        """
        self.data = Data(X_train.copy(), y_train.copy())
        self.data.set_train_test(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
        lb = [-8, ] * X_train.shape[1]
        ub = [8, ] * X_train.shape[1]
        if self.problem == "classification":
            if len(np.unique(y_train)) == 2:
                self.obj_paras = {"average": "micro"}
            else:
                self.obj_paras = {"average": "weighted"}
            if self.obj_name is None:
                self.obj_name = "AS"
            else:
                self.obj_name = self._set_metric(self.obj_name, self.SUPPORT["classification_objective"])
            minmax = self.SUPPORT["classification_objective"][self.obj_name]
            metric_class = ClassificationMetric
        else:
            self.obj_paras = {"decimal": 4}
            if self.obj_name is None:
                self.obj_name = "MSE"
            else:
                self.obj_name = self._set_metric(self.obj_name, self.SUPPORT["regression_objective"])
            minmax = self.SUPPORT["regression_objective"][self.obj_name]
            metric_class = RegressionMetric
        fit_sign = -1 if minmax == "max" else 1
        log_to = "console" if verbose else "None"
        prob = FeatureSelectionProblem_custom(lb, ub, minmax, data=self.data,
                                       estimator=self.estimator, transfer_func=self.transfer_func_, obj_name=self.obj_name,
                                       metric_class=metric_class, fit_weights=fit_weights, fit_sign=fit_sign, log_to=log_to,
                                       obj_weights=(1.0, 0.), obj_paras=self.obj_paras)
        best_position, best_fitness = self.optimizer.solve(prob, mode=mode, n_workers=n_workers, termination=termination)
        self.selected_feature_solution = np.array(best_position, dtype=int)
        self.selected_feature_masks = np.where(self.selected_feature_solution == 0, False, True)
        self.selected_feature_indexes = np.where(self.selected_feature_masks)[0]