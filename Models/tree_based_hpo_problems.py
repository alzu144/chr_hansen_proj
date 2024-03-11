# !usr/bin/env python
# Created by "Alexander Zubov", 21/11/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

from mealpy.utils.problem import Problem
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from permetrics.regression import RegressionMetric


class RFReg_HPOProblem(Problem):
    """
    A custom hyperparameter optimization problem for use with the mealpy package.
    """
    def __init__(self, bounds=None, minmax="min", data=None, perf_weights=(0.2, 0.5, 0.3), **kwargs):
        self.data = data
        self.perf_weights = perf_weights
        self.metric_class = RegressionMetric
        super().__init__(bounds, minmax, **kwargs)
    
    def obj_func(self, x):
        # decode solution to obtain the actual hyperparameter values
        x_decoded = self.decode_solution(x)
        n_estimators = x_decoded["n_estimators"]
        min_samples_split = x_decoded["min_samples_split"]
        min_samples_leaf = x_decoded["min_samples_leaf"]
        max_samples = x_decoded["max_samples"]
        max_depth = x_decoded["max_depth"]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,                  # <--- can change
            criterion="squared_error",
            max_depth=max_depth,                        # <--- can change
            min_samples_split=min_samples_split,        # <--- can change
            min_samples_leaf=min_samples_leaf,          # <--- can change
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            ccp_alpha=0.0,
            max_samples=max_samples                     # <--- can change
        )

        # train and test model
        rf.fit(self.data["X_train"], self.data["y_train"])
        y_test_hat = rf.predict(self.data["X_test"])

        # define objective value as linear combination of different performance metrics
        # (based on my custom fitness function)
        evaluator = self.metric_class(self.data["y_test"], y_test_hat)
        obj_dict = evaluator.get_metrics_by_list_names(list_metric_names=["RMSE", "PCC", "EVS"])
        w1, w2, w3 = self.perf_weights
        obj = w1*obj_dict.get("RMSE") + w2*1/obj_dict.get("PCC") + w3*1/obj_dict.get("EVS")

        return obj