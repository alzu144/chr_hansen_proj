# !usr/bin/env python
# Created by "Alexander Zubov", 01/11/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import numpy as np
from mafese.utils.mealpy_util import FeatureSelectionProblem


class FeatureSelectionProblem_custom(FeatureSelectionProblem):
    def __init__(self, lb, ub, minmax, **kwargs):
        super().__init__(lb, ub, minmax, **kwargs)
    
    def fit_func(self, solution):
        """
        A special (hard-coded) fitness function with variable weight change.
        """
        cols = np.flatnonzero(solution)
        self.estimator.fit(self.data.X_train[:, cols], self.data.y_train)
        y_valid_pred = self.estimator.predict(self.data.X_test[:, cols])
        evaluator = self.metric_class(self.data.y_test, y_valid_pred)
        obj_dict = evaluator.get_metrics_by_list_names(list_metric_names=["RMSE", "PCC", "EVS"])
        obj = 0.2*obj_dict.get("RMSE") + 0.5*1/obj_dict.get("PCC") + 0.3*1/obj_dict.get("EVS")
        feat_ratio = np.sum(solution)/self.n_dims
        w_obj = np.mean([np.mean([self.fit_weights[0], 1 - feat_ratio**1.5]), 1.0049])
        w_feat = np.mean([np.mean([self.fit_weights[0], feat_ratio**1.5]), -0.0049])
        fitness = w_obj*obj + self.fit_sign*w_feat*feat_ratio
        return [fitness, obj]