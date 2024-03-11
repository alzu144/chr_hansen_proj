# !usr/bin/env python
# Created by "Alexander Zubov", 05/10/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import time
import numpy as np
from mealpy.optimizer import Optimizer
from mafese.wrapper.mha import MhaSelector
from mafese.utils.mealpy_util import FeatureSelectionProblem
from mafese.utils.data_loader import Data
from permetrics.regression import RegressionMetric
from permetrics.classification import ClassificationMetric


class CBPI(Optimizer):
    """
    Cluster-based population initialization. First stage of SAGA algorithm

    end_pop_size should correspond to the pop-size for the actual metaheuristic

    References
    ~~~~~~~~~~
    [1] Marjit, S., Bhattacharyya, T., Chatterjee, B., & Sarkar, R., 2023. Simulated
    annealing aided genetic algorithm for gene selection from microarray data. Computers
    in Biology and Medicine, 158, 106854.
    """

    ID_POS = 0  # Index of position/location of solution/agent
    ID_TAR = 1  # Index of target list, (includes fitness value and objectives list)

    ID_FIT = 0  # Index of target (the final fitness) in fitness
    ID_OBJ = 1  # Index of objective list in target

    EPSILON = 10E-10

    def __init__(self, epoch=1, pop_size=100, end_pop_size=10, eta=0.7, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of initial population size, default = 100
            end_pop_size (int): number of final population size after collapsing
                                the clusters, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.end_pop_size = self.validator.check_int("end_pop_size", end_pop_size, [1, 10000])
        self.eta = self.validator.check_float("eta", eta, [0.01, 1.0])
        self.set_parameters(["epoch", "pop_size", "end_pop_size"])
        self.sort_flag = False
    
    def is_equal(self, ind_1, ind_2):
        """
        Check if two individuals are identical

        Args:
            ind_1 (list): the first individual
            ind_2 (list): the second individual
        
        Returns:
            status (bool): whether or not the individuals are identical
        """
        status = False
        for k in range(0, len(ind_1)):
            if ind_1[k] == ind_2[k]:
                status = True
                continue
            else:
                status = False
                break
        return status

    def hamming_dist(self, ind_1, ind_2):
        """
        Calculate the Hamming distance between two indiduals

        Args:
            ind_1 (list): the first individual
            ind_2 (list): the second individual
        
        Returns:
            count (int): the hamming distance between the two input individuals
        """
        i, count = 0, 0
        while(i < len(ind_1)):
            if(ind_1[i] != ind_2[i]):
                count += 1
            i += 1
        return count
    
    def calculate_similarity(self, non_cluster, cluster):
        """
        Calculate the similarity between non-cluster center and cluster center

        Args:
            ind_1 (list): the first individual
            ind_2 (list): the second individual
        
        Returns:
            similarity (float): the similarity as a linear combination of the
                                Hamming distance and the fitness value
        """

        non_cluster_pos = non_cluster[self.ID_POS].copy()
        non_cluster_fit = non_cluster[self.ID_TAR][self.ID_FIT].copy()
        cluster_pos = cluster[self.ID_POS].copy()
        cluster_fit = cluster[self.ID_TAR][self.ID_FIT].copy()

        hamming_dist = self.hamming_dist(cluster_pos, non_cluster_pos)
        fit_diff = abs(non_cluster_fit - cluster_fit)
        if fit_diff <= 10e-5:
            fit_diff = 0.001
        if hamming_dist <= 10e-5:
            hamming_dist = 0.001
        
        similarity = self.eta*(1/fit_diff) + (1 - self.eta)*(1/hamming_dist)
        return similarity
    
    def evaluate_goodness(self, cluster):
        """
        Evaluate the goodness of the cluster population

        Args:
            cluster (list): list of population members belonging to a particular
                            cluster
        
        Returns:
            final_individual (list): the position and target value of the final
                                     representative cluster member
        """
        final_center = np.zeros(self.problem.n_dims)

        if len(cluster) == 1:
            return cluster[0]

        # extract positional information
        cluster_pos = np.zeros((len(cluster), self.problem.n_dims))
        cluster_fit = np.zeros(len(cluster))
        for idx in range(0, len(cluster)):
            pos = cluster[idx][self.ID_POS].copy()
            fit = cluster[idx][self.ID_TAR][self.ID_FIT]
            cluster_pos[idx] = pos
            cluster_fit[idx] = fit
        
        gij = np.zeros(self.problem.n_dims)
        for ith_feat in range(0, cluster_pos.shape[1]):
            numerator, denominator = 0, 0
            for num_indexes in range(0, cluster_pos.shape[0]):
                numerator += cluster_pos[num_indexes][ith_feat] * cluster_fit[num_indexes]
                denominator += cluster_pos[num_indexes][ith_feat]
            
            gij[ith_feat] = numerator/denominator
        t_mean = np.mean(gij[~np.isnan(gij)])

        for ith_feat in range(0, gij.shape[0]):
            if np.isnan(gij[ith_feat]):
                final_center[ith_feat] = 0
                continue
            if gij[ith_feat] >= t_mean:
                final_center[ith_feat] = 1
            else:
                final_center[ith_feat] = 0

        if np.sum(final_center) == 0:
            final_center = np.array([0]*(self.problem.n_dims - 10) + [1]*10)
            np.random.shuffle(final_center)
        elif final_center is None:
            final_center = np.array([0]*(self.problem.n_dims - 10) + [1]*10)
            np.random.shuffle(final_center)

        final_target = self.get_target_wrapper(final_center)
        
        return [final_center, final_target]
    
    def cbpi(self):
        """
        Perform the clustering-based population initialization
        """
        # randomly initiate cluster centers and get non-center individuals
        cluster_centers_indexes = np.random.choice(self.pop_size, self.end_pop_size, replace=False)
        cluster_centers = [self.pop[idx] for idx in cluster_centers_indexes]

        # remove cluster centers from original population
        pop_new = []
        remove_indexes = []
        for i_idx in range(0, self.pop_size):
            pos = self.pop[i_idx][self.ID_POS].copy()
            for j_idx in range(0, self.end_pop_size):
                center_pos = cluster_centers[j_idx][self.ID_POS].copy()
                if self.is_equal(pos, center_pos):
                    remove_indexes.append(i_idx)
                    break
        remove_indexes.sort(reverse = True)
        pop_new = [self.pop[idx] for idx in range(0, self.pop_size) if idx not in remove_indexes]
                    
        # evaluate cluster centers
        for idx in range(0, len(cluster_centers)):
            pos = cluster_centers[idx][self.ID_POS].copy()
            if self.mode not in self.AVAILABLE_MODES:
                cluster_centers[idx][self.ID_TAR] = self.get_target_wrapper(pos)
        
        if self.mode in self.AVAILABLE_MODES:
            cluster_centers = self.update_target_wrapper_population(cluster_centers)
        
        # evaluate non-cluster individuals
        for idx in range(0, len(pop_new)):
            pos = pop_new[idx][self.ID_POS].copy()
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx][self.ID_TAR] = self.get_target_wrapper(pos)
        
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
        
        # assign individuals to clusters
        cluster_assignment = []
        for i_idx in range(0, len(pop_new)):
            non_center_ind = pop_new[i_idx].copy()
            similarity_ar = np.zeros(len(cluster_centers))
            for j_idx in range(0, len(cluster_centers)):
                cluster_center_ind = cluster_centers[j_idx].copy()
                similarity = self.calculate_similarity(non_center_ind, cluster_center_ind)
                similarity_ar[j_idx] = similarity
            cluster_assignment.append(np.argmax(similarity_ar))
        
        # make list of lists with cluster members
        all_clusters = []
        for center_idx in range(0, len(cluster_centers)):
            cluster_i = []
            cluster_i.append(cluster_centers[center_idx].copy())
            
            for non_center_idx in range(0, len(pop_new)):
                clust_assign = cluster_assignment[non_center_idx]
                if clust_assign == center_idx:
                    cluster_i.append(pop_new[non_center_idx].copy())
            
            all_clusters.append(cluster_i)
        
        pop_fin = []
        # collapse clusters to single individuals
        for idx in range(0, len(all_clusters)):
            single_cluster = self.evaluate_goodness(all_clusters[idx].copy())
            pop_fin.append(single_cluster)
        
        return pop_fin
    
    def solve(self, problem=None, mode='single', starting_positions=None, n_workers=None, termination=None):
        """
        Modified version for the specific case of CBPI

        Args:
            problem (Problem, dict): an instance of Problem class or a dictionary

                problem = {
                    "fit_func": your objective function,
                    "lb": list of value
                    "ub": list of value
                    "minmax": "min" or "max"
                    "verbose": True or False
                    "n_dims": int (Optional)
                    "obj_weights": list weights corresponding to all objectives (Optional, default = [1, 1, ...1])
                }

            mode (str): Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                * 'process': The parallel mode with multiple cores run the tasks
                * 'thread': The parallel mode with multiple threads run the tasks
                * 'swarm': The sequential mode that no effect on updating phase of other agents
                * 'single': The sequential mode that effect on updating phase of other agents, default

            starting_positions(list, np.ndarray): List or 2D matrix (numpy array) of starting positions with length equal pop_size parameter
            n_workers (int): The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
            termination (dict, None): The termination dictionary or an instance of Termination class

        Returns:
            list: [position, fitness value]
        """
        self.check_problem(problem)
        self.check_mode_and_workers(mode, n_workers)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_positions)
        self.initialization()
        self.after_initialization()

        self.before_main_loop()
        for epoch in range(0, self.epoch):
            time_epoch = time.perf_counter()

            self.pop = self.cbpi()

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch + 1, time_epoch)
        self.track_optimize_process()

        return self.pop

    def track_optimize_process(self):
        """
        Save some historical data after training process finished
        """
        self.history.epoch = len(self.history.list_diversity)
        div_max = np.max(self.history.list_diversity)
        self.history.list_exploration = 100 * (np.array(self.history.list_diversity) / div_max)
        self.history.list_exploitation = 100 - self.history.list_exploration

class CBPI_Wrapper(MhaSelector):
    """
    Wrapper for the CBPI algorithm for the specific purpose of feature selection
    """
    def __init__(self, *args, **kwargs):
        """
        Pass down all arguments to the parent class
        """
        super().__init__(*args, **kwargs)
    
    def fit(self, X_train, y_train, X_test, y_test, fit_weights=(0.9, 0.1), verbose=True, mode='single', n_workers=None, termination=None):
        """
        A customized version to return a clustered population via CBPI

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
        clustered_pop = self.optimizer.solve(prob, mode=mode, n_workers=n_workers, termination=termination)
        return clustered_pop