# !usr/bin/env python
# Created by "Alexander Zubov", 05/10/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.physics_based.SA import OriginalSA
from permetrics.regression import RegressionMetric
from permetrics.classification import ClassificationMetric
from .CBPI import CBPI, CBPI_Wrapper
from .custom_selector import MhaSelector_custom


from mealpy.evolutionary_based.GA import BaseGA, EliteMultiGA

class SAGA(EliteMultiGA):
    """
    The original version of: Simulated Annealing-aided Genetic Algorithm (SAGA)

    Utilizes the original SA with the developed elite multipoints-mutation version of GA

    Links:
        1. https://www.baeldung.com/cs/elitism-in-evolutionary-algorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation (str): Optional, can be ["flip", "swap"] for multipoints
        + elite_best (float/int): Optional, can be float (percentage of the best in elite group), or int (the number of best elite), default = 0.1
        + elite_worst (float/int): Opttional, can be float (percentage of the worst in elite group), or int (the number of worst elite), default = 0.3
        + strategy (int): Optional, can be 0 or 1. If = 0, the selection is select parents from (elite_worst + non_elite_group).
            Else, the selection will select dad from elite_worst and mom from non_elite_group.
        + pop_size = elite_group (elite_best + elite_worst) + non_elite_group

    References
    ~~~~~~~~~~
    [1] Marjit, S., Bhattacharyya, T., Chatterjee, B., & Sarkar, R., 2023. Simulated
    annealing aided genetic algorithm for gene selection from microarray data. Computers
    in Biology and Medicine, 158, 106854.
    """
    def __init__(self, epoch=10000, pop_size=100, init_pop_size=400, sa_epoch=10, sa_stepsize=0.01, sa_tempinit=1000, pc=0.95, pm=0.8, 
                 selection="roulette", crossover="uniform", mutation="swap", k_way=0.2,
                 elite_best=0.1, elite_worst=0.3, strategy=0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            init_pop_size (int): number of initial population size for use in CBPI, default = 400
            pc (float): cross-over probability, default = 0.95
            pm (float): mutation probability, default = 0.025
            selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = False
            mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point, default="flip"
        """
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.init_pop_size = self.validator.check_int("init_pop_size", init_pop_size, [20, 10000])
        self.sa_epoch = self.validator.check_int("sa_epoch", sa_epoch, (1, 100000))
        self.sa_stepsize = self.validator.check_float("sa_stepsize", sa_stepsize, (0.00000001, 10.0))
        self.sa_tempinit = self.validator.check_int("sa_tempinit", sa_tempinit, (1, 200))
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "init_pop_size", "pc", "pm", "sa_epoch", "sa_stepsize", "sa_tempinit"])
        self.sort_flag = False
        self.selection = "tournament"
        self.k_way = 0.2
        self.crossover = "uniform"
        self.mutation = "flip"
        self.mutation_multipoints = True
    
    def save_to_history(self, secondary_history):
        """
        Save historical information of the pre-optimization steps

        Args:
            secondary_history (dict):
        """
        self.history.list_epoch_time = secondary_history["list_epoch_time"] + self.history.list_epoch_time
        self.history.list_current_best = secondary_history["list_current_best"] + self.history.list_current_best
        self.history.list_current_best_fit = secondary_history["list_current_best_fit"] + self.history.list_current_best_fit
        self.history.list_current_worst = secondary_history["list_current_worst"] + self.history.list_current_worst
        self.history.list_global_best = secondary_history["list_global_best"] + self.history.list_global_best
        self.history.list_global_best_fit = secondary_history["list_global_best_fit"] + self.history.list_global_best_fit
        self.history.list_global_worst = secondary_history["list_global_worst"] + self.history.list_global_worst
        self.history.list_population = secondary_history["list_population"] + self.history.list_population
        self.history.list_diversity = secondary_history["list_diversity"] + self.history.list_diversity

        if isinstance(self.history.list_exploration, np.ndarray):
            self.history.list_exploration = self.history.list_exploration.tolist()
        if isinstance(secondary_history["list_exploration"], np.ndarray):
            secondary_history["list_exploration"] = secondary_history["list_exploration"].tolist()
        self.history.list_exploration = secondary_history["list_exploration"] + self.history.list_exploration

        if isinstance(self.history.list_exploitation, np.ndarray):
            self.history.list_exploitation = self.history.list_exploitation.tolist()
        if isinstance(secondary_history["list_exploitation"], np.ndarray):
            secondary_history["list_exploitation"] = secondary_history["list_exploitation"].tolist()
        self.history.list_exploitation = secondary_history["list_exploitation"] + self.history.list_exploitation
    
    def save_cbpi_history(self, cbpi_history):
        """
        Save historical information of the CBPI step

        Args:
            cbpi_history (History): An object of class History with attributes storing the historical
                                    information of the CBPI step

        Returns:
            cbpi_history_dict (dict): A dictionary storing the lists containing the historical information
        """
        cbpi_history_dict = {
            "list_epoch_time": cbpi_history.list_epoch_time,
            "list_current_best": cbpi_history.list_current_best,
            "list_current_best_fit": cbpi_history.list_current_best_fit,
            "list_current_worst": cbpi_history.list_current_worst,
            "list_global_best": cbpi_history.list_global_best,
            "list_global_best_fit": cbpi_history.list_global_best_fit,
            "list_global_worst": cbpi_history.list_global_worst,
            "list_population": cbpi_history.list_population,
            "list_diversity": cbpi_history.list_diversity,
            "list_exploration": cbpi_history.list_exploration,
            "list_exploitation": cbpi_history.list_exploitation
        }
        return cbpi_history_dict
    
    def append_to_sa_histories(self, sa_histories, current_sa_history):
        sa_histories["list_epoch_time"].append(current_sa_history.list_epoch_time)
        sa_histories["list_current_best"].append(current_sa_history.list_current_best)
        sa_histories["list_current_best_fit"].append(current_sa_history.list_current_best_fit)
        sa_histories["list_current_worst"].append(current_sa_history.list_current_worst)
        sa_histories["list_global_best"].append(current_sa_history.list_global_best)
        sa_histories["list_global_best_fit"].append(current_sa_history.list_global_best_fit)
        sa_histories["list_global_worst"].append(current_sa_history.list_global_worst)
        sa_histories["list_population"].append(current_sa_history.list_population)
        sa_histories["list_diversity"].append(current_sa_history.list_diversity)
        sa_histories["list_exploration"].append(current_sa_history.list_exploration)
        sa_histories["list_exploitation"].append(current_sa_history.list_exploitation)
    
    def save_sa_history(self, sa_histories):
        """
        Save historical information of the SAGA steps

        Args:
            sa_histories (dict): A dictionary storing the histories of the individual SA optimizers for the individual
                                 cluster centers

        Returns:
            sa_history_dict (dict): A dictionary storing the per-epoch history of all individual SA optimizers combined
        """
        list_epoch_time = []
        for epoch_info in zip(*sa_histories["list_epoch_time"]):
            epoch_time_i = [i for i in epoch_info]
            list_epoch_time.append(epoch_time_i)
        
        list_current_best = []
        for current_best_info in zip(*sa_histories["list_current_best"]):
            current_best_i = [i for i in current_best_info]
            list_current_best.append(current_best_i)
        list_current_best_fit = []
        for current_best_fit_info in zip(*sa_histories["list_current_best_fit"]):
            current_best_fit_i = [i for i in current_best_fit_info]
            list_current_best_fit.append(current_best_fit_i)
        list_current_worst = []
        for current_worst_info in zip(*sa_histories["list_current_worst"]):
            current_worst_i = [i for i in current_worst_info]
            list_current_worst.append(current_worst_i)

        list_global_best = []
        for global_best_info in zip(*sa_histories["list_global_best"]):
            global_best_i = [i for i in global_best_info]
            list_global_best.append(global_best_i)
        list_global_best_fit = []
        for global_best_fit_info in zip(*sa_histories["list_global_best_fit"]):
            global_best_fit_i = [i for i in global_best_fit_info]
            list_global_best_fit.append(global_best_fit_i)
        list_global_worst = []
        for global_worst_info in zip(*sa_histories["list_global_worst"]):
            global_worst_i = [i for i in global_worst_info]
            list_global_worst.append(global_worst_i)
        
        list_population = []
        for population_info in zip(*sa_histories["list_population"]):
            population_i = [i for i in population_info]
            list_population.append(population_i)
        list_diversity = []
        for diversity_info in zip(*sa_histories["list_diversity"]):
            diversity_i = [i for i in diversity_info]
            list_diversity.append(diversity_i)
        list_exploration = []
        for exploration_info in zip(*sa_histories["list_exploration"]):
            exploration_i = [i for i in exploration_info]
            list_exploration.append(exploration_i)
        list_exploitation = []
        for exploitation_info in zip(*sa_histories["list_exploitation"]):
            exploitation_i = [i for i in exploitation_info]
            list_exploitation.append(exploitation_i)
        
        sa_history_dict = {
            "list_epoch_time": list_epoch_time,
            "list_current_best": list_current_best,
            "list_current_best_fit": list_current_best_fit,
            "list_current_worst": list_current_worst,
            "list_global_best": list_global_best,
            "list_global_best_fit": list_global_best_fit,
            "list_global_worst": list_global_worst,
            "list_population": list_population,
            "list_diversity": list_diversity,
            "list_exploration": list_exploration,
            "list_exploitation": list_exploitation
        }
        return sa_history_dict
    
    def initialization(self):
        """
        Clustering-based population initialization and subsequent pre-optimization via SA
        """
        estimator_ = self.problem.estimator
        transfer_func_ = self.problem.transfer_func
        metric_class_ = self.problem.metric_class
        if metric_class_ == RegressionMetric:
            problem_dir = "regression"
        elif metric_class_ == ClassificationMetric:
            problem_dir = "classification"
        data_ = self.problem.data
        
        # clustering-based population initialization
        cbpi = CBPI(epoch=1, pop_size=self.init_pop_size, end_pop_size=self.pop_size)
        cbpi_selector = CBPI_Wrapper(
            problem=problem_dir, estimator= estimator_, optimizer=cbpi,
            optimizer_paras=None, transfer_func=transfer_func_
        )
        clustered_pop = cbpi_selector.fit(
            data_.X_train, data_.y_train, data_.X_test, data_.y_test, 
            fit_weights=(0.9, 0.1), verbose=True
        )
        self.cbpi_history = self.save_cbpi_history(cbpi_selector.optimizer.history)

        # simulated annealing on clustered population
        pre_optim_pop = []
        sa_histories = {
            "list_epoch_time": [], "list_current_best": [], "list_current_best_fit": [],
            "list_current_worst": [], "list_global_best": [], "list_global_best_fit": [],
            "list_global_worst": [], "list_population": [], "list_diversity": [],
            "list_exploration": [], "list_exploitation": []}
        for idx in range(0, self.pop_size):
            pop_i = clustered_pop[idx].copy()
            sa_cbpi = SA_CBPI(pop_i, epoch=10, step_size=0.0001, temp_init=100)
            sa_cbpi_selector = MhaSelector_custom(
                problem=problem_dir, estimator= estimator_, optimizer=sa_cbpi,
                optimizer_paras=None, transfer_func=transfer_func_
            )
            sa_cbpi_selector.fit(
                data_.X_train, data_.y_train, data_.X_test, data_.y_test, 
                fit_weights=(0.9, 0.1), verbose=False
            )
            pre_optim_pop_i = sa_cbpi_selector.optimizer.g_best
            pre_optim_pop.append(pre_optim_pop_i)
            self.append_to_sa_histories(sa_histories, sa_cbpi_selector.optimizer.history)
        self.sa_history = self.save_sa_history(sa_histories)
        
        self.pop = pre_optim_pop
    
    def track_optimize_process(self):
        """
        Save some historical data after training process finished

        Modified by saving history of the pre-optimization steps (CBPI and SA)
        """
        self.history.epoch = len(self.history.list_diversity)
        div_max = np.max(self.history.list_diversity)
        self.history.list_exploration = 100 * (np.array(self.history.list_diversity) / div_max)
        self.history.list_exploitation = 100 - self.history.list_exploration
        self.history.list_global_best = self.history.list_global_best[1:]
        self.history.list_current_best = self.history.list_current_best[1:]
        self.solution = self.history.list_global_best[-1]
        self.history.list_global_worst = self.history.list_global_worst[1:]
        self.history.list_current_worst = self.history.list_current_worst[1:]

        # insert pre-optimization history values in reverse order to create final history in correct order
        self.save_to_history(self.sa_history)
        self.save_to_history(self.cbpi_history)

class SA_CBPI(OriginalSA):
    """
    Original version of simulated annealing (SA) for pre-optimization during the SAGA algorithm

    Notes:
        + SA_CBPI is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + temp_init (float): [1, 10000], initial temperature, default=100
        + step_size (float): the step size of random movement, default=0.1

    References
    ~~~~~~~~~~
    [1] Marjit, S., Bhattacharyya, T., Chatterjee, B., & Sarkar, R., 2023. Simulated
    annealing aided genetic algorithm for gene selection from microarray data. Computers
    in Biology and Medicine, 158, 106854.
    """

    def __init__(self, pop_init, epoch=10000, pop_size=2, temp_init=100, step_size=0.1,  **kwargs):
        """
        Args:
            pop_init (list): the initial population member as prepared via CBPI
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            temp_init (float): initial temperature, default=100
            step_size (float): the step size of random movement, default=0.1
        """
        super().__init__(**kwargs)
        self.pop_init = pop_init
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.temp_init = self.validator.check_float("temp_init", temp_init, [1, 10000])
        self.step_size = self.validator.check_float("step_size", step_size, (-100., 100.))
        self.set_parameters(["epoch", "temp_init", "step_size"])
    
    def initialization(self):
        pop = []
        for i in range(2):
            pop.append(deepcopy(self.pop_init))
        self.pop = pop