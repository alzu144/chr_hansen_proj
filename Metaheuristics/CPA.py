# !usr/bin/env python
# Created by "Alexander Zubov", 26/09/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseCPA(Optimizer):
    """
    The original version of: CPA with Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + w (float): [0.5, 9.0], weight for adjusting the prey power over time, default = 9.0

    References
    ~~~~~~~~~~
    [1] Tu, J., Chen, H., Wang, M. and Gandomi, A. H., 2021. The colony predation algorithm.
    Journal of Bionic Engineering, 18, pp.674-710.
    """
    def __init__(self, epoch=10000, pop_size=100, w=9.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            w (float): weight for the base power of the prey, default = 2.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.w = self.validator.check_float("w", w, [0.5, 10.0])
        self.set_parameters(["epoch", "pop_size", "w"])
        self.sort_flag = False
    
    def get_pos_only(self):
        """
        Get only positional information for iterating over the dimensions

        Returns:
            hunter_pos (np.array): the positional information of the population 
        """
        hunter_pos = np.zeros((self.pop_size, self.problem.n_dims))
        for idx in range(0, self.pop_size):
            hunter = self.pop[idx].copy()
            hunter_pos_only = hunter[self.ID_POS]
            hunter_pos[idx] = hunter_pos_only
        return hunter_pos

    def communication(self, epoch):
        """
        The population based-communication step

        Args:
            epoch (int): The current iteration

        Returns:
            hunter_pos (np.array): the (updated) positional information of the population 
        """
        hunter_pos = self.get_pos_only()
        for j in range(self.problem.n_dims):
            dimensional_pop = hunter_pos[:, j]
            dimensional_pop.sort()
            hunter_j_close1 = dimensional_pop[0]
            hunter_j_close1 = dimensional_pop[1]

            com_idx = round(np.random.rand()*(hunter_pos.shape[0] - 1) + 0.5)
            if np.random.rand() < epoch/self.epoch:
                hunter_pos[com_idx] = self.g_best[self.ID_POS]

            r_com = np.random.rand()
            hunter_pos[com_idx, j] = r_com*hunter_pos[com_idx, j] + (1 - r_com)*((hunter_j_close1 + hunter_j_close1)/2)
            return hunter_pos

    def decision_making(self, epoch, hunter_pos):
        """
        Decision-making on an individual basis, based on the position in the search space and the prey power

        No return argument, updates the self.pop argument as the final step

        Args:
            epoch (int): The current iteration
            hunter_pos (np.array): the (updated) positional information of the population
        """
        a = np.exp(self.w - 2*self.w*(epoch/self.epoch))
        S_0 = a*(1 - epoch/self.epoch)

        pop_new = []
        for idx in range(0, self.pop_size):
            already_evaluated = False
            hunter = hunter_pos[idx].copy()

            S = 2*S_0*np.random.rand() - S_0
            l = np.random.rand()
            # prey-power: balance between exploration and exploitation
            if abs(S) < 2/3 * a:
                if np.random.rand() > 0.5:
                    # dispersion
                    x_new = self.g_best[self.ID_POS] - S*(np.random.rand(self.problem.n_dims)*(self.problem.ub - self.problem.lb) + self.problem.lb)
                else:
                    # encirclement
                    D_prey = np.abs(self.g_best[self.ID_POS] - hunter)
                    x_new = self.g_best[self.ID_POS] - 2*S*D_prey*np.exp(l)*np.tan(l * np.pi/4)
            else:
                D_strat = 4*np.random.rand() - 2
                if abs(D_strat) < 1:
                    # peer support (modified to reduce number of fes)
                    P = hunter_pos.copy()
                    D_support = np.linalg.norm(self.g_best[self.ID_POS] - P)
                    P_closest = P[np.argmin(D_support)]
                    P_pos_new = np.random.rand(self.problem.n_dims)*P_closest
                    P_pos_new = self.amend_position(P_pos_new, self.problem.lb, self.problem.ub)
                    P_target_new = self.get_target_wrapper(P_pos_new)
                    P_sup_new = [P_pos_new, P_target_new]
                    if self.compare_agent(P_sup_new, self.g_best):
                        pop_new.append(P_sup_new)
                    else:
                        pop_new.append(self.g_best)
                    self.pop[idx] = pop_new[-1]
                    already_evaluated = True
                    break
                else:
                    # search for new food source
                    alternative_food = np.random.rand(self.problem.n_dims)*(self.problem.ub - self.problem.lb) + self.problem.lb
                    D_alt = np.abs(2*np.random.rand()*alternative_food - hunter)
                    x_new = alternative_food - S*D_alt

            if not already_evaluated:
                pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
                    self.pop[idx] = pop_new[-1]
            
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_wrapper_population(pop_new)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # population-based communication
        hunter_pos = self.communication(epoch)

        # individual decision-making
        self.decision_making(epoch, hunter_pos)