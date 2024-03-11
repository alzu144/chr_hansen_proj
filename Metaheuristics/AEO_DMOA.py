# !usr/bin/env python
# Created by "Alexander Zubov", 05/10/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class AEO_DMOA(Optimizer):
    """
    The original version of: OriginalAEO with OriginalDMOA.

    Combination of artificial ecosystem optimization and dwarf mongoose optimization algorithm.
    Utilizes AEO in the first half for exploration and DMOA in the second half for exploitation.


    References
    ~~~~~~~~~~
    [1] Al-Shourbaji, I., Kachare, P., Fadlelseed, S., Jabbari, A., Hussien, A. G., Al-Saggar, F., ...
    & Alameen, A., 2023. Artificial Ecosystem-Based Optimization with Dwarf Mongooese Optimization
    for Feature Selection and Global Optimization Problems. International Journal of Computational
    Intelligence Systems, 16(1), pp.1-24.
    """
    def __init__(self, epoch=10000, pop_size=100, n_baby_sitter=3, peep=2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_baby_sitter (int): number of babysitters, default = 3
            peep (float): ?[WIP], default = 2.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_baby_sitter = self.validator.check_int("n_baby_sitter", n_baby_sitter, [2, 10])
        self.peep = self.validator.check_float("peep", peep, [1, 10.])
        self.n_scout = self.pop_size - self.n_baby_sitter
        self.support_parallel_modes = False
        self.set_parameters(["epoch", "pop_size", "n_baby_sitter", "peep"])
        self.sort_flag = False

    def initialize_variables(self):
        self.C = np.zeros(self.pop_size)
        self.tau = -np.inf
        self.L = np.round(0.6 * self.problem.n_dims * self.n_baby_sitter)
    
    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # check for exploration or exploitation phase
        if epoch <= int(self.epoch/2):
            ## Production   - Update the worst agent
            # Eq. 2, 3, 1
            a = (1.0 - epoch / self.epoch) * np.random.uniform()
            x1 = (1 - a) * self.pop[-1][self.ID_POS] + a * np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(x1, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            self.pop[-1] = [pos_new, target]

            ## Consumption - Update the whole population left
            pop_new = []
            for idx in range(0, self.pop_size - 1):
                rand = np.random.random()
                # Eq. 4, 5, 6
                v1 = np.random.normal(0, 1)
                v2 = np.random.normal(0, 1)
                c = 0.5 * v1 / abs(v2)  # Consumption factor
                j = 1 if idx == 0 else np.random.randint(0, idx)
                ### Herbivore
                if rand < 1.0 / 3:
                    x_t1 = self.pop[idx][self.ID_POS] + c * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])  # Eq. 6
                ### Carnivore
                elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                    x_t1 = self.pop[idx][self.ID_POS] + c * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS])  # Eq. 7
                ### Omnivore
                else:
                    r2 = np.random.uniform()
                    x_t1 = self.pop[idx][self.ID_POS] + c * (r2 * (self.pop[idx][self.ID_POS] - self.pop[0][self.ID_POS])
                                                            + (1 - r2) * (self.pop[idx][self.ID_POS] - self.pop[j][self.ID_POS]))
                pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_wrapper_population(pop_new)
                self.pop[:-1] = self.greedy_selection_population(self.pop[:-1], pop_new)

            ## find current best used in decomposition
            _, best = self.get_global_best_solution(self.pop)

            ## Decomposition
            ### Eq. 10, 11, 12, 9
            pop_child = []
            for idx in range(0, self.pop_size):
                r3 = np.random.uniform()
                d = 3 * np.random.normal(0, 1)
                e = r3 * np.random.randint(1, 3) - 1
                h = 2 * r3 - 1
                x_t1 = best[self.ID_POS] + d * (e * best[self.ID_POS] - h * self.pop[idx][self.ID_POS])
                pos_new = self.amend_position(x_t1, self.problem.lb, self.problem.ub)
                pop_child.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
            if self.mode in self.AVAILABLE_MODES:
                pop_child = self.update_target_wrapper_population(pop_child)
                self.pop = self.greedy_selection_population(pop_child, self.pop)
        else:
            ## Abandonment Counter
            CF = (1 - (epoch + 1) / self.epoch) ** (2 * (epoch + 1) / self.epoch)
            fit_list = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
            mean_cost = np.mean(fit_list)
            fi = np.exp(-fit_list / mean_cost)

            ## Foraging led by Alpha female
            for idx in range(0, self.pop_size):
                alpha = self.get_index_roulette_wheel_selection(fi)
                k = np.random.choice(list(set(range(0, self.pop_size)) - {idx, alpha}))
                ## Define Vocalization Coeff.
                phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
                new_pos = self.pop[alpha][self.ID_POS] + phi * (self.pop[alpha][self.ID_POS] - self.pop[k][self.ID_POS])
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_tar = self.get_target_wrapper(new_pos)
                if self.compare_agent([new_pos, new_tar], self.pop[idx]):
                    self.pop[idx] = [new_pos, new_tar]
                else:
                    self.C[idx] += 1

            ## Scout group
            SM = np.zeros(self.pop_size)
            for idx in range(0, self.pop_size):
                k = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                ## Define Vocalization Coeff.
                phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
                new_pos = self.pop[idx][self.ID_POS] + phi * (self.pop[idx][self.ID_POS] - self.pop[k][self.ID_POS])
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_tar = self.get_target_wrapper(new_pos)
                ## Sleeping mould
                SM[idx] = (new_tar[self.ID_FIT] - self.pop[idx][self.ID_TAR][self.ID_FIT]) / np.max([new_tar[self.ID_FIT], self.pop[idx][self.ID_TAR][self.ID_FIT]])
                if self.compare_agent([new_pos, new_tar], self.pop[idx]):
                    self.pop[idx] = [new_pos, new_tar]
                else:
                    self.C[idx] += 1

            ## Baby sitters
            for idx in range(0, self.pop_size):
                if self.C[idx] >= self.L:
                    self.pop[idx] = self.create_solution(self.problem.lb, self.problem.ub)
                    self.C[idx] = 0

            ## Next Mongoose position
            new_tau = np.mean(SM)
            for idx in range(0, self.pop_size):
                phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
                if new_tau > SM[idx]:
                    new_pos = self.g_best[self.ID_POS] - CF * phi * (self.g_best[self.ID_POS] - SM[idx] * self.pop[idx][self.ID_POS])
                else:
                    new_pos = self.pop[idx][self.ID_POS] + CF * phi * (self.g_best[self.ID_POS] - SM[idx] * self.pop[idx][self.ID_POS])
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                new_tar = self.get_target_wrapper(new_pos)
                if self.compare_agent([new_pos, new_tar], self.pop[idx]):
                    self.pop[idx] = [new_pos, new_tar]