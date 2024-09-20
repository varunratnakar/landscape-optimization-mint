from pymoo.core.callback import Callback
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.termination import get_termination
from pymoo.problems.many import get_ref_dirs
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
import numpy as np

class GenCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["gen200"] = []
        self.data["gen500"] = []
        self.data["gen1000"] = []
        # self.data["best"] = []

    def notify(self, algorithm):
      if algorithm.n_gen == 200:
        self.data["gen200"].append(algorithm.opt.get("F"))
      if algorithm.n_gen == 500:
        self.data["gen500"].append(algorithm.opt.get("F"))
      if algorithm.n_gen == 1000:
        self.data["gen1000"].append(algorithm.opt.get("F"))

class ModGenCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["gen200"] = []
        self.data["gen500"] = []
        self.data["gen1000"] = []
        # self.data["best"] = []

    def notify(self, algorithm):
      if algorithm.n_gen == 1:
        self.data["gen200"].append(algorithm.opt.get("F"))
      if algorithm.n_gen == 2:
        self.data["gen500"].append(algorithm.opt.get("F"))
      if algorithm.n_gen == 3:
        self.data["gen1000"].append(algorithm.opt.get("F"))

class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X
    
class MySamplingWeight(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            rand_perm = np.random.permutation(problem.n_var)
            I = []
            cur_area = 0
            for idx_I in range(problem.n_var):
                if cur_area + problem.areas[rand_perm[idx_I]] <= problem.n_max:
                    I.append(rand_perm[idx_I])
                    cur_area += problem.areas[rand_perm[idx_I]]
            # I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X

class BinaryCrossover(Crossover):

    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X

class BinaryCrossoverWeight(Crossover):

    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(problem.areas[both_are_true])

            I = np.where(np.logical_xor(p1, p2))[0]

            # S = I[np.random.permutation(len(I))][:n_remaining]
            rand_I = np.random.permutation(len(I))
            for idx_I in range(len(rand_I)):
                if n_remaining >= problem.areas[I[rand_I[idx_I]]]:
                    n_remaining -= problem.areas[I[rand_I[idx_I]]]
                    _X[0, k, I[rand_I[idx_I]]] = True
                # else:
                #     break
            # _X[0, k, S] = True

        return _X

class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_true)] = False
            rand_idx = np.random.choice(is_false)
            # print(is_false, X[i])
            if problem.n_max >= np.sum(problem.areas[rand_idx]) + np.sum(problem.areas[np.where(X[i, :])[0]]):
                X[i, rand_idx] = True

        return X


class BaseProblem(ElementwiseProblem):
    def __init__(self,
                 values_df,
                 n_max,
                 rx_burn_units,
                #  ignition_points,
                 prevention_df
                 ):
        super().__init__(n_var=rx_burn_units.shape[0], n_obj=3, n_constr=1)
        self.values_df = values_df
        self.n_max = n_max
        # self.rx_burn_units = rx_burn_units
        # self.ignition_points = ignition_points
        self.prevention_df = prevention_df

    def _evaluate(self, x, out, *args, **kwargs):
        
        # # TODO: MAKE THE FOLLOWING SECTION FASTER, VECTORIZE WHEREVER POSSIBLE
        # plan_burns = self.rx_burn_units.iloc[x]
        
        # # TODO: PREPROCESS THE CONTAINED IGNITIONS FOR EACH RX_BURN_UNIT
        # plan_burns_dissolved = plan_burns.dissolve()
        # plan_polys = plan_burns_dissolved.geometry[0]
        # contained_idx = np.apply_along_axis(lambda x : point_in_poly(x, plan_polys), 1, self.ignition_points)

        f1 = -np.sum(self.prevention_df[x].f1)
        f2 = -np.sum(self.prevention_df[x].f2)
        f3 = -np.sum(self.prevention_df[x].f3)

        out["F"] = [f1, f2, f3]
        out["G"] = (self.n_max - np.sum(x)) ** 2

class HazardProblem(ElementwiseProblem):
    def __init__(self,
                 values_df,
                 n_max,
                 rx_burn_units,
                #  ignition_points,
                 prevention_df
                 ):
        
        self.non_zero_idx = np.where(prevention_df.burned_area > 0)[0]
        print('non_zero_idx:')
        print(self.non_zero_idx)
        super().__init__(n_var=len(self.non_zero_idx), n_obj=3, n_constr=1)
        # choose the subset of values_df and prevention_df that are non-zero
        self.values_df = values_df.iloc[self.non_zero_idx]
        self.n_max = n_max
        self.areas = prevention_df.area[self.non_zero_idx].values
        # self.rx_burn_units = rx_burn_units
        # self.ignition_points = ignition_points
        self.prevention_df = prevention_df.iloc[self.non_zero_idx]

    def _evaluate(self, x, out, *args, **kwargs):
        # print(x)
        # f1 = -np.sum(self.prevention_df[x].intensity)
        f2 = -np.sum(self.prevention_df[x].bldg)
        f3 = -np.sum(self.prevention_df[x].habitat)
        f4 = -np.sum(self.prevention_df[x].hazard)

        out["F"] = [f2, f3, f4]
        out["G"] = np.sum(self.prevention_df[x].area) - self.n_max

class OneDimProblem(ElementwiseProblem):
    def __init__(self,
                 n_max,
                 rx_burn_units,
                 function_vals
                 ):
        super().__init__(n_var=rx_burn_units.shape[0], n_obj=1, n_constr=1)
        self.n_max = n_max
        self.rx_burn_units = rx_burn_units
        self.function_vals = function_vals

    def _evaluate(self, x, out, *args, **kwargs):
        
        out["F"] = -np.sum(self.function_vals[x])
        out["G"] = (self.n_max - np.sum(x)) ** 2