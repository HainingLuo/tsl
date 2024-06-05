import numpy as np
from scipy.spatial import distance
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.termination.default import DefaultSingleObjectiveTermination


class OrderGaussianCentres(Problem):

    def __init__(self, centres):
        # centres: n x d
        self.n_cnetres = np.shape(centres)[0]
        self.centres = centres
        super().__init__(n_var=self.n_cnetres, n_obj=1, xl=0, xu=self.n_cnetres, vtype=float)

    def dist_sum(self, x):
        return np.sum([distance.euclidean(self.centres[x[i]], self.centres[x[i+1]]) for i in range(self.n_cnetres-1)])

    def _evaluate(self, xs, out, *args, **kwargs):
        out["F"] = [self.dist_sum(x) for x in xs]
        
def order_gaussian_centres(centres):

    problem = OrderGaussianCentres(centres)

    algorithm = GA(
        pop_size=50,
        sampling=PermutationRandomSampling(),
        mutation=InversionMutation(),
        crossover=OrderCrossover(),
        eliminate_duplicates=True
    )

    # if the algorithm did not improve the last 200 generations then it will terminate (and disable the max generations)
    termination = DefaultSingleObjectiveTermination(period=500, n_max_gen=np.inf)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
    )
    # rotate

    return res.X

if __name__ == "__main__":
    length = 1
    n_points = 10
    start_pos = [0, -length/2, 0]
    end_pos = [0, length/2, 0]
    test_states = [start_pos]*n_points # (num_particles*dim)
    for pi in range(n_points):
        test_states[pi] = [(n_points-pi)*a/(n_points-1) + pi*b/(n_points-1) for a, b in zip(start_pos, end_pos)]
    
    res = order_gaussian_centres(test_states)