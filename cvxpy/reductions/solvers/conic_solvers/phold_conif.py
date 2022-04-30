from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.constraints import Zero, NonNeg, SOC
import cvxpy.settings as s


class PHOLD(Solver):

    # Solver capabilities
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg, SOC]

    def accepts(self, problem) -> bool:
        return len(problem.variables()) == 0

    def apply(self, problem):
        return problem, []

    def invert(self, solution, inverse_data):
        return solution

    def name(self) -> str:
        return s.PHOLD

    def import_solver(self) -> None:
        return

    def is_installed(self) -> bool:
        return True

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        return self.solve(data, warm_start, verbose, solver_opts)

    def solve(self, problem, warm_start: bool, verbose: bool, solver_opts):
        if all(c.value() for c in problem.constraints):
            return Solution(s.OPTIMAL, problem.objective.value, {}, {}, {})
        else:
            return Solution(s.INFEASIBLE, None, {}, {}, {})
