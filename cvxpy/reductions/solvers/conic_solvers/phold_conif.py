from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers import utilities
from cvxpy.constraints import Zero, NonNeg, SOC
import cvxpy.settings as s


# Utility method for formatting a ConeDims instance into a dictionary
def dims_to_solver_dict(cone_dims):
    cones = {
        'l': cone_dims.nonneg,
        'q': cone_dims.soc,
    }
    return cones


class PHOLD(ConicSolver):
    """ Conic Solver interface for the PHOLD solver """

    # Solver capabilities
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg, SOC]

    # Map of PHOLD status flags to CVXPY status
    STATUS_MAP = { 1: s.OPTIMAL,
                   2: s.OPTIMAL_INACCURATE,
                  -2: s.SOLVER_ERROR,           # Maxiter reached
                  -3: s.INFEASIBLE,
                   3: s.INFEASIBLE_INACCURATE,
                  -4: s.UNBOUNDED,              # dual infeasible
                   4: s.UNBOUNDED_INACCURATE,   # dual infeasible inaccurate
                  -6: s.USER_LIMIT,             # time limit reached
                  -5: s.SOLVER_ERROR,           # Interrupted by user
                 -10: s.SOLVER_ERROR}           # Unsolved

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.
        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        return super(PHOLD, self).apply(problem)

    def invert(self, solution, inverse_data):
        """ Returns the solution to the original problem given the inverse_data """
        attr = {
            s.SETUP_TIME  : solution.info.timings.setup_time,
            s.SOLVE_TIME  : solution.info.timings.solve_time,
            s.NUM_ITERS   : solution.info.num_iters,
            s.EXTRA_STATS : solution,
            }

        # Map PHOLD status flags back to CVXPY status flags
        status = self.STATUS_MAP[solution.info.exitFlag]

        if status in s.SOLUTION_PRESENT:
            # get the objective value of the problem
            opt_val = solution.info.obj_val + inverse_data[s.OFFSET]
            # get the primal values
            primal_vars = {inverse_data[PHOLD.VAR_ID]: solution['x']}
            # get the dual values
            eq_dual_vars   = utilities.get_dual_values(
                solution['y'][:inverse_data[ConicSolver.DIMS].zero],
                self.extract_dual_value,
                inverse_data[PHOLD.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution['y'][inverse_data[ConicSolver.DIMS].zero:],
                self.extract_dual_value,
                inverse_data[PHOLD.NEQ_CONSTR]
            )
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def name(self) -> str:
        """ return the name of the solver """
        return s.PHOLD

    def import_solver(self) -> None:
        """ import the solver """
        import phold
        phold

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        import phold
        # input problem data
        problem_data  = {"A": data[s.A], "b": data[s.B], "c": data[s.C]}
        # warmstart data (if available)
        if warm_start and solver_cache is not None and self.name() in solver_cache:
            problem_data['x'] = solver_cache[self.name()]['x']
            problem_data['y'] = solver_cache[self.name()]['y']
        # cones data for the solver
        cones_data = dims_to_solver_dict(data[ConicSolver.DIMS])

        # Overwrite defaults eps_abs=eps_rel=1e-3, max_iter=4000
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-5)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-5)
        solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)

        # invoke the solver
        solution = phold.solve(problem_data, cones_data, verbose=verbose, **solver_opts)
        status   = self.STATUS_MAP[solution.info.exitFlag]

        # update the solver cache if the solution is optimal
        if solver_cache is not None and status == s.OPTIMAL:
            solver_cache[self.name()] = solution
        
        # return the solver output
        return solution