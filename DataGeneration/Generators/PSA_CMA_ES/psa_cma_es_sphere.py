import os
import sys
import ioh
import numpy as np
import csv

# Add ModularCMAES to path (assume it's now a sibling directory or update as needed)
modularcmaes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'modcma'))
sys.path.append(modularcmaes_path)
from modcma import ModularCMAES

def run_psa_cmaes_sphere():
    dim = 10
    budget_factor = 2500
    reps = 1  # Only one repetition for Sphere
    algo = 'psa'
    function_id = 1  # Sphere function

    problem = ioh.get_problem(
        fid=function_id,
        instance=1,
        dimension=dim,
        problem_class=ioh.ProblemClass.BBOB
    )

    csv_dir = os.path.join(os.path.dirname(__file__), f'psa-test-data/{algo}-fid{function_id}-{dim}D')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'psa_vars_rep1.csv')
    print("Saving CSV to:", csv_path)
    try:
        cma = ModularCMAES(
            problem,
            dim,
            budget=dim*budget_factor,
            pop_size_adaptation='psa',
            min_lambda_=4,
            max_lambda_=512,
            x0=np.zeros((dim, 1)),
            lb=np.full((dim, 1), -5),
            ub=np.full((dim, 1), 5),
        )
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'generation', 'lambda', 'psa_beta', 'ptnorm', 'alpha', 'update_term', 'exp_update_value', 'next_lambda_unrounded'
            ])
            cma.step()
            print("First population fitnesses:", cma.parameters.population.f)
            generation = cma.parameters.t
            lambda_ = cma.parameters.lambda_
            psa_beta = cma.parameters.psa_beta
            ptnorm = cma.parameters.ptnorm
            alpha = cma.parameters.alpha
            update_term = 1 - (ptnorm / alpha)
            exp_update_value = psa_beta * update_term
            next_lambda_unrounded = lambda_ * np.exp(exp_update_value)
            writer.writerow([
                generation, lambda_, psa_beta, ptnorm, alpha, update_term, exp_update_value, next_lambda_unrounded
            ])
            for _ in range(9):  # 9 more steps for a total of 10
                cma.step()
                generation = cma.parameters.t
                lambda_ = cma.parameters.lambda_
                psa_beta = cma.parameters.psa_beta
                ptnorm = cma.parameters.ptnorm
                alpha = cma.parameters.alpha
                update_term = 1 - (ptnorm / alpha)
                exp_update_value = psa_beta * update_term
                next_lambda_unrounded = lambda_ * np.exp(exp_update_value)
                writer.writerow([
                    generation, lambda_, psa_beta, ptnorm, alpha, update_term, exp_update_value, next_lambda_unrounded
                ])
    except Exception as e:
        print("Error writing CSV:", e)
    print(f'fid={function_id}/Sphere, rep=1/1, best y={problem.state.current_best.y}')
    problem.reset()

if __name__ == "__main__":
    run_psa_cmaes_sphere() 