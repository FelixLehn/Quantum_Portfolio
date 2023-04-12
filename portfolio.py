from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider, WikipediaDataProvider
from qiskit_finance import QiskitFinanceError
from qiskit_optimization.algorithms import MinimumEigenOptimizer

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from dotenv import load_dotenv
import nasdaqdatalink

load_dotenv()

TOKEN = os.getenv('TOKEN')

# set number of assets (= number of qubits)
num_assets = 4
seed = 123

''' RANDOMIZED'''
# # Generate expected return and covariance matrix from (random) time-series
# stocks = [("TICKER%s" % i) for i in range(num_assets)]
# data = RandomDataProvider(
#     tickers=stocks,
#     start=datetime.datetime(2016, 1, 1),
#     end=datetime.datetime(2016, 1, 30),
#     seed=seed,
# )
# data.run()
# mu = data.get_period_return_mean_vector()
# sigma = data.get_period_return_covariance_matrix()

# # plot sigma
# plt.imshow(sigma, interpolation="nearest")
# plt.show()

'''REAL DATA'''
###REAL DATA
stocks = ["GOOG", "AAPL", "MSFT", "ABT"]

token = TOKEN
if token == TOKEN:
    try:
        wiki = WikipediaDataProvider(
            token=token,
            tickers=stocks,
            start=datetime.datetime(2018, 1, 1),
            end=datetime.datetime(2018, 12, 30),
        )
        wiki.run()
    except QiskitFinanceError as ex:
        print(ex)
        print("Error retrieving data.")

if token == TOKEN:
    if wiki._data:
        if wiki._n <= 1:
            print(
                "Not enough wiki data to plot covariance or time-series similarity. Please use at least two tickers."
            )
        else:
            rho = wiki.get_similarity_matrix()
            print("A time-series similarity measure:")
            print(rho)
            plt.imshow(rho)
            plt.show()

            mu = wiki.get_period_return_mean_vector()
            print("time-series MEAN:")
            print(mu)

            sigma = wiki.get_period_return_covariance_matrix()
            print("A covariance matrix:")
            print(sigma)
            plt.imshow(sigma)
            plt.show()
    else:
        print("No wiki data loaded.")

if token == TOKEN:
    if wiki._data:
        print("The underlying evolution of stock prices:")
        for (cnt, s) in enumerate(stocks):
            plt.plot(wiki._data[cnt], label=s)
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()

        for (cnt, s) in enumerate(stocks):
            print(s)
            print(wiki._data[cnt])
    else:
        print("No wiki data loaded.")

q = 0.5  # set risk factor
budget = num_assets // 2  # set budget
penalty = num_assets  # set parameter to scale the budget penalty term

portfolio = PortfolioOptimization(
    expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
)
qp = portfolio.to_quadratic_program()
qp

def print_result(result, kind):
    selection = result.x
    value = result.fval
    print("-----{}-----".format(kind))
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))

exact_mes = NumPyMinimumEigensolver()
exact_eigensolver = MinimumEigenOptimizer(exact_mes)

result = exact_eigensolver.solve(qp)

print_result(result,"Classical")

### SAMPLING VQE
from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 1234

cobyla = COBYLA()
cobyla.set_options(maxiter=500)
ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
vqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
vqe = MinimumEigenOptimizer(vqe_mes)
result = vqe.solve(qp)

print_result(result, "Quantum SamplingVQE")

### QAOA
algorithm_globals.random_seed = 1234

cobyla = COBYLA()
cobyla.set_options(maxiter=250)
qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

print_result(result, "Quantum QAOA")