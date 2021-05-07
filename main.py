from scipy.optimize import minimize
from time import time
from functools import partial
import cirq, sympy
import matplotlib.pyplot as plt
import numpy as np

from functions import *

## Ansatz hyperparameters
n_qubits = 3
shots = 2**14

# Shallow QCBM
depth_shallow = 2
n_params_shallow = 2 * depth * n_qubits

# Deep QCBM
depth_deep = 5
n_params_deep = 2 * depth_deep * n_qubits


## Target distribution : shallow QCBM with random parameters
np.random.seed(1)
theta_shallow = np.random.random(n_params)*2*np.pi
ansatz_shallow = variational_circuit(n_qubits, depth, theta_shallow)
pg = estimate_probs(ansatz_shallow, theta_shallow, n_shots=shots)
#plt.plot(pg)
#plt.show()

# Deep QCBM
theta_entry_symbols = [sympy.Symbol('theta_' + str(i)) for i in range(2 * n_qubits * depth)]
theta_symbol = sympy.Matrix(theta_entry_symbols)
ansatz = variational_circuit(n_qubits, depth, theta_symbol)
print(ansatz.to_text_diagram(transpose=True))

# MMD kernel
basis = np.arange(2**n_qubits)
sigma_list = [0.25,4]
kernel_matrix = multi_rbf_kernel(basis, basis, sigma_list)

# Initial theta
theta0 = np.random.random(n_params)*2*np.pi

# Initializing loss function with our ansatz, target and kernel matrix
# Now loss ansatz is another function were we fixed the values circuit, target and kernel_matrix, 
# so it only takes theta (nice to put into the minimizer)
loss_ansatz = partial(loss, circuit=ansatz, target=pg, kernel_matrix=kernel_matrix)

# Callback function to track status 
step = [0]
tracking_cost = []
def callback(x, *args, **kwargs):
    step[0] += 1
    tracking_cost.append(loss_ansatz(x))
    print('step = %d, loss = %s'%(step[0], loss_ansatz(x)))

# Training the QCBM.
start_time = time()
final_params = minimize(loss_ansatz,
                        theta0,
                        method="L-BFGS-B", 
                        jac=partial(gradient, kernel_matrix=kernel_matrix, ansatz=ansatz, pg=pg),
                        tol=10**-5, 
                        options={'maxiter':20, 'disp': 0, 'gtol':1e-10, 'ftol':0}, 
                        callback=callback)
end_time = time() 
print(f'It took {end_time-start_time} to train the QCBM')

plt.plot(list(range(len(tracking_cost))), tracking_cost)
plt.show()
plt.plot(estimate_probs(ansatz, final_params.x), 'ro')
plt.show()
