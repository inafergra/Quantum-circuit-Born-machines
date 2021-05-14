from scipy.optimize import minimize
from time import time
from functools import partial
import cirq, sympy
import matplotlib.pyplot as plt
import numpy as np

from functions import *
from cirq_design import * 

# TODO: generate training set to benchmark 

sparse_model = True # set True if we want to train a sparse model

## Ansatz hyperparameters
n_qubits = 4
shots = 2**14
lamda_list = np.linspace(0,1,11)

# Shallow QCBM
depth_shallow = 3
n_params_shallow = 2 * depth_shallow * n_qubits

# Deep QCBM
depth = 7
n_params = 2 * depth * n_qubits

## Target distribution : shallow QCBM with random parameters
np.random.seed(1)
theta_shallow = np.random.random(n_params_shallow)*2*np.pi
ansatz_shallow = variational_circuit(n_qubits, depth_shallow, theta_shallow)
target_distrib = estimate_probs(ansatz_shallow, theta_shallow, n_shots=shots)
#plt.plot(target_distrib)
#plt.show()

# Deep QCBM
theta_entry_symbols = [sympy.Symbol('theta_' + str(i)) for i in range(2 * n_qubits * depth)]
theta_symbol = sympy.Matrix(theta_entry_symbols)
ansatz_deep = variational_circuit(n_qubits, depth, theta_symbol)
#print(ansatz.to_text_diagram(transpose=True))

# MMD kernel
basis = np.arange(2**n_qubits)
sigma_list = [0.25,4]
kernel_matrix = multi_rbf_kernel(basis, basis, sigma_list)

# Initial theta
theta0 = np.random.random(n_params)*2*np.pi

# Callback function to track status 
final_loss = []
for lamda in lamda_list:
    # Initializing loss function with our ansatz, target and kernel matrix
    if sparse_model:
        # Set up a sparse model 
        loss_ansatz = partial(loss_l2, lamda=lamda, circuit=ansatz_deep, target=target_distrib, kernel_matrix=kernel_matrix, n_shots=shots)
        gradient_func = partial(gradient_l2, lamda=lamda, kernel_matrix=kernel_matrix, ansatz=ansatz_deep, target_distrib=target_distrib, n_shots=shots)
    else:
        # Set up a dense model
        loss_ansatz = partial(loss, lamda=lamda, circuit=ansatz_deep, target=target_distrib, kernel_matrix=kernel_matrix, n_shots=shots)
        gradient_func = partial(gradient, lamda=lamda, kernel_matrix=kernel_matrix, ansatz=ansatz_deep, target_distrib=target_distrib, n_shots=shots)
    
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
                            jac=gradient_func,
                            tol=10**-5, 
                            options={'maxiter':15, 'disp': 0, 'gtol':1e-10, 'ftol':0}, 
                            callback=callback)
    end_time = time() 
    print(f'It took {end_time-start_time} to train the QCBM')
    print(f'The final parameters are {final_params}')

    final_loss.append(tracking_cost[-1])

plt.plot(final_loss)
plt.plot(list(range(len(tracking_cost))), tracking_cost, 'g')
plt.title('Loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()


plt.plot(estimate_probs(ansatz_deep, final_params.x ,n_shots=shots), 'ro', label='Generated model')
plt.plot(target_distrib, 'bo', label='Input model')
plt.title('Generated model vs Input model')
plt.xlabel('Input data')
plt.ylabel('Probability')
plt.legend()
plt.show()
