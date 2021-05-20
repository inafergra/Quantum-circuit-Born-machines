from scipy.optimize import minimize
from time import time
from functools import partial
import cirq, sympy
import matplotlib.pyplot as plt
import numpy as np

from learning_functions import *
from circuit_design import * 
from utils import * 


## Model hyperparameters
n_qubits = 4
depth = 7
n_params = 2 * depth * n_qubits
shots = 0
lamda_list = np.linspace(0,.01,11)

training_distribution = generate_bas_complete(int(np.sqrt(n_qubits)))
print(training_distribution)
print(relative_entropy(training_distribution,training_distribution))
#plt.plot(training_distribution)
#plt.show()

# Generate the QCBM circuit
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
training_entropy_list = []
#test_entropy_list = []
for lamda in lamda_list:
    print(f'Lambda = {lamda}')

    # Set up the regularized model 
    training_loss = partial(loss_l2, lamda=lamda, circuit=ansatz_deep, target=training_distribution, kernel_matrix=kernel_matrix, n_shots=shots)
    #test_loss = partial(loss_l2, lamda=lamda, circuit=ansatz_deep, target=test_distribution, kernel_matrix=kernel_matrix, n_shots=shots)
    gradient_func = partial(gradient_l2, lamda=lamda, kernel_matrix=kernel_matrix, ansatz=ansatz_deep, target=training_distribution, n_shots=shots)

    # Set up the callback to track the losses
    step = [0]
    training_loss_list = []
    #test_loss_list = []
    def callback(x, *args, **kwargs):
        step[0] += 1
        training_loss_list.append(training_loss(x))
        #test_loss_list.append(test_loss(x))
        print(f'step = {step[0]}')
        print(f'Training loss {training_loss(x)}')
        #print(f'Test loss {test_loss(x)}')

    # Training the QCBM.
    start_time = time()
    final_params = minimize(training_loss,
                            theta0,
                            method="L-BFGS-B", 
                            jac=gradient_func,
                            tol=10**-5, 
                            options={'maxiter':30, 'disp': 0, 'gtol':1e-10, 'ftol':0}, 
                            callback=callback)
    end_time = time()
    print(f'It took {end_time-start_time} to train the QCBM')
    #print(f'The final parameters are {final_params}')

    # Generated distribution
    generated_distribution = estimate_probs(ansatz_deep, final_params.x ,n_shots=shots)
    print(generated_distribution)
    print(f'Norm of the distribution {np.sum(generated_distribution)}')

    # Benchmark training set
    training_entropy = relative_entropy(target_distribution, generated_distribution)
    training_entropy_list.append(training_entropy)

    # Benchmark test set
    #test_entropy = relative_entropy(test_distribution, generated_distribution)
    #test_entropy_list.append(test_entropy)

    print(f'Relative entropy with the training set {training_entropy}')
    #print(f'Relative entropy with the test set {test_entropy}')


plt.plot(lamda_list, training_entropy_list, label = 'Training set')
#plt.plot(lamda_list, test_entropy_list, label = 'Test set')
#plt.title('Relative entropy vs lambda')
plt.xlabel('Lambda')
plt.ylabel('Relative entropy')
plt.legend()
plt.show()

plt.plot(list(range(len(training_loss_list))), training_loss_list, label = 'Training set')
#plt.plot(list(range(len(test_loss_list))), test_loss_list, label = 'Test set')
plt.title('Loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(generated_distribution, 'ro', label='Generated model')
plt.plot(target_distribution, 'bo', label='Input model')
plt.title('Generated model vs Input model')
plt.xlabel('Input data')
plt.ylabel('Probability')
plt.legend()
plt.show()
