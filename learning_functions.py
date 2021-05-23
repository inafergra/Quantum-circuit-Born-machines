import cirq, sympy
import matplotlib.pyplot as plt
import numpy as np
import pydeep.base.numpyextension as npext
from scipy.stats import entropy

def estimate_probs(circuit, theta, n_shots):
    ''' Estimate all probabilities of the PQCs distribution.'''

    n_qubits = len(circuit.all_qubits())

    # Creating parameter resolve dict by adding state and theta.
    try:
        theta_mapping = [('theta_' + str(i), theta[i]) for i in range(len(theta))]
    except IndexError as error:
        print("Could not resolve theta symbol, array of wrong size.")
    resolve_dict = dict(theta_mapping)
    resolver = cirq.ParamResolver(resolve_dict)
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)
    
    # Run the circuit.
    if n_shots==0:
        final_state = resolved_circuit.final_state_vector()
        probs = np.array([np.abs(final_state[i])**2 for i in range(len(final_state))])
    else:
        results = cirq.sample(resolved_circuit, repetitions=n_shots)
        frequencies = results.histogram(key='m')
        probs = np.zeros(2**n_qubits)
        for key, value in frequencies.items():
            probs[key] = value / n_shots
    return probs

def multi_rbf_kernel(x, y, sigma_list):
    '''
    Function that computes the kernel for the MMD loss (multi-RBF kernel).
    
    Args:
        x (1darray|2darray): the collection of samples A.
        x (1darray|2darray): the collection of samples B.
        sigma_list (list): a list of bandwidths.
        
    Returns:
        2darray: kernel matrix.
    '''
    ndim = x.ndim
    if ndim == 1:
        exponent = np.abs(x[:, None] - y[None, :])**2
    elif ndim == 2:
        exponent = ((x[:, None, :] - y[None, :, :])**2).sum(axis=2)
    else:
        raise
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma)
        K = K + np.exp(-gamma * exponent)
    return K

def kernel_expectation(px, py, kernel_matrix):
    '''Function that computes expectation of kernel in MMD loss'''
    return px.dot(kernel_matrix).dot(py)

def squared_MMD_loss(probs, target, kernel_matrix):
    '''Function that computes the squared MMD loss related to the given kernel_matrix.'''
    dif_probs = probs - target
    return kernel_expectation(dif_probs,dif_probs,kernel_matrix)

def loss(theta, lamda, circuit, target, kernel_matrix, n_shots, norm = 'L2'):
    probs = estimate_probs(circuit, theta, n_shots=n_shots)
    if norm == 'L2':
        regul_term = np.sqrt(np.sum(np.square(theta)))
    elif norm == 'L0':
        regul_term = np.count_nonzero(theta)
    else:
        regul_term = 0
    return squared_MMD_loss(probs, target, kernel_matrix) + lamda*regul_term

def gradient(theta, lamda, kernel_matrix, ansatz, target, n_shots, norm = 'L2'):
    '''Get gradient of the loss with the L2 regularization term'''
    prob = estimate_probs(ansatz, theta, n_shots=n_shots)
    grad = []
    for i in range(len(theta)):
        # regularization term
        if norm == 'L2':
            grad_regul = abs(theta[i]) /  np.sqrt(np.sum(np.square(theta)))

        elif norm == 'L0':
            grad_regul = 0
        else:
            grad_regul = 0
        # pi/2 phase
        theta[i] += np.pi/2.
        prob_pos = estimate_probs(ansatz, theta, n_shots=n_shots)
        # -pi/2 phase
        theta[i] -= np.pi
        prob_neg = estimate_probs(ansatz, theta, n_shots=n_shots)
        # recover
        theta[i] += np.pi/2.
        grad_pos = kernel_expectation(prob, prob_pos, kernel_matrix) - kernel_expectation(prob, prob_neg, kernel_matrix)
        grad_neg = kernel_expectation(target, prob_pos, kernel_matrix) - kernel_expectation(target, prob_neg, kernel_matrix)

        grad.append(grad_pos - grad_neg + lamda*grad_regul)
        
    return np.array(grad)

def relative_entropy(P,Q):
    '''Calculates the relative entropy of the target distribution P and the generated distribution Q'''
    return  entropy(P, qk=Q) #np.sum(P*np.log(P/Q))


