import cirq, sympy
import matplotlib.pyplot as plt
import numpy as np
#Implementing the ansatz

n_qubits = 3
depth = 2
n_params = 2 * depth * n_qubits
shots = 2**14

def rot_z_layer(n_qubits, parameters):
    '''Layer of single qubit z rotations'''
    if n_qubits != len(parameters):
        raise ValueError("Too many or few parameters, must equal n_qubits")
    for i in range(n_qubits):
        yield cirq.rz(2 * parameters[i])(cirq.GridQubit(i, 0))


def rot_y_layer(n_qubits, parameters):
    '''Layer of single qubit y rotations'''
    if n_qubits != len(parameters):
        raise ValueError("Too many of few parameters, must equal n_qubits")
    for i in range(n_qubits):
        yield cirq.ry(parameters[i])(cirq.GridQubit(i, 0))


def entangling_layer(n_qubits):
    '''Layer of entangling CZ(i,i+1 % n_qubits) gates.'''
    if n_qubits == 2:
        yield cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))
        return
    for i in range(n_qubits):
        yield cirq.CZ(cirq.GridQubit(i, 0), cirq.GridQubit((i+1) % n_qubits, 0))

def variational_circuit(n_qubits, depth, theta):
    '''Variational circuit, i.e., the ansatz.'''

    if len(theta) != (2 * depth * n_qubits):
        raise ValueError("Theta of incorrect dimension, must equal 2*depth*n_qubits")
    
    # Initializing qubits and circuit
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    
    # Adding layers of rotation gates and entangling gates.
    for d in range(depth):
        # Adding single qubit rotations
        circuit.append(rot_z_layer(n_qubits, theta[d * 2 * n_qubits : (d+1) * 2 * n_qubits : 2]))
        circuit.append(rot_y_layer(n_qubits, theta[d * 2 * n_qubits + 1 : (d+1) * 2 * n_qubits + 1 : 2]))
        # Adding entangling layer
        circuit.append(entangling_layer(n_qubits))
    
    # Adding measurement at the end.
    circuit.append(cirq.measure(*qubits, key='m'))
    return circuit


def estimate_probs(circuit, theta, n_shots=shots):
    ''' Estimate all probabilities of the PQCs distribution.'''

    # Creating parameter resolve dict by adding state and theta.
    try:
        theta_mapping = [('theta_' + str(i), theta[i]) for i in range(len(theta))]
    except IndexError as error:
        print("Could not resolve theta symbol, array of wrong size.")
    resolve_dict = dict(theta_mapping)
    resolver = cirq.ParamResolver(resolve_dict)
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)
    
    # Run the circuit.
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

def loss(theta, circuit, target, kernel_matrix, n_shots=shots):
    '''The loss function that we aim to minimize.'''
    probs = estimate_probs(circuit, theta, n_shots=n_shots)
    return squared_MMD_loss(probs, target, kernel_matrix)

def gradient(theta, kernel_matrix, ansatz, pg, n_shots=shots):
    '''Get gradient'''
    prob = estimate_probs(ansatz, theta, n_shots=shots)
    grad = []
    for i in range(len(theta)):
        # pi/2 phase
        theta[i] += np.pi/2.
        prob_pos = estimate_probs(ansatz, theta, n_shots=shots)
        # -pi/2 phase
        theta[i] -= np.pi
        prob_neg = estimate_probs(ansatz, theta, n_shots=shots)
        # recover
        theta[i] += np.pi/2.
        grad_pos = kernel_expectation(prob, prob_pos, kernel_matrix) - kernel_expectation(prob, prob_neg, kernel_matrix)
        grad_neg = kernel_expectation(pg, prob_pos, kernel_matrix) - kernel_expectation(pg, prob_neg, kernel_matrix)
        grad.append(grad_pos - grad_neg)
    return np.array(grad)