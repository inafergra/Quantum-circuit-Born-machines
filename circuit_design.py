import numpy as np
import cirq, sympy
from scipy.sparse.csgraph import minimum_spanning_tree

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
    '''Constructs the variational circuit, i.e., the ansatz.'''

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

