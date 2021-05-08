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



def chowliu_tree(pdata):
    '''
    generate chow-liu tree.
    Args:
        pdata (1darray): empirical distribution in dataset
    Returns:
        list: entangle pairs.
    '''
    X = mutual_information(pdata)
    Tcsr = -minimum_spanning_tree(-X)
    Tcoo = Tcsr.tocoo()
    pairs = list(zip(Tcoo.row, Tcoo.col))
    print('Chow-Liu tree pairs = %s'%pairs)
    return pairs

def mutual_information(pdata):
    '''
    calculate mutual information I = \sum\limits_{x,y} p(x,y) log[p(x,y)/p(x)/p(y)]
    Args:
        pdata (1darray): empirical distribution in dataset
    Returns:
        2darray: mutual information table.
    '''
    sl = [0, 1]  # possible states
    d = len(sl)  # number of possible states
    num_bit = int(np.round(np.log(len(pdata))/np.log(2)))
    basis = np.arange(2**num_bit, dtype='uint32')

    pxy = np.zeros([num_bit, num_bit, d, d])
    px = np.zeros([num_bit, d])
    pdata2d = np.broadcast_to(pdata[:,None], (len(pdata), num_bit))
    pdata3d = np.broadcast_to(pdata[:,None,None], (len(pdata), num_bit, num_bit))
    offsets = np.arange(num_bit-1,-1,-1)

    for s_i in sl:
        mask_i = (basis[:,None]>>offsets)&1 == s_i
        px[:,s_i] = np.ma.array(pdata2d, mask=~mask_i).sum(axis=0)
        for s_j in sl:
            mask_j = (basis[:,None]>>offsets)&1 == s_j
            pxy[:,:,s_i,s_j] = np.ma.array(pdata3d, mask=~(mask_i[:,None,:]&mask_j[:,:,None])).sum(axis=0)

    # mutual information
    pratio = pxy/np.maximum(px[:,None,:,None]*px[None,:,None,:], 1e-15)
    for i in range(num_bit):
        pratio[i, i] = 1
    I = (pxy*np.log(pratio)).sum(axis=(2,3))
    return I