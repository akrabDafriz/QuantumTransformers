from typing import Callable

import tensorcircuit as tc
import jax.numpy as jnp
import flax.linen

K = tc.set_backend("jax")


def angle_embedding(c: tc.Circuit, inputs):
    num_qubits = inputs.shape[-1]

    for j in range(num_qubits):
        c.rx(j, theta=inputs[j])


# A basic variational quantum circuit (VQC) with RX rotations and CNOT entanglement
def basic_vqc(c: tc.Circuit, inputs, weights):
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2]

    for i in range(num_qlayers):
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, j])
        if num_qubits == 2:
            c.cnot(0, 1)
        elif num_qubits > 2:
            for j in range(num_qubits):
                c.cnot(j, (j + 1) % num_qubits)
## Core idea of Daffa's thesis is to implement and compare different types of VQC in the quantum layer, and how they affect the model performance. Therefore, I am going to make the VQC modular so that we can easily swap different types of VQC.
## The current VQC is a simple one with RX rotations and CNOT entanglement. I will try to implement other types of VQC such as CZ entangleme complex VQC such as hardware-efficient ansatz, QAOA, etc.

#########################################################
#################### Custom VQCs ########################
#########################################################

# Custom VQC Design 1: RY rotations + CNOT circular entanglements + CNOT circular entanglement
def vqc_ry_cnot(c: tc.Circuit, inputs, weights):
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2] # L

    for i in range(num_qlayers):
        # 1. Rotation Layer
        for j in range(num_qubits):
            c.ry(j, theta=weights[i, j])
        # 2. Entanglement Layer
        for j in range(num_qubits):
            c.cnot(j, (j + 1) % num_qubits)

# Custom VQC Design 2: Hadamard + RX rotations + CNOT circular entanglement
def vqc_h_rx_cnot(c: tc.Circuit, inputs, weights):
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2] # L

    for i in range(num_qlayers):
        # 1. Basis Change Layer
        for j in range(num_qubits):
            c.h(j)
        # 2. Rotation Layer
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, j])
        # 3. Entanglement Layer
        for j in range(num_qubits):
            c.cnot(j, (j + 1) % num_qubits)

# Custom VQC Design 3: Full-rotation (RX,RY,RZ) + CNOT circular entanglement
def vqc_all_rot_cnot(c: tc.Circuit, inputs, weights):
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-3] # L (shape is L, 3, num_qubits)

    for i in range(num_qlayers):
        # 1. Rotation Layer (uses 3 params per qubit)
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, 0, j])
            c.ry(j, theta=weights[i, 1, j])
            c.rz(j, theta=weights[i, 2, j])
        # 2. Entanglement Layer
        for j in range(num_qubits):
            c.c# Custom VQC Design 4: RX rotations + CRX circular entanglement
def vqc_rx_crx(c: tc.Circuit, inputs, weights):
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-3] # L (shape is L, 2, num_qubits)

    for i in range(num_qlayers):
        # 1. Rotation Layer (uses 1st param)
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, 0, j])
        # 2. Entanglement Layer (uses 2nd param)
        for j in range(num_qubits):
            c.crx(j, (j + 1) % num_qubits, theta=weights[i, 1, j])

####################################################################

def get_quantum_layer_circuit(inputs, weights,
                              embedding: Callable = angle_embedding, vqc: Callable = basic_vqc):
    """
    Equivalent to the following PennyLane circuit:
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
    """

    num_qubits = inputs.shape[-1]

    c = tc.Circuit(num_qubits)
    embedding(c, inputs)
    vqc(c, inputs, weights)

    return c


def get_circuit(embedding: Callable = angle_embedding, vqc: Callable = basic_vqc,
                torch_interface: bool = False):
    def qpred(inputs, weights):
        c = get_quantum_layer_circuit(inputs, weights, embedding, vqc)
        return K.real(jnp.array([c.expectation_ps(z=[i]) for i in range(weights.shape[1])]))

    qpred_batch = K.vmap(qpred, vectorized_argnums=0)
    if torch_interface:
        qpred_batch = tc.interfaces.torch_interface(qpred_batch, jit=True)

    return qpred_batch



class QuantumLayer(flax.linen.Module):
    circuit: Callable
    num_qubits: int
    w_shape: tuple = (1,)

    @flax.linen.compact
    def __call__(self, x):
        shape = x.shape
        x = jnp.reshape(x, (-1, shape[-1]))
        w = self.param('w', flax.linen.initializers.xavier_normal(), self.w_shape + (self.num_qubits,))
        x = self.circuit(x, w)
        x = jnp.concatenate(x, axis=-1)
        x = jnp.reshape(x, tuple(shape))
        return x      x = jnp.concatenate(x, axis=-1)
        x = jnp.reshape(x, tuple(shape))
        return x