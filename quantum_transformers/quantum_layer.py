from typing import Callable, Tuple
import tensorcircuit as tc
import jax.numpy as jnp
import flax.linen as nn

K = tc.set_backend("jax")

def angle_embedding(c: tc.Circuit, inputs):
    """Encodes input data into quantum states using RX rotations."""
    num_qubits = inputs.shape[-1]
    for j in range(num_qubits):
        c.rx(j, theta=inputs[j])

# --- Custom VQC Designs ---

# Base Design: Simple RX + CNOT
def basic_vqc(c: tc.Circuit, inputs, weights):
    """
    Standard VQC with RX rotations and CNOT entanglement.
    Weights shape: (num_layers, num_qubits)
    """
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

# Design 1: RY + CNOT (Rotational Complexity 1)
def vqc_ry_cnot(c: tc.Circuit, inputs, weights):
    """
    VQC with RY rotations and CNOT entanglement.
    Weights shape: (num_layers, num_qubits)
    """
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2]

    for i in range(num_qlayers):
        # 1. Rotation Layer
        for j in range(num_qubits):
            c.ry(j, theta=weights[i, j])
        # 2. Entanglement Layer
        if num_qubits == 2:
            c.cnot(0, 1)
        elif num_qubits > 2:
            for j in range(num_qubits):
                c.cnot(j, (j + 1) % num_qubits)

# Design 2: Hadamard + RX + CNOT (Basis Change)
def vqc_h_rx_cnot(c: tc.Circuit, inputs, weights):
    """
    VQC with Hadamard basis change, RX rotations, and CNOT entanglement.
    Weights shape: (num_layers, num_qubits)
    """
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2]

    for i in range(num_qlayers):
        # 1. Basis Change Layer
        for j in range(num_qubits):
            c.h(j)
        # 2. Rotation Layer
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, j])
        # 3. Entanglement Layer
        if num_qubits == 2:
            c.cnot(0, 1)
        elif num_qubits > 2:
            for j in range(num_qubits):
                c.cnot(j, (j + 1) % num_qubits)

# Design 3: Full Rotations (RX, RY, RZ) + CNOT (Rotational Complexity 2)
def vqc_all_rot_cnot(c: tc.Circuit, inputs, weights):
    """
    VQC with full single-qubit rotations (RX, RY, RZ) and CNOT entanglement.
    Weights shape: (num_layers, 3, num_qubits)
    """
    num_qubits = inputs.shape[-1]
    # Expect weights to be (Layers, 3, Qubits)
    num_qlayers = weights.shape[-3] 

    for i in range(num_qlayers):
        # 1. Rotation Layer (uses 3 params per qubit)
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, 0, j])
            c.ry(j, theta=weights[i, 1, j])
            c.rz(j, theta=weights[i, 2, j])
        # 2. Entanglement Layer
        if num_qubits == 2:
            c.cnot(0, 1)
        elif num_qubits > 2:
            for j in range(num_qubits):
                c.cnot(j, (j + 1) % num_qubits)

# Design 4: RX + Controlled-RX (Trainable Entanglement)
def vqc_rx_crx(c: tc.Circuit, inputs, weights):
    """
    VQC with RX rotations and trainable Controlled-RX (CRX) entanglement.
    Weights shape: (num_layers, 2, num_qubits)
    """
    num_qubits = inputs.shape[-1]
    # Expect weights to be (Layers, 2, Qubits)
    num_qlayers = weights.shape[-3]

    for i in range(num_qlayers):
        # 1. Rotation Layer (uses 1st param)
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, 0, j])
        # 2. Entanglement Layer (uses 2nd param)
        if num_qubits == 2:
             c.crx(0, 1, theta=weights[i, 1, 0])
        elif num_qubits > 2:
            for j in range(num_qubits):
                # Use the 2nd parameter set for CRX gates
                c.crx(j, (j + 1) % num_qubits, theta=weights[i, 1, j])


def get_quantum_layer_circuit(inputs, weights,
                              embedding: Callable = angle_embedding, vqc: Callable = basic_vqc):
    """
    Constructs the TensorCircuit object using the provided embedding and VQC functions.
    """
    num_qubits = inputs.shape[-1]
    c = tc.Circuit(num_qubits)
    embedding(c, inputs)
    vqc(c, inputs, weights)
    return c


def get_circuit(embedding: Callable = angle_embedding, vqc: Callable = basic_vqc,
                torch_interface: bool = False):
    """
    Returns a vectorized JAX function that executes the quantum circuit.
    """
    def qpred(inputs, weights):
        c = get_quantum_layer_circuit(inputs, weights, embedding, vqc)
        # We measure the expectation of Z on all qubits (conceptually similar to measurement)
        # If the weight shape is complex (e.g. Design 3), we need to ensure we map output correctly.
        # Here we just return Z expectation on all qubits, which matches input dimension (num_qubits).
        # We assume weights.shape[1] is NOT used for determining output size, 
        # but inputs.shape[-1] (num_qubits) is the output size.
        num_qubits = inputs.shape[-1]
        return K.real(jnp.array([c.expectation_ps(z=[i]) for i in range(num_qubits)]))

    qpred_batch = K.vmap(qpred, vectorized_argnums=0)
    if torch_interface:
        qpred_batch = tc.interfaces.torch_interface(qpred_batch, jit=True)

    return qpred_batch


class QuantumLayer(nn.Module):
    """
    A Flax Linen Module wrapping the quantum circuit.
    """
    num_qubits: int
    w_shape: tuple = (1,) # Shape of weights (e.g., (layers,) or (layers, 3))
    circuit: Callable = get_circuit()

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        # Flatten input to (batch * seq_len, hidden_size)
        x = jnp.reshape(x, (-1, shape[-1]))
        
        # Initialize weights. 
        # The final weight shape will be self.w_shape + (self.num_qubits,)
        # e.g., if w_shape is (2, 3), weights will be (2, 3, num_qubits)
        w = self.param('w', nn.initializers.xavier_normal(), self.w_shape + (self.num_qubits,))
        
        # Execute circuit
        x = self.circuit(x, w)
        
        # Reshape back to original dimensions
        # x comes out as (batch * seq_len, num_qubits)
        # We assume num_qubits == hidden_size
        x = jnp.reshape(x, tuple(shape))
        return x