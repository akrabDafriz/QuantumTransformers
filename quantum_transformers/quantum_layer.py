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
## The current VQC is a simple one with RX rotations and CNOT entanglement. I will try to implement other types of VQC such as CZ entanglement, CRX entanglement, and more complex VQC such as hardware-efficient ansatz, QAOA, etc.


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

# I am asking my github copilot about this part: "vqc: Callable = basic_vqc". Explain what it means!!
# It means that the function get_circuit has a parameter vqc which is a callable (i.e., a function) that defaults to the function basic_vqc if no other function is provided when calling get_circuit. This allows for flexibility in choosing different variational quantum circuits (VQCs) to be used in the quantum layer, while still providing a default option. That means, if you don't specify a different VQC function when calling get_circuit, it will use basic_vqc by default.
# How do I specify a different VQC function when calling get_circuit?
# You can specify a different VQC function by passing it as an argument when calling get_circuit. For example, if you have a custom VQC function called custom_vqc, you can call get_circuit like this:
# custom_circuit = get_circuit(vqc=custom_vqc)
# But I'm gonna need a function named custom_vqc first?
# Yes, you need to define the custom_vqc function first before you can pass it to get_circuit. The custom_vqc function should have the same signature as basic_vqc, meaning it should take a tc.Circuit object, inputs, and weights as parameters and modify the circuit accordingly. Here's an example of how you might define a custom VQC function. Here's an example of how you might define a custom VQC function:

# def custom_vqc(c: tc.Circuit, inputs, weights):
#     num_qubits = inputs.shape[-1]
#     num_qlayers = weights.shape[-2]

#     for i in range(num_qlayers):
#         for j in range(num_qubits):
#             c.rx(j, theta=weights[i, j])
#         if num_qubits == 2:
#             c.cz(0, 1)
#         elif num_qubits > 2:
#             for j in range(num_qubits):
#                 c.cz(j, (j + 1) % num_qubits)

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
        return x