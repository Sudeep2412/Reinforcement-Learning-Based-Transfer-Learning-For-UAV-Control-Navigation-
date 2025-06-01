import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, assemble
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import MCMT
from math import sqrt, pow, pi

# Initialize the Aer simulator
simulator = Aer.get_backend('qasm_simulator')

# Example: Create a simple quantum circuit
qc = QuantumCircuit(2)  # 2-qubit quantum circuit
qc.h(0)  # Apply Hadamard gate on the first qubit
qc.cx(0, 1)  # Apply CNOT gate

# Transpile the circuit for the simulator
compiled_circuit = transpile(qc, simulator)

# Run the simulation
job = simulator.run(assemble(compiled_circuit))
result = job.result()

# Print the result
print(result.get_counts())
