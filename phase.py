from qiskit import QuantumCircuit, Aer, assemble
from math import pi
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex

# Create quantum circuit
qc = QuantumCircuit(2)

# Apply Hadamard gates
qc.h(0)
qc.h(1)

# Apply CNOT gate
qc.cx(0, 1)

# Draw the circuit
qc.draw()

# Simulate the result
svsim = Aer.get_backend('aer_simulator')
qc.save_statevector()
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()

# Display the result
display(array_to_latex(final_state, prefix="\\text{Statevector}="))
plot_bloch_multivector(final_state)

# Apply Controlled-Phase gate (T-gate variant)
qc.cp(pi / 4, 0, 1)
display(qc.draw())

# Save and simulate the state
qc.save_statevector()
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()

# Plot the result
plot_bloch_multivector(final_state)
