import pennylane as qml
from pennylane import numpy as np

# Step 1: Define graph (edges)
edges = [(0,1), (1,2)]

n_qubits = 3

# Step 2: Device
dev = qml.device("default.qubit", wires=n_qubits)

# Step 3: Cost Hamiltonian
cost_h = 0
for i, j in edges:
    cost_h += 0.5 * (1 - qml.PauliZ(i) @ qml.PauliZ(j))

# Step 4: Mixer Hamiltonian
mixer_h = sum(qml.PauliX(i) for i in range(n_qubits))

# Step 5: QAOA circuit
@qml.qnode(dev)
def circuit(gamma, beta):
    
    # Initial state
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # Problem unitary
    for i, j in edges:
        qml.CNOT(wires=[i,j])
        qml.RZ(2*gamma, wires=j)
        qml.CNOT(wires=[i,j])
    
    # Mixer unitary
    for i in range(n_qubits):
        qml.RX(2*beta, wires=i)
    
    return qml.expval(cost_h)

# Step 6: Cost function
def cost(params):
    gamma, beta = params
    return -circuit(gamma, beta)  # maximize

# Step 7: Initialize params
params = np.array([0.1, 0.1], requires_grad=True)

# Step 8: Optimizer
opt = qml.AdamOptimizer(stepsize=0.1)

# Step 9: Training loop
for i in range(50):
    params = opt.step(cost, params)
    
    if i % 10 == 0:
        print(f"Step {i}: Value = {-cost(params)}")

print("\nOptimal parameters:", params)
print("Max Cut Value:", -cost(params))