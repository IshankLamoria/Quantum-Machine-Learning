import pennylane as qml
from pennylane import numpy as np

# Step 1: Create device (quantum simulator)
dev = qml.device("default.qubit", wires=2)

# Step 2: Define Hamiltonian
coeffs = [0.5, -1.2, 0.8]

observables = [
    qml.PauliZ(0),
    qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1)
]

H = qml.Hamiltonian(coeffs, observables)

# Step 3: Variational circuit
@qml.qnode(dev)
def circuit(params):
    
    # Prepare state
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    
    # Entanglement
    qml.CNOT(wires=[0, 1])
    
    return qml.expval(H)

# Step 4: Cost function
def cost(params):
    return circuit(params)

# Step 5: Initialize parameters
params = np.array([0.1, 0.2], requires_grad=True)

# Step 6: Optimizer
opt = qml.AdamOptimizer(stepsize=0.1)

# Step 7: Training loop
steps = 50

for i in range(steps):
    params = opt.step(cost, params)
    
    if i % 10 == 0:
        print(f"Step {i}: Energy = {cost(params)}")

print("\nFinal Energy:", cost(params))
print("Final Parameters:", params)