import pennylane as qml
from pennylane import numpy as np

# Step 1: Create a quantum device with 2 qubits
dev = qml.device("default.qubit", wires=2)

def quantum(): circuit
@qml.qnode(dev)
def circuit(x, weights):
    
    # Step 3: Encode classical data into qubits (Feature Map)
    for i in range(2):
        qml.RY(x[i], wires=i)
    
    # Step 4: Variational Circuit (trainable part)
    for i in range(2):
        qml.RY(weights[i], wires=i)
    
    # Step 5: Entanglement
    qml.CNOT(wires=[0, 1])
    
    # Step 6: Measurement
    return qml.expval(qml.PauliZ(0))


# Step 7: Define a simple dataset
X = np.array([[0.1, 0.2],
              [0.4, 0.6],
              [0.8, 0.9],
              [0.3, 0.7]])

Y = np.array([0, 0, 1, 1])

# Step 8: Initialize weights randomly
weights = np.random.rand(2)

# Step 9: Define loss function
def loss(weights):
    loss = 0
    for i in range(len(X)):
        prediction = circuit(X[i], weights)
        loss += (prediction - Y[i])**2
    return loss / len(X)

# Step 10: Optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# Step 11: Training loop
for i in range(50):
    weights = opt.step(loss, weights)
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss(weights)}")

print("Final weights:", weights)