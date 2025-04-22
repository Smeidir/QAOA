# GNN-QAOA Training Pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
import random

# ==== GNN Model Definition ====
class QAOAInitialiserGNN(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.theta_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.beta_gamma_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        h = F.relu(self.conv3(x, edge_index))

        theta = torch.pi * self.theta_head(h).squeeze()
        graph_embedding = global_mean_pool(h, batch)
        beta_gamma = 2 * torch.pi * self.beta_gamma_head(graph_embedding).squeeze()
        return theta, beta_gamma

# ==== Cost Function & Simulation ====
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow import StateFn, AerPauliExpectation, CircuitSampler, PauliOp, SummedOp
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

def maxcut_hamiltonian(graph: nx.Graph):
    n = graph.number_of_nodes()
    terms = []
    for i, j in graph.edges():
        z_term = ['I'] * n
        z_term[i], z_term[j] = 'Z', 'Z'
        terms.append(PauliOp(Pauli(''.join(z_term)), coeff=-1))
    return SummedOp(terms)

def build_qaoa_circuit(theta_vec, beta, gamma, graph):
    n = len(theta_vec)
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    for i, theta in enumerate(theta_vec):
        qc.ry(theta.item(), qr[i])
    for i, j in graph.edges():
        qc.cx(qr[i], qr[j])
        qc.rz(-2 * gamma.item(), qr[j])
        qc.cx(qr[i], qr[j])
    for i in range(n):
        qc.rx(2 * beta.item(), qr[i])
    return qc

def simulate_expectation(circuit, hamiltonian):
    simulator = AerSimulator()
    psi = StateFn(circuit)
    measurable_expr = StateFn(hamiltonian, is_measurement=True) @ psi
    expectation = AerPauliExpectation().convert(measurable_expr)
    sampler = CircuitSampler(simulator).convert(expectation)
    return sampler.eval().real

# ==== Training Data Generation ====
def generate_graph_data(num_graphs=100, n_nodes_range=(4, 8)):
    dataset = []
    for _ in range(num_graphs):
        n = random.randint(*n_nodes_range)
        G = nx.erdos_renyi_graph(n, p=0.5)
        if not nx.is_connected(G):
            continue
        data = from_networkx(G)
        data.x = torch.ones((G.number_of_nodes(), 1))
        data.graph = G
        data.batch = torch.zeros(G.number_of_nodes(), dtype=torch.long)
        dataset.append(data)
    return dataset

# ==== Training Loop ====
def train_gnn(model, dataset, epochs=20, batch_size=1, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            theta, (beta, gamma) = model(batch)
            qaoa_circuit = build_qaoa_circuit(theta, beta, gamma, batch.graph)
            cost_h = maxcut_hamiltonian(batch.graph)
            energy = simulate_expectation(qaoa_circuit, cost_h)
            loss = energy  # lower is better

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Energy (Loss): {total_loss/len(loader):.4f}")

# ==== Execute Pipeline ====
if __name__ == "__main__":
    dataset = generate_graph_data(50)
    model = QAOAInitialiserGNN()
    train_gnn(model, dataset)
