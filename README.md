# Self-Assembling Expert Graphs (SAM-v2)

*A federated learning framework that co-evolves weights **and** topology for Large Language Models (LLMs)*

---

## 1 Problem Setting & Motivation

In practical federated learning (FL), multiple clients collaboratively train a shared model while keeping their data private. Applying FL to large language models (LLMs) introduces unique challenges:

| Challenge                      | Explanation                                                                           | Practical Impact                                       |
| ------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Communication overhead**     | Clients sending multi-billion parameter models is infeasible.                         | High bandwidth & latency                               |
| **Device heterogeneity**       | Different devices (e.g., mobile phones vs. GPUs) cannot handle identical model sizes. | Training inefficiency, device crashes                  |
| **Non-IID data distributions** | Each client has domain-specific data (medical, legal, coding).                        | Performance degradation due to catastrophic forgetting |

Traditional federated algorithms like FedAvg ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)) assume a uniform model and transmit the entire model each round, causing severe inefficiency.

Recently, conditional computation methods (Mixture-of-Experts, MoE) ([Fedus et al., 2021](https://arxiv.org/abs/2101.03961)) significantly reduce computational overhead by activating only a subset of experts per token. Further, the **Emergent MoE (EMoE)** framework ([Sun & Li, 2023](https://arxiv.org/abs/2305.06275)) reveals that pretrained transformers naturally cluster their neurons into distinct functional groups, called **latent experts**.

**Core Insight**: We propose combining **latent expert discovery (EMoE)** with **conditional computation (MoE)** in a federated learning setup. Each client maintains its own adaptive expert graph tailored to its data and hardware, while the server helps evolve the expert structure over time—hence **Self-Assembling Expert Graphs (SAM-v2)**.

---

## 2 Background & Key Papers

| Area                              | Key Paper                               | Contribution                                                                                         |
| --------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Emergent modularity**           | EMoE (Sun & Li, 2023)                   | Latent experts discovered via balanced k-means clustering of FFN neurons in pretrained transformers. |
| **Conditional computation (MoE)** | Switch Transformer (Fedus et al., 2021) | Each token activates a small number of experts, drastically reducing computation.                    |
| **Federated Learning**            | FedAvg (McMahan et al., 2017)           | Iterative averaging of client model updates, foundational FL convergence theory.                     |
| **Sparse FL updates**             | FedMoE (Du et al., 2023)                | Demonstrated that sparse MoE updates reduce communication, though the expert structure is fixed.     |
| **Device heterogeneity**          | HeteroFL (Li et al., 2020)              | Personalized model subsets per device, allowing heterogeneous model sizes with unbiased updates.     |

Our approach synthesizes these ideas to create an adaptive federated MoE framework that is both communication-efficient and adaptive to client diversity.

---

## 3 Mathematical Formulation (with Natural Language Explanation)

### 3.1 Expert Discovery via Clustering

Given a pretrained LLM, we first cluster its FFN layers into latent experts using balanced k-means:

$$
  \min_{C,Z}\|W - ZC\|_F^2\quad
  \text{s.t. }Z\in\{0,1\}^{m\times E},\; Z\mathbf1 = \tfrac{m}{E}\mathbf1.
$$

Here, $W$ is the weight matrix, $Z$ assigns neurons (rows) to exactly one of $E$ clusters, and $C$ are the cluster centroids. Each cluster naturally forms an "expert," specializing in different skills or knowledge domains.

### 3.2 Local Expert Graph Construction

Each client $i$ maintains an **assembler** (a small neural network), denoted $A_i$, that dynamically outputs an expert graph mask $M_i$:

* **Expert Graph (Mask)**: $M_i \in \{0,1\}^{E\times E}$, upper-triangular to guarantee the graph is a Directed Acyclic Graph (DAG).

The mask $M_i$ controls expert routing: if $M_{i,pq}=1$, expert $p$ passes information to expert $q$.

### 3.3 Local Objective (Sparse Execution)

For each client $i$, the loss is computed only over the experts activated by their mask:

* Define the active expert set as:

$$
\mathcal P_i(x) = \{(p,q)\mid M_{i,pq}=1,\; x\text{ reaches }q\text{ from }p\}.
$$

* The client's objective becomes:

$$
F_i(\Theta,M_i) = \mathbb E_{(x,y)\sim D_i}\bigl[\ell(f_\Theta^{M_i}(x),y)\bigr].
$$

In words: each client computes loss only on the subset of experts active for their data, greatly reducing computational and communication overhead.

### 3.4 Federated Sparse Gradient Aggregation

Clients send only gradients for activated experts, significantly compressing the model update:

* Client gradient updates:

$$
\Delta_i = \{(e, g_{i,e}) \mid e\text{ activated by client }i\}.
$$

* Server-side sparse aggregation:

$$
\theta_e^{(t+1)} = \theta_e^{(t)} - \eta \sum_{i:e\in\Delta_i}\frac{|D_i|}{\sum_j|D_j|}p_{i,e}^{-1}g_{i,e},
$$

where $p_{i,e}$ corrects for sampling bias, ensuring unbiased updates despite sparsity.

### 3.5 Global Graph Adaptation (Server-side)

Periodically (every $T_m$ rounds), the server aggregates edge usage statistics across clients and edits the global expert graph (adding or removing edges):

* **Perturbation Lemma** (graph stability): Changing $\rho_t$ edges affects the global objective at most by:

$$
|L(\Theta,M_{t})-L(\Theta,M_{t-1})|\le \rho_t\gamma,
$$

where $\gamma$ measures how sensitive the loss is to graph edits.

* **Bounded Drift Theorem** (convergence guarantee): Under standard assumptions (PL condition with constant $\mu$, bounded gradient noise $\sigma^2$), we have:

$$
\mathbb E[L_T - L^\*]\le(1-\eta\mu)^T(L_0-L^\*)+\frac{\eta\sigma^2}{2\mu}+\rho_{\max}\gamma T_m.
$$

Intuitively, this ensures the system converges to a small error region, controlled explicitly by the frequency and extent of graph changes.

---

## 4 System Architecture (Simplified Overview)

```
Clients                                   Server
┌───────────────┐                  ┌───────────────────────┐
│ Assembler (MLP)│───mask─────────▶│ Global Edge Counter C │
│               │                  │(aggregates client data)│
│ Executes      │───sparse grad───▶│ Aggregate & update     │
│subset of experts│                │  global experts        │
└───────────────┘                  └───────────┬───────────┘
                                               │
                                               │ (edge edit list)
                                               ▼
                                      Update local expert graphs
```

* **Static**: Initial clustering (via EMoE) to define latent experts.
* **Dynamic**: Lightweight updates to experts and adaptive expert graph topology.

---

## 5 Implementation Roadmap

| Component              | Functionality                          |
| ---------------------- | -------------------------------------- | 
| Expert Clustering      | Balanced k-means initialization (EMoE) | 
| DAG Assembler          | MLP → DAG-safe adjacency masks         | 
| Sparse Routing         | DFS through expert graph               | 
| Gradient Compression   | Run-length encoding (RLE) utilities    | 
| Server Merge Logic     | Graph pruning and promotion logic      | 
| Federated Strategy     | Sparse gradient aggregation (Flower)   | 
| Dataset & Client Logic | Data loader for non-IID distribution   | 

---

## 6 Experimental Setup

* **Model**: Llama-2-7B, MoE-layered via EMoE method.
* **Federation**: 50 clients simulating real-world non-IID distributions.
* **Metrics**: Perplexity, communication cost, client computational load, robustness to data shifts.
* **Baselines**: FedAvg, FedMoE, HeteroFL, pFedGate.
* **Hardware**: Cluster of 8 × NVIDIA A40 GPUs.

