## 0 TL;DR for the Agent

You are building **Self-Assembling Expert Graphs (SAM-v2)**: a federated-learning framework that lets a pre-trained LLM **discover, train, and re-wire** its own Mixture-of-Experts (MoE) topology on non-IID client data—while communicating only the parameters that matter.
The repo already contains clustering helpers from **EMoE**, lightweight FL utilities from **FedIT / Shepherd**, plus empty **stubs** you will fill.  Keep every change green on `pytest -q`.

---

## 1 Scientific Primer

### 1.1 Emergent MoE (EMoE)

*Observation* In a trained Transformer, the rows of an FFN’s weight matrix often partition into **functional clusters** (e.g., arithmetic vs. syntax neurons).
EMoE formalises this:

1. **Cluster** rows of the gate/up/down projection with *balanced* k-means-constrained

   $$
     \min_{\boldsymbol Z\in\{0,1\}^{m\times E}}\;
       \|W - Z\,C\|_F^2
     \quad\text{s.t. } Z\mathbf1= \frac{m}{E}\mathbf1,
   $$

   where $m$ rows → $E$ equal-size clusters.

2. **Re-order** rows so that each contiguous block forms an *expert*.

EMoE’s top-$k$ gating is then

$$
\textstyle y = \sum_{e\in\operatorname{Top}_k(g(x))}\!
         p_e(x)\;\bigl[\operatorname{FFN}_e(x)\bigr],
\quad
p_e(x)=\frac{\exp g_e(x)}{\sum_{j\in \operatorname{Top}_k} \exp g_j(x)}.
$$

We adapt only **step 1** (clustering) to obtain the initial expert library; SAM-v2 replaces EMoE’s fixed router with a learnable **graph assembler**.

---

### 1.2 Conditional Computation (MoE quick maths)

If each token activates $k$ of $E$ experts, per-token FLOPs scale as

$$
\text{FLOPs}_\text{MoE} \;\approx\; \frac{k}{E}\,\text{FLOPs}_\text{dense}.
$$

For $k=2,\;E=16$, that is a **8 ×** speed-up while parameter count remains full.

---

### 1.3 Federated Learning in 60 seconds

*Setting* $N$ clients, local datasets $D_i$, server aggregates updates.
*FedAvg* update at round $t$:

$$
\Theta_{t+1} \;=\; \sum_{i=1}^N \frac{|D_i|}{\sum_j |D_j|}\;
                 \Bigl[\Theta_t - \eta\,\nabla_\Theta
                     \ell_i\bigl(\Theta_t; D_i\bigr)\Bigr].
$$

Key pain-points for LLMs:

* **Bandwidth** Uploading ≈ 7 B params per client per round is infeasible.
* **Device heterogeneity** Phones vs. servers can’t all process the same tensor shapes.
* **Non-IID data** One global model forgets minority domains.

**SAM-v2** tackles all three by:

* sending only *k* experts’ gradients,
* letting each client pick a sub-graph that fits its budget,
* allowing the graph to grow (spawn experts) when data drift occurs.

---

## 2 Repository Layout & File Roles (2025-08 snapshot)

| Path                                               | Role                                                 |
| -------------------------------------------------- | ---------------------------------------------------- |
| **`experts/emoe_cluster.py`**                      | K-means-constrained clustering → initial experts.    |
| **`experts/moe_utils.py`**                         | Example conversion routines (reference only).        |
| **`common/emoe_utils.py`**                         | JSON helpers + prompt templates.                     |
| **`common/prompting.py`**                          | `Prompter` class for instruction-tuning data.        |
| **`common/callbacks.py`**                          | Trainer callbacks for logging.                       |
| **`common/utils.py`**                              | **Stub** → will house RLE, Rich logger, timers.      |
| **`fedit/fed_utils/`**                             | Lightweight FedAvg, client sampling from Shepherd.   |
| **`fedit/client_data_allocation.py`**              | Splits The Pile into non-IID client shards.          |
| **`models/llama_loader.py`**                       | **Stub** → load Llama-2-7B in 4-bit QLoRA.           |
| **`models/assembler.py`**                          | **Stub** → DAG-safe mask generator.                  |
| **`federated/fl_wrapper.py`**                      | **Stub** → Flower `Strategy` for sparse FedAvg.      |
| **`federated/server_merge.py`**                    | **Stub** → prune/promote edges; cap edits ≤ `ρ_max`. |
| **`clients/client_app.py` / `dataset_wikimix.py`** | **Stubs** → Flower `Client` + dataset stream.        |
| **`scripts/bootstrap_experts.sh`**                 | **Stub** → run EMoE clustering once.                 |
| **`scripts/fl_finetune.py`**                       | Shepherd’s demo loop (reference only).               |
| **`configs/`**                                     | Llama base config + prompt templates.                |
| **`tests/`**                                       | PyTest suite (starts with placeholder).              |

---

## 3 Coding & Testing Conventions

* **Python 3.11**, formatted with `black` + `isort`.
* **Edge mask** tensor: `bool` shape `(E,E)` and `mask == mask.triu()`.
* **Sparse gradients**: dict `{expert_id: tensor}`.
* Add a **unit test** for every new public function.
* Logs → `logs/` (use **Rich**).
* CI = `pytest -q` + 2-client Flower smoke test.

---

## 4 Roadmap (actor ↔ module ↔ test)

| # | Actor     | Build / modify                                               | Must pass ...                            |
| - | --------- | ------------------------------------------------------------ | ---------------------------------------- |
| 1 | **Codex** | `models/assembler.py` — MLP → upper-triangular mask.         | `tests/test_assembler.py`; dump mask.    |
| 2 | **Codex** | `experts/moe_layers.py` — DFS routing.                       | `tests/test_router.py`; call trace JSON. |
| 3 | **Codex** | `common/utils.py` — RLE helpers.                             | `tests/test_rle.py` round-trip.          |
| 4 | **Codex** | `federated/server_merge.py` — prune/promote.                 | synthetic counts → edit list ≤ `ρ_max`.  |
| 5 | **Codex** | `federated/fl_wrapper.py` — sparse FedAvg + broadcast edits. | 2-client sim, loss ↓, edits OK.          |
| 6 | **Codex** | `clients/client_app.py` — apply edits, track usage.          | edge-usage non-zero.                     |
| 7 | **Human** | Fill `scripts/bootstrap_experts.sh`; run.                    | `logs/cluster_stats.txt`.                |
| 8 | **Human** | Launch 50-client run on A40 cluster.                         | TensorBoard + JSON logs.                 |
| 9 | **Codex** | Ablation runner & bound-check tests.                         | CI green; CSV summary produced.          |

---

## 5 Debug-Log Targets

| File                          | Insight                                      |
| ----------------------------- | -------------------------------------------- |
| `logs/mask_debug.npy`         | First assembler mask (check DAG / sparsity). |
| `logs/expert_call_trace.json` | Which experts fired per sequence.            |
| `logs/merge_round*.json`      | Edge edits per merge (≤ `ρ_max`).            |
| `logs/cluster_stats.txt`      | EMoE cluster counts & variance.              |
| `logs/pack_sizes.txt`         | Compression ratios for sparse updates.       |

---

## 6 How Codex Should Work

1. Load only the file being edited + its interfaces.
2. Add/adjust unit tests; run `pytest -q`.
3. If tests fail, show failing lines & propose patch.
4. Commit only when CI green.

---

