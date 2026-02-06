# FusionAlpha Source Documentation

## 1. System Overview

**FusionAlpha** is a Rust-based planning engine that fuses graph-based navigation with epistemic uncertainty estimates (from ENNs) and path-based reliability metrics (from BICEP). It is designed to solve long-horizon tasks by propagating "committor values" (probability of reaching a goal before failure) across a state-space graph.

The system is built as a Rust crate (`fusion-core`, `fusion-envs`, `fusion-bindings`) with Python bindings for integration into machine learning pipelines (e.g., OGBench).

## 2. Core Primitives (`fusion-core`)

### 2.1. Graph Representation
*   **`NodeFeat`**: Represents a node in the state space.
    *   `x`, `y`: Spatial coordinates for distance calculations.
    *   `extra`: Vector for high-dimensional state information (e.g., joint angles, light configurations).
*   **`Edge`**: Weighted directed connection between nodes.
    *   `w`: Represents transition reliability or frequency.
*   **`Graph`**: container for nodes and edges, supporting fast neighbor lookups via adjacency lists.

### 2.2. Priors & Confidence (`priors.rs`)
The planner does not start from scratch; it incorporates external knowledge:
*   **`Priors`**: A collection of initial $q$-values (`q0`) and confidence weights (`eta`) for each node.
*   **`PriorSource`**: Enum handling different sources of information:
    *   `ENN`: Uses $Q_{ENN}(s,a)$ predictions scaled by epistemic uncertainty (severity/entropy).
    *   `BICEP`: Uses empirical success rates from rolled-out trajectories.
    *   `Manual`: Heuristics (e.g., Euclidean distance).

### 2.3. Committor Propagation (`propagation.rs`)
This is the core algorithm, solving a boundary value problem on the graph.
*   **Equation**: The committor function $q(s)$ satisfies a discrete Dirichlet problem:
    $$ q(s) = \frac{\sum_{s'} w_{ss'} q(s') + \eta(s) q_{prior}(s)}{\sum_{s'} w_{ss'} + \eta(s)} $$
*   **Boundary Conditions**:
    *   Goal nodes: $q(s) = 1$ (fixed, $\eta \to \infty$).
    *   Fail/Dead-end nodes: $q(s) = 0$ (fixed).
*   **Risk Sensitivity**: Supports a "risk-sensitive" propagation mode (controlled by `alpha` and `risk_aversion`) using an exponential transform:
    $$ q(s) \propto \frac{1}{\alpha} \log \sum w_{ss'} e^{\alpha q(s')} $$
    This allows the planner to be optimistic ($\alpha < 0$) or pessimistic ($\alpha > 0$) about transition outcomes.

### 2.4. Action Selection (`actions.rs`)
*   **`pick_next_node`**: Greedily selects the neighbor with the highest committor value.
*   **`pick_next_node_weighted`**: Softmax sampling for exploration.
*   **`ActionDecoder`**: Traits for converting the abstract "next node" decision into environment-specific actions (e.g., velocity vectors, joint torques).

## 3. Environment Modules (`fusion-envs`)

Contains specific implementations for building graphs from environment observations.

*   **Humanoid Maze**:
    *   Grid-based graph with walls.
    *   Special "teleporter" edges with stochastic success probabilities.
    *   White holes (dead ends) act as absorbing failure states.
*   **Ant Soccer**:
    *   Ball-centric graph nodes.
    *   Edges represent ball movements; weights penalized by the cost of the ant repositioning itself to push the ball.
    *   Special "shot" edges for long-range kicks.
*   **Puzzle (4x5 Lights)**:
    *   Nodes are 20-bit bitmasks representing light configurations.
    *   Edges represent button presses (toggling a light and its neighbors).
    *   Uses BFS to expand the local state space graph up to a depth limit.

## 4. Python Bindings (`fusion-bindings`)

Exposes core functionality to Python via `pyo3`.

*   **`simple_propagate`**: The main entry point. Accepts numpy arrays for nodes and edges, constructs the Rust graph, runs propagation, and returns the $q$-values.
*   **`graph_builder.py`**: Python-side helpers (using `sklearn.neighbors`) to build k-NN graphs from replay buffers or BICEP trajectories, bridging the gap between raw data and the Rust solver.

## 5. Usage Flow

1.  **Graph Construction**: The Python agent observes the current state and constructs a local graph (using `GraphBuilder` or BICEP rollouts).
2.  **Prior Injection**: ENN predictions and BICEP success rates are converted into `Priors` (value + confidence).
3.  **Propagation**: `simple_propagate` solves for the committor function $q(s)$ on the graph.
4.  **Action Selection**: The agent moves towards the neighbor with the highest $q(s)$.

```python
# Conceptual Python usage
nodes, edges = build_local_graph(current_state, replay_buffer)
q_values = fusion_alpha.simple_propagate(
    nodes, edges, goal_node, current_node, 
    enn_q_prior=0.7, severity=0.2, t_max=50
)
next_node = np.argmax(q_values[neighbors])
```
