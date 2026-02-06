use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use fusion_core::*;

const ETA_MIN: f32 = 0.1;
const ETA_MAX: f32 = 10.0;

/// Simplified Python bindings with working example
#[pymodule]
fn fusion_alpha(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simple_propagate, m)?)?;
    m.add_function(wrap_pyfunction!(create_simple_graph, m)?)?;
    Ok(())
}

/// Simple committor propagation (core functionality)
#[pyfunction]
fn simple_propagate<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArray2<f32>,        // (N, 2) node coordinates
    edges: PyReadonlyArray2<f32>,        // (M, 3) edge list [u, v, w]
    goal_node: usize,
    current_node: usize,
    enn_q_prior: f32,
    severity: f32,
    t_max: usize,
) -> PyResult<&'py PyArray1<f32>> {
    
    // Build graph
    let nodes_array = nodes.as_array();
    let edges_array = edges.as_array();
    
    let mut node_list = Vec::new();
    for i in 0..nodes_array.nrows() {
        let x = nodes_array[[i, 0]];
        let y = nodes_array[[i, 1]];
        node_list.push(NodeFeat::new(x, y));
    }
    
    let mut edge_list = Vec::new();
    for i in 0..edges_array.nrows() {
        let u = edges_array[[i, 0]] as usize;
        let v = edges_array[[i, 1]] as usize;
        let w = edges_array[[i, 2]];
        edge_list.push(Edge::new(u, v, w));
    }
    
    let graph = Graph::new(node_list, edge_list);
    
    // Set up priors
    let mut q0 = vec![None; graph.num_nodes()];
    let mut eta = vec![0.0; graph.num_nodes()];
    
    // Goal boundary
    q0[goal_node] = Some(1.0);
    eta[goal_node] = 1e9;
    
    let fusion_state = FusionState::from_severity(enn_q_prior, severity, 0.0);
    q0[current_node] = Some(fusion_state.q_prior_enn);
    eta[current_node] = eta_from_reliability(fusion_state.obs_reliability);

    // Propagate
    let config = PropConfig {
        t_max,
        eps: 1e-4,
        use_parallel: true,
        alpha_max: 6.0,
        step_policy: StepPolicy::RiskScaled,
    };
    
    let t_steps = fusion_state.propagation_steps(&config);
    let q_values = propagate_committor(
        &graph,
        &q0,
        &eta,
        &config,
        t_steps,
        fusion_state.effective_risk(),
    );
    
    Ok(PyArray1::from_vec(py, q_values))
}

fn eta_from_reliability(rel: f32) -> f32 {
    let clamped = rel.clamp(0.0, 1.0);
    ETA_MIN + (ETA_MAX - ETA_MIN) * clamped
}

/// Create a simple test graph
#[pyfunction]
fn create_simple_graph<'py>(py: Python<'py>) -> PyResult<(
    &'py PyArray2<f32>,  // nodes
    &'py PyArray2<f32>,  // edges
    usize,               // current_node
    usize,               // goal_node
)> {
    // 3-node chain: 0 -- 1 -- 2
    let nodes = vec![
        vec![0.0, 0.0],  // Node 0
        vec![1.0, 0.0],  // Node 1
        vec![2.0, 0.0],  // Node 2 (goal)
    ];
    
    let edges = vec![
        vec![0.0, 1.0, 1.0],  // 0 -> 1
        vec![1.0, 0.0, 1.0],  // 1 -> 0
        vec![1.0, 2.0, 1.0],  // 1 -> 2
        vec![2.0, 1.0, 1.0],  // 2 -> 1
    ];
    
    let nodes_array = PyArray2::from_vec2(py, &nodes)?;
    let edges_array = PyArray2::from_vec2(py, &edges)?;
    
    Ok((nodes_array, edges_array, 0, 2))
}
