from __future__ import annotations

from typing import List, Tuple

import numpy as np


def build_knn_graph(points: np.ndarray, k: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, float]:
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    idx = np.argpartition(dist2, k + 1, axis=1)[:, 1 : k + 1]
    d2_knn = dist2[np.arange(n)[:, None], idx]

    length_scale = np.median(np.sqrt(d2_knn[:, -1]))
    if not np.isfinite(length_scale) or length_scale <= 0:
        length_scale = 0.5

    adjacency = [dict() for _ in range(n)]
    denom = 2.0 * (length_scale ** 2) + 1e-12

    for i in range(n):
        for j, d2 in zip(idx[i], d2_knn[i]):
            if i == j:
                continue
            weight = float(np.exp(-d2 / denom))
            if weight < 1e-6:
                continue
            if j not in adjacency[i] or weight > adjacency[i][j]:
                adjacency[i][j] = weight
            if i not in adjacency[j] or weight > adjacency[j][i]:
                adjacency[j][i] = weight

    neighbors: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    degrees = np.zeros(n, dtype=np.float64)

    for i in range(n):
        neigh = np.array(list(adjacency[i].keys()), dtype=np.int32)
        w = np.array(list(adjacency[i].values()), dtype=np.float64)
        neighbors.append(neigh)
        weights.append(w)
        degrees[i] = float(w.sum())

    return neighbors, weights, degrees, length_scale


def apply_operator(
    vec: np.ndarray,
    neighbors: List[np.ndarray],
    weights: List[np.ndarray],
    degrees: np.ndarray,
    mu: float,
    confidence: np.ndarray,
) -> np.ndarray:
    out = degrees * vec
    for i, (nbrs, wts) in enumerate(zip(neighbors, weights)):
        if nbrs.size > 0:
            out[i] -= np.dot(wts, vec[nbrs])
    out += mu * confidence * vec
    return out


def conjugate_gradient(
    neighbors: List[np.ndarray],
    weights: List[np.ndarray],
    degrees: np.ndarray,
    mu: float,
    confidence: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> Tuple[np.ndarray, int, float]:
    x = x0.copy()
    apply = lambda v: apply_operator(v, neighbors, weights, degrees, mu, confidence)
    r = b - apply(x)
    p = r.copy()
    rsold = np.dot(r, r)

    for it in range(max_iter):
        Ap = apply(p)
        denom = np.dot(p, Ap)
        if abs(denom) < 1e-12:
            break
        alpha = rsold / denom
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            return x, it + 1, float(np.sqrt(rsnew))
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, max_iter, float(np.sqrt(rsold))


def estimate_diag_inverse(
    neighbors: List[np.ndarray],
    weights: List[np.ndarray],
    degrees: np.ndarray,
    mu: float,
    confidence: np.ndarray,
    samples: int = 16,
    tol: float = 1e-4,
    max_iter: int = 2000,
) -> np.ndarray:
    n = degrees.shape[0]
    diag_accum = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(2024)

    for _ in range(samples):
        z = rng.choice([-1.0, 1.0], size=n)
        x0 = np.zeros(n, dtype=np.float64)
        sol, _, _ = conjugate_gradient(
            neighbors,
            weights,
            degrees,
            mu,
            confidence,
            z,
            x0,
            tol=tol,
            max_iter=max_iter,
        )
        diag_accum += sol * z

    diag = diag_accum / samples
    diag = np.maximum(diag, 0.0)
    return diag

