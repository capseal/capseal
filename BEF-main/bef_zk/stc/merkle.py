"""Merkle tree utilities (binary + k-ary) for STC."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import hashlib


def _hash_children(children: List[bytes]) -> bytes:
    h = hashlib.sha256()
    for child in children:
        h.update(child)
    return h.digest()


def _hash_pair(left: bytes, right: bytes) -> bytes:
    return _hash_children([left, right])


def build_levels(leaves: List[bytes]) -> List[List[bytes]]:
    if not leaves:
        raise ValueError("cannot build Merkle tree with zero leaves")
    levels: List[List[bytes]] = [leaves]
    current = leaves
    while len(current) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            nxt.append(_hash_pair(left, right))
        levels.append(nxt)
        current = nxt
    return levels


def root_from_levels(levels: List[List[bytes]]) -> bytes:
    return levels[-1][0]


def prove(levels: List[List[bytes]], index: int) -> List[bytes]:
    if index < 0 or index >= len(levels[0]):
        raise ValueError("index out of range")
    proof: List[bytes] = []
    idx = index
    for level in levels[:-1]:
        sibling = idx ^ 1
        if sibling >= len(level):
            sibling = idx
        proof.append(level[sibling])
        idx //= 2
    return proof


def verify(root: bytes, leaf: bytes, index: int, proof: List[bytes]) -> bool:
    acc = leaf
    idx = index
    for sibling in proof:
        if idx % 2 == 0:
            acc = _hash_pair(acc, sibling)
        else:
            acc = _hash_pair(sibling, acc)
        idx //= 2
    return acc == root


def build_kary_levels(leaves: List[bytes], arity: int) -> List[List[bytes]]:
    if arity < 2:
        raise ValueError("arity must be >= 2")
    if not leaves:
        raise ValueError("cannot build Merkle tree with zero leaves")
    levels: List[List[bytes]] = [leaves]
    current = leaves
    while len(current) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(current), arity):
            block = current[i : i + arity]
            if len(block) == 1:
                nxt.append(block[0])
                continue
            padded = list(block)
            while len(padded) < arity:
                padded.append(padded[-1])
            nxt.append(_hash_children(padded))
        levels.append(nxt)
        current = nxt
    return levels


def prove_kary(levels: List[List[bytes]], index: int, arity: int) -> List[List[bytes]]:
    if index < 0 or index >= len(levels[0]):
        raise ValueError("index out of range")
    proof: List[List[bytes]] = []
    idx = index
    for level in levels[:-1]:
        group_start = (idx // arity) * arity
        siblings: List[bytes] = []
        for offset in range(arity):
            pos = group_start + offset
            if pos >= len(level):
                pos = len(level) - 1
            if pos == idx:
                continue
            siblings.append(level[pos])
        proof.append(siblings)
        idx //= arity
    return proof


def verify_kary(
    root: bytes,
    leaf: bytes,
    index: int,
    proof: List[List[bytes]],
    arity: int,
    total_leaves: int,
) -> bool:
    if index < 0 or index >= total_leaves:
        return False
    acc = leaf
    idx = index
    level_size = total_leaves
    for siblings in proof:
        group_start = (idx // arity) * arity
        children: List[bytes] = []
        sib_iter = iter(siblings)
        for offset in range(arity):
            pos = group_start + offset
            if pos >= level_size:
                pos = level_size - 1
            if pos == idx:
                children.append(acc)
            else:
                try:
                    children.append(next(sib_iter))
                except StopIteration:
                    return False
        if next(sib_iter, None) is not None:
            return False
        if len(siblings) == 0:
            acc = children[0]
        else:
            acc = _hash_children(children)
        idx //= arity
        level_size = max(1, math.ceil(level_size / arity))
    return acc == root


@dataclass
class MerkleMultiProof:
    tree_size: int
    arity: int
    sibling_levels: List[List[bytes]]


def multiproof(
    levels: List[List[bytes]],
    indices: Iterable[int],
    arity: int,
) -> MerkleMultiProof:
    base_size = len(levels[0])
    idxs = sorted(set(indices))
    if any(idx < 0 or idx >= base_size for idx in idxs):
        raise ValueError("multiproof index out of range")
    sibling_levels: List[List[bytes]] = []
    current = idxs
    for level in levels[:-1]:
        layer_siblings: List[bytes] = []
        current_set = set(current)
        next_set: set[int] = set()
        used: set[int] = set()
        for idx in current:
            if idx in used:
                continue
            group_start = (idx // arity) * arity
            for offset in range(arity):
                child = group_start + offset
                if child >= len(level):
                    child = len(level) - 1
                if child == idx:
                    continue
                if child not in current_set:
                    layer_siblings.append(level[child])
            for offset in range(arity):
                child = group_start + offset
                if child >= len(level):
                    child = len(level) - 1
                if child in current_set:
                    used.add(child)
            next_set.add(idx // arity)
        sibling_levels.append(layer_siblings)
        current = sorted(next_set)
    return MerkleMultiProof(tree_size=base_size, arity=arity, sibling_levels=sibling_levels)


def verify_multiproof(
    root: bytes,
    leaf_hashes: List[bytes],
    indices: List[int],
    proof: MerkleMultiProof,
) -> bool:
    if len(leaf_hashes) != len(indices):
        return False
    nodes = {idx: leaf for idx, leaf in zip(indices, leaf_hashes)}
    tree_size = proof.tree_size
    arity = proof.arity
    current_indices = sorted(nodes.keys())
    for siblings in proof.sibling_levels:
        next_nodes: dict[int, bytes] = {}
        sibling_iter = iter(siblings)
        used: set[int] = set()
        for idx in current_indices:
            if idx in used:
                continue
            group_start = (idx // arity) * arity
            children: List[bytes] = []
            for offset in range(arity):
                pos = group_start + offset
                if pos >= tree_size:
                    pos = tree_size - 1
                if pos == idx:
                    children.append(nodes[idx])
                elif pos in nodes:
                    children.append(nodes[pos])
                    used.add(pos)
                else:
                    try:
                        children.append(next(sibling_iter))
                    except StopIteration:
                        return False
            parent = _hash_children(children)
            parent_idx = idx // arity
            if parent_idx in next_nodes and next_nodes[parent_idx] != parent:
                return False
            next_nodes[parent_idx] = parent
            used.add(idx)
        if next(sibling_iter, None) is not None:
            return False
        nodes = next_nodes
        current_indices = sorted(nodes.keys())
        tree_size = max(1, math.ceil(tree_size / arity))
    return nodes.get(0) == root


__all__ = [
    "build_levels",
    "prove",
    "verify",
    "build_kary_levels",
    "prove_kary",
    "verify_kary",
    "MerkleMultiProof",
    "multiproof",
    "verify_multiproof",
    "root_from_levels",
]
