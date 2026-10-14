#!/usr/bin/env python3

# Import required libraries.
import argparse
import hashlib
import heapq
import json
import math
import os
import pickle
import time
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads

# Configure model and runtime parameters.
P2P_DIR = "p2p"
P2P_FILENAME = "p2p_composable_TMSV.txt"
P2P_EPSILON_TARGET = 0.01
SNAP_P2P = 0.10
R_DB_MIN = 1
R_DB_MAX = 30

SWAP_XI = 0.01
SWAP_ALPHA_DB_PER_KM = 0.2
SWAP_BLOCK_SIZE = int(1e9)
SWAP_PE_FRACTION = 0.10
SWAP_BETA = 0.98
SWAP_FER = 0.10
SWAP_ALPHABET_SIZE = 2**5
SWAP_EPS_COR = 1e-10
SWAP_EPS_SMOOTH = 1e-10
SWAP_EPS_HASH = 1e-10
SWAP_EPS_PE = 1e-10
SWAP_ETA_BELL = 0.98

SWAP_R_DB_START = 14
SWAP_R_DB_MIN = 1
SWAP_R_DB_MAX = 30
SWAP_R_DB_IMPROVEMENT_TOL = 1e-10
MIN_SWAP_LINKS = 2
MAX_SWAP_LINKS = 4
LIVE_RATE_DISTANCE_DECIMALS = 2

STATIC_PAIR_INNER_FRACTION = 0.7
NETWORK_CACHE_DIR = "network_topology_cache"
NETWORK_CACHE_VERSION = 3

OPTIMIZATION_ITERATIONS = 8000
OPTIMIZATION_RANDOM_SEED = 100
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0
WEIGHT_DECIMALS = 2
RATE_WEIGHT_FIXED = 1.0
A_STAR_PENALTY_STRENGTH = 0.5
REMOVE_USED_NODES = True
COUNT_ENDPOINTS_IN_N = False

NUMBA_PARALLEL_WITHIN_JOB = False
NUMBA_THREADS_STANDALONE = 1
NUMBA_THREADS_PER_JOB = 1

WEIGHT_NAMES = (
    "rate",
    "distance",
    "swap_relay",
    "hop",
    "swap_away",
    "swap_detour",
    "heuristic",
)

P2P_CACHE = {}
OPTIMIZED_SWAP_CACHE = {}
LIVE_SWAP_RDB_CACHE = {}

# Set the Numba thread count for this process.
def configure_numba_threads(thread_count):
    global NUMBA_THREADS_PER_JOB
    thread_count = int(thread_count)
    if thread_count <= 0:
        raise ValueError("Numba thread count must be positive")
    thread_count = min(thread_count, os.cpu_count() or 1)
    NUMBA_THREADS_PER_JOB = thread_count
    set_num_threads(thread_count)
    return thread_count

# List possible locations for the P2P lookup file.
def candidate_p2p_files():
    return list(
        dict.fromkeys(
            [
                os.path.join(P2P_DIR, P2P_FILENAME),
                P2P_FILENAME,
                os.path.join(P2P_DIR, P2P_FILENAME.upper()),
                P2P_FILENAME.upper(),
            ]
        )
    )

# Load the P2P lookup rates into memory.
def load_p2p_data(epsilon=P2P_EPSILON_TARGET):
    P2P_CACHE.clear()
    paths = candidate_p2p_files()
    path = next((value for value in paths if os.path.exists(value)), None)
    if path is None:
        raise FileNotFoundError("P2P lookup not found. Tried: " + ", ".join(paths))

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"r_db", "eps", "d_AB", "SKR"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"P2P lookup is missing columns: {missing}")

    df = df[np.isclose(df["eps"], float(epsilon), atol=1e-4)].copy()
    if df.empty:
        raise ValueError(f"P2P lookup contains no rows for eps={epsilon}")

    for row in df.itertuples(index=False):
        key = (round(float(row.r_db), 1), round(float(row.d_AB), 2))
        P2P_CACHE[key] = float(row.SKR)
    return path

# Build the network-cache metadata.
def network_cache_descriptor(seed, num_nodes, beta_waxman, r0, radius):
    return {
        "cache_version": int(NETWORK_CACHE_VERSION),
        "seed": int(seed),
        "num_nodes": int(num_nodes),
        "beta_waxman": round(float(beta_waxman), 12),
        "r0": round(float(r0), 12),
        "radius": round(float(radius), 12),
    }

# Create a deterministic path for a network cache.
def network_cache_path(seed, num_nodes, beta_waxman, r0, radius):
    descriptor = network_cache_descriptor(
        seed, num_nodes, beta_waxman, r0, radius
    )
    digest = hashlib.sha256(
        pickle.dumps(descriptor, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()[:24]
    os.makedirs(NETWORK_CACHE_DIR, exist_ok=True)
    return os.path.join(NETWORK_CACHE_DIR, f"network_{digest}.pkl")

# Generate one Waxman network realization.
def create_base_network(seed, num_nodes, beta_waxman, r0, radius):
    rng = np.random.RandomState(int(seed))
    radii = float(radius) * np.sqrt(rng.rand(int(num_nodes)))
    angles = 2.0 * np.pi * rng.rand(int(num_nodes))
    positions = {
        i: np.array(
            [radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i])],
            dtype=float,
        )
        for i in range(int(num_nodes))
    }

    graph = nx.Graph()
    graph.add_nodes_from(positions)
    graph.graph["radius"] = float(radius)

    for u, v in combinations(positions, 2):
        distance = float(np.linalg.norm(positions[u] - positions[v]))
        probability = float(beta_waxman) * math.exp(-distance / float(r0))
        if rng.rand() < probability:
            graph.add_edge(u, v, true_weight=distance)

    return graph, positions, rng.get_state()

# Create and save a reusable network realization.
def create_network_cache(seed, num_nodes, beta_waxman, r0, radius):
    descriptor = network_cache_descriptor(
        seed, num_nodes, beta_waxman, r0, radius
    )
    path = network_cache_path(seed, num_nodes, beta_waxman, r0, radius)

    if os.path.exists(path):
        payload = load_network_cache(path)
        if payload["descriptor"] != descriptor:
            raise ValueError(f"Cache descriptor mismatch: {path}")
        return path

    graph, positions, compromise_rng_state = create_base_network(
        seed, num_nodes, beta_waxman, r0, radius
    )
    payload = {
        "descriptor": descriptor,
        "graph": graph,
        "positions": positions,
        "compromise_rng_state": compromise_rng_state,
    }

    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    return path

# Load and validate a cached network.
def load_network_cache(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    required = {"descriptor", "graph", "positions", "compromise_rng_state"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Invalid network cache payload: {path}")
    if not isinstance(payload["graph"], nx.Graph):
        raise TypeError(f"Cached graph is not a NetworkX graph: {path}")
    if not isinstance(payload["positions"], dict):
        raise TypeError(f"Cached positions are invalid: {path}")
    return payload

# Split nodes into trusted and compromised sets.
def classify_network_nodes(graph, compromise_rng_state, prob_comp, degree_threshold):
    rng = np.random.RandomState()
    rng.set_state(compromise_rng_state)
    safe_nodes = set()
    compromised_nodes = set()

    for node in graph.nodes:
        if graph.degree(node) > int(degree_threshold) or rng.rand() < float(prob_comp):
            compromised_nodes.add(int(node))
        else:
            safe_nodes.add(int(node))
    return safe_nodes, compromised_nodes

# Measure the physical distance between two nodes.
def euclidean_distance(u, v, positions):
    return float(np.linalg.norm(positions[u] - positions[v]))

# Find the nearest available trusted node.
def nearest_safe_node(positions, safe_nodes, target_point, excluded=None):
    excluded = set() if excluded is None else set(excluded)
    candidates = [node for node in safe_nodes if node not in excluded]
    if not candidates:
        return None
    target = np.asarray(target_point, dtype=float)
    return int(min(candidates, key=lambda node: np.linalg.norm(positions[node] - target)))

# Choose fixed source and target nodes.
def select_static_pair(positions, safe_nodes, radius):
    source = nearest_safe_node(
        positions,
        safe_nodes,
        (-STATIC_PAIR_INNER_FRACTION * float(radius), 0.0),
    )
    if source is None:
        return None, None
    target = nearest_safe_node(
        positions,
        safe_nodes,
        (STATIC_PAIR_INNER_FRACTION * float(radius), 0.0),
        excluded={source},
    )
    return source, target

# Assemble the routing context for one network.
def prepare_network_context(cache_path, prob_comp, degree_threshold):
    payload = load_network_cache(cache_path)
    graph = payload["graph"]
    positions = payload["positions"]
    descriptor = payload["descriptor"]
    safe_nodes, compromised_nodes = classify_network_nodes(
        graph,
        payload["compromise_rng_state"],
        prob_comp,
        degree_threshold,
    )
    source, target = select_static_pair(
        positions, safe_nodes, descriptor["radius"]
    )
    return {
        "cache_path": cache_path,
        "descriptor": descriptor,
        "graph": graph,
        "positions": positions,
        "safe_nodes": safe_nodes,
        "compromised_nodes": compromised_nodes,
        "source": source,
        "target": target,
        "logical_adjacency": {},
    }

# Select the best P2P rate for a physical link.
def get_optimal_p2p_skr(distance):
    snapped = round(round(float(distance) / SNAP_P2P) * SNAP_P2P, 2)
    values = []
    for r_db in range(int(R_DB_MIN), int(R_DB_MAX) + 1):
        key = (round(float(r_db), 1), snapped)
        if key in P2P_CACHE:
            values.append((float(P2P_CACHE[key]), float(r_db)))
    if not values:
        return -math.inf, 0.0
    best_rate, best_r_db = max(values, key=lambda item: item[0])
    if best_rate <= 0.0:
        return -math.inf, best_r_db
    return best_rate, best_r_db

# Evaluate the bosonic entropy term.
def bosonic_entropy(symplectic_eigenvalue):
    if symplectic_eigenvalue < 1.0 - 1e-8:
        raise ValueError(
            f"Nonphysical symplectic eigenvalue: {symplectic_eigenvalue}"
        )
    nu = max(float(symplectic_eigenvalue), 1.0)
    nbar = (nu - 1.0) / 2.0
    if nbar == 0.0:
        return 0.0
    return (nbar + 1.0) * math.log2(nbar + 1.0) - nbar * math.log2(nbar)

# Compute the covariance symplectic eigenvalues.
def symplectic_eigenvalues(a, b, c):
    delta = a * a + b * b - 2.0 * c * c
    determinant = (a * b - c * c) ** 2
    discriminant = delta * delta - 4.0 * determinant
    if discriminant < -1e-10:
        raise ValueError(f"Negative symplectic discriminant: {discriminant}")
    root = math.sqrt(max(discriminant, 0.0))
    nu1_sq = (delta - root) / 2.0
    nu2_sq = (delta + root) / 2.0
    if nu1_sq < -1e-10 or nu2_sq < -1e-10:
        raise ValueError(f"Negative symplectic eigenvalue square: {nu1_sq}, {nu2_sq}")
    nu1 = math.sqrt(max(nu1_sq, 0.0))
    nu2 = math.sqrt(max(nu2_sq, 0.0))
    if nu1 < 1.0 - 1e-8 or nu2 < 1.0 - 1e-8:
        raise ValueError(f"Nonphysical symplectic eigenvalues: {nu1}, {nu2}")
    return max(nu1, 1.0), max(nu2, 1.0)

# Build one elementary Gaussian link covariance.
def elementary_link(distance, r_db):
    r = float(r_db) * math.log(10.0) / 20.0
    mu = math.cosh(2.0 * r)
    transmittance = 10.0 ** (
        -float(SWAP_ALPHA_DB_PER_KM) * float(distance) / 10.0
    )
    a = mu
    b = transmittance * mu + 1.0 - transmittance + float(SWAP_XI)
    c = math.sqrt(transmittance * (mu * mu - 1.0))
    return a, b, c

# Combine two link covariances through swapping.
def swap_covariance(left, right):
    a1, b1, c1 = left
    a2, b2, c2 = right
    denominator = (
        float(SWAP_ETA_BELL) * (b1 + a2) + 1.0 - float(SWAP_ETA_BELL)
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError(f"Invalid swap denominator: {denominator}")
    return (
        a1 - float(SWAP_ETA_BELL) * c1 * c1 / denominator,
        b2 - float(SWAP_ETA_BELL) * c2 * c2 / denominator,
        float(SWAP_ETA_BELL) * c1 * c2 / denominator,
    )

# Evaluate the composable key rate of a swap chain.
def composable_swap_key(distances, r_db):
    distances = tuple(float(value) for value in distances)
    if len(distances) not in (2, 3, 4):
        raise ValueError("A swap chain must contain 2, 3, or 4 links")
    if min(distances) <= 0.0:
        raise ValueError(f"Nonpositive physical distance: {distances}")

    links = [elementary_link(distance, r_db) for distance in distances]
    if len(links) == 2:
        a, b, c = swap_covariance(links[0], links[1])
    elif len(links) == 3:
        a, b, c = swap_covariance(swap_covariance(links[0], links[1]), links[2])
    else:
        a, b, c = swap_covariance(
            swap_covariance(links[0], links[1]),
            swap_covariance(links[2], links[3]),
        )

    m_pe = int(SWAP_BLOCK_SIZE * SWAP_PE_FRACTION)
    n_key = int(SWAP_BLOCK_SIZE) - m_pe
    p_ec = 1.0 - float(SWAP_FER)
    if m_pe <= 0 or n_key <= 0:
        raise ValueError("Invalid finite-size block split")
    if not 0.0 < p_ec <= 1.0:
        raise ValueError(f"Invalid error-correction success probability: {p_ec}")

    delta_pe = math.sqrt(4.0 * math.log(8.0 / SWAP_EPS_PE) / m_pe)
    a_wc = a + delta_pe * (a + 1.0)
    b_wc = b + delta_pe * (b + 1.0)
    c_wc = c - 0.5 * delta_pe * (a + b + 2.0)

    nu1, nu2 = symplectic_eigenvalues(a_wc, b_wc, c_wc)
    nu_cond = a_wc - c_wc * c_wc / (b_wc + 1.0)
    if nu_cond < 1.0 - 1e-8:
        raise ValueError(f"Nonphysical conditional eigenvalue: {nu_cond}")
    nu_cond = max(nu_cond, 1.0)

    i_ab = math.log2((a_wc + 1.0) / (nu_cond + 1.0))
    chi_eb = bosonic_entropy(nu1) + bosonic_entropy(nu2) - bosonic_entropy(nu_cond)
    asymptotic_term = SWAP_BETA * i_ab - chi_eb

    delta_aep = (
        4.0
        * math.log2(math.sqrt(float(SWAP_ALPHABET_SIZE)) + 2.0)
        * math.sqrt(
            math.log2(
                18.0 * p_ec ** (-2.0) * float(SWAP_EPS_SMOOTH) ** (-4.0)
            )
        )
    )
    theta = math.log2(
        p_ec * (1.0 - float(SWAP_EPS_SMOOTH) ** 2 / 3.0)
    ) + 2.0 * math.log2(math.sqrt(2.0) * float(SWAP_EPS_HASH))

    return (n_key * p_ec / SWAP_BLOCK_SIZE) * (
        asymptotic_term - delta_aep / math.sqrt(n_key) + theta / n_key
    )

# Round link distances for rate-cache reuse.
def rounded_distances(distances):
    return tuple(
        round(float(value), int(LIVE_RATE_DISTANCE_DECIMALS))
        for value in distances
    )

# Evaluate and cache a swap rate at one squeezing value.
def live_key_for_rdb(distances, r_db):
    key = (rounded_distances(distances), int(r_db))
    if key not in LIVE_SWAP_RDB_CACHE:
        LIVE_SWAP_RDB_CACHE[key] = float(composable_swap_key(key[0], key[1]))
    return LIVE_SWAP_RDB_CACHE[key]

# Search the squeezing grid for the best swap rate.
def optimize_live_swap_skr(distances):
    key = rounded_distances(distances)
    if key in OPTIMIZED_SWAP_CACHE:
        return OPTIMIZED_SWAP_CACHE[key]

    start = int(SWAP_R_DB_START)
    if not SWAP_R_DB_MIN <= start <= SWAP_R_DB_MAX:
        raise ValueError("SWAP_R_DB_START must be inside the configured range")
    evaluated = {}

    def evaluate(r_db):
        if int(r_db) not in evaluated:
            evaluated[int(r_db)] = live_key_for_rdb(key, int(r_db))
        return evaluated[int(r_db)]

    candidates = [(evaluate(start), start)]
    if start > SWAP_R_DB_MIN:
        candidates.append((evaluate(start - 1), start - 1))
    if start < SWAP_R_DB_MAX:
        candidates.append((evaluate(start + 1), start + 1))

    best_rate, best_r_db = max(candidates, key=lambda item: item[0])
    start_rate = evaluate(start)
    direction = 0
    if best_rate > start_rate + SWAP_R_DB_IMPROVEMENT_TOL:
        direction = -1 if best_r_db < start else 1

    current_r_db = best_r_db
    while direction:
        next_r_db = current_r_db + direction
        if next_r_db < SWAP_R_DB_MIN or next_r_db > SWAP_R_DB_MAX:
            break
        next_rate = evaluate(next_r_db)
        if next_rate > best_rate + SWAP_R_DB_IMPROVEMENT_TOL:
            best_rate = next_rate
            best_r_db = next_r_db
            current_r_db = next_r_db
        else:
            break

    result = (-math.inf, float(best_r_db)) if best_rate <= 0.0 else (float(best_rate), float(best_r_db))
    OPTIMIZED_SWAP_CACHE[key] = result
    return result

# Normalize an undirected physical-edge key.
def edge_key(u, v):
    return (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))

# Compute the unweighted QUR transition-cost terms.
def transition_cost_bases(rate, segment, segment_distance, swap_relays, context):
    radius = float(context["descriptor"]["radius"])
    positions = context["positions"]
    target = context["target"]
    key_base = max(0.0, -math.log10(float(rate)))
    distance_base = float(segment_distance) / radius
    relay_base = float(swap_relays)
    hop_base = 1.0

    if swap_relays:
        target_distances = [
            euclidean_distance(node, target, positions) for node in segment
        ]
        outward = sum(
            max(0.0, target_distances[i + 1] - target_distances[i])
            for i in range(len(target_distances) - 1)
        )
        away_base = outward / radius
        direct = euclidean_distance(segment[0], segment[-1], positions)
        detour_base = max(0.0, float(segment_distance) - direct) / radius
    else:
        away_base = 0.0
        detour_base = 0.0

    return (
        key_base,
        distance_base,
        relay_base,
        hop_base,
        away_base,
        detour_base,
    )

# Package one logical routing transition.
def make_transition(segment, distances, rate, r_db, context):
    segment = tuple(int(node) for node in segment)
    distances = tuple(float(value) for value in distances)
    swap_relays = max(len(distances) - 1, 0)
    total_distance = float(sum(distances))
    return {
        "next_node": segment[-1],
        "segment": segment,
        "physical_edges": tuple(
            edge_key(u, v) for u, v in zip(segment, segment[1:])
        ),
        "distances": distances,
        "rate": float(rate),
        "r_db": float(r_db),
        "distance": total_distance,
        "swap_relays": int(swap_relays),
        "cost_bases": transition_cost_bases(
            rate, segment, total_distance, swap_relays, context
        ),
    }

# Build valid P2P and swap transitions from a node.
def generate_logical_transitions(context, current):
    graph = context["graph"]
    compromised = context["compromised_nodes"]
    if current in compromised or not graph.has_node(current):
        return []

    transitions = []
    for neighbor in graph.neighbors(current):
        if neighbor in compromised:
            continue
        distance = float(graph[current][neighbor]["true_weight"])
        rate, r_db = get_optimal_p2p_skr(distance)
        if rate > 0.0:
            transitions.append(
                make_transition([current, neighbor], [distance], rate, r_db, context)
            )

    def explore(prefix_nodes, prefix_distances):
        last = prefix_nodes[-1]
        physical_nodes = [current] + prefix_nodes
        excluded = set(physical_nodes)

        terminals = []
        for terminal in graph.neighbors(last):
            if terminal in compromised or terminal in excluded:
                continue
            terminals.append(
                (float(graph[last][terminal]["true_weight"]), int(terminal))
            )
        terminals.sort(key=lambda item: item[0])

        for final_distance, terminal in terminals:
            distances = prefix_distances + [final_distance]
            if MIN_SWAP_LINKS <= len(distances) <= MAX_SWAP_LINKS:
                rate, r_db = optimize_live_swap_skr(distances)
                if rate > 0.0:
                    transitions.append(
                        make_transition(
                            physical_nodes + [terminal],
                            distances,
                            rate,
                            r_db,
                            context,
                        )
                    )

        if len(prefix_distances) >= MAX_SWAP_LINKS - 1:
            return

        extensions = []
        for next_node in graph.neighbors(last):
            if next_node not in compromised or next_node in excluded:
                continue
            extensions.append(
                (float(graph[last][next_node]["true_weight"]), int(next_node))
            )
        extensions.sort(key=lambda item: item[0])

        for edge_distance, next_node in extensions:
            explore(
                prefix_nodes + [next_node],
                prefix_distances + [edge_distance],
            )

    first_untrusted = []
    for neighbor in graph.neighbors(current):
        if neighbor in compromised:
            first_untrusted.append(
                (float(graph[current][neighbor]["true_weight"]), int(neighbor))
            )
    first_untrusted.sort(key=lambda item: item[0])

    for distance, node in first_untrusted:
        explore([node], [distance])
    return transitions

# Return cached logical transitions for a node.
def get_logical_transitions(context, current):
    adjacency = context["logical_adjacency"]
    if current not in adjacency:
        adjacency[current] = generate_logical_transitions(context, current)
    return adjacency[current]

# Check whether a transition still has free resources.
def transition_available(transition, removed_nodes, removed_edges):
    if any(node in removed_nodes for node in transition["segment"]):
        return False
    if any(edge in removed_edges for edge in transition["physical_edges"]):
        return False
    return True

# Rebuild the full physical path from predecessors.
def reconstruct_physical_path(came_from, target):
    current = target
    reverse_segments = []
    while current in came_from:
        previous, segment = came_from[current]
        reverse_segments.append(segment)
        current = previous
    path = [current]
    for segment in reversed(reverse_segments):
        path.extend(segment[1:])
    return path

# Find one route using the A* distance objective.
def a_star_route(context, removed_nodes, removed_edges, weights=None):
    del weights
    source = context["source"]
    target = context["target"]
    positions = context["positions"]
    if source is None or target is None:
        raise nx.NetworkXNoPath

    queue = [(euclidean_distance(source, target, positions), 0.0, source)]
    g_score = {source: 0.0}
    bottleneck = {source: math.inf}
    came_from = {}

    while queue:
        _, current_g, current = heapq.heappop(queue)
        if current_g > g_score.get(current, math.inf) + 1e-15:
            continue
        if current == target:
            return reconstruct_physical_path(came_from, target), bottleneck[target]

        for transition in get_logical_transitions(context, current):
            if not transition_available(transition, removed_nodes, removed_edges):
                continue
            next_node = transition["next_node"]
            tentative = current_g + transition["distance"]
            if tentative < g_score.get(next_node, math.inf) - 1e-15:
                g_score[next_node] = tentative
                bottleneck[next_node] = min(
                    bottleneck[current], transition["rate"]
                )
                came_from[next_node] = (current, transition["segment"])
                priority = tentative + euclidean_distance(
                    next_node, target, positions
                )
                heapq.heappush(queue, (priority, tentative, next_node))

    raise nx.NetworkXNoPath

# Find one route using the weighted QUR objective.
def fsqur_route(context, removed_nodes, removed_edges, weights):
    source = context["source"]
    target = context["target"]
    positions = context["positions"]
    radius = float(context["descriptor"]["radius"])
    if source is None or target is None:
        raise nx.NetworkXNoPath

    weight_values = np.asarray(weights, dtype=float)
    if weight_values.shape != (7,):
        raise ValueError("FS-QUR weights must contain seven values")

    queue = [
        (
            weight_values[6] * euclidean_distance(source, target, positions) / radius,
            0.0,
            -math.inf,
            source,
        )
    ]
    best_g = {source: 0.0}
    bottleneck = {source: math.inf}
    came_from = {}

    while queue:
        _, current_g, _, current = heapq.heappop(queue)
        if current_g > best_g.get(current, math.inf) + 1e-15:
            continue
        if current == target:
            return reconstruct_physical_path(came_from, target), bottleneck[target]

        for transition in get_logical_transitions(context, current):
            if not transition_available(transition, removed_nodes, removed_edges):
                continue

            edge_cost = float(
                np.dot(weight_values[:6], np.asarray(transition["cost_bases"]))
            )
            if not math.isfinite(edge_cost):
                raise ValueError(f"Nonfinite FS-QUR edge cost: {edge_cost}")

            next_node = transition["next_node"]
            new_g = current_g + edge_cost
            new_bottleneck = min(bottleneck[current], transition["rate"])
            previous_g = best_g.get(next_node, math.inf)
            previous_bottleneck = bottleneck.get(next_node, -math.inf)

            better = new_g < previous_g - 1e-15
            tie_better = (
                math.isclose(new_g, previous_g, rel_tol=1e-9, abs_tol=1e-15)
                and new_bottleneck > previous_bottleneck
            )
            if better or tie_better:
                best_g[next_node] = new_g
                bottleneck[next_node] = new_bottleneck
                came_from[next_node] = (current, transition["segment"])
                heuristic_cost = (
                    weight_values[6]
                    * euclidean_distance(next_node, target, positions)
                    / radius
                )
                heapq.heappush(
                    queue,
                    (new_g + heuristic_cost, new_g, -new_bottleneck, next_node),
                )

    raise nx.NetworkXNoPath

# Reserve nodes and edges used by an accepted path.
def remove_path_resources(path, source, target, removed_nodes, removed_edges):
    for u, v in zip(path, path[1:]):
        removed_edges.add(edge_key(u, v))
    if REMOVE_USED_NODES:
        for node in path[1:-1]:
            if node not in (source, target):
                removed_nodes.add(int(node))

# Measure physical and logical properties of one path.
def evaluate_path_metrics(context, path):
    graph = context["graph"]
    compromised = context["compromised_nodes"]
    metrics = {
        "distance": 0.0,
        "p2p_links": 0,
        "swap_links": 0,
        "swap_2_links": 0,
        "swap_3_links": 0,
        "swap_4_links": 0,
        "r_db_sum": 0.0,
        "logical_links": 0,
    }

    index = 0
    while index < len(path) - 1:
        start = path[index]
        next_node = path[index + 1]
        if next_node not in compromised:
            distance = float(graph[start][next_node]["true_weight"])
            rate, r_db = get_optimal_p2p_skr(distance)
            if rate <= 0.0:
                raise ValueError("Routed P2P link has nonpositive key rate")
            metrics["distance"] += distance
            metrics["p2p_links"] += 1
            metrics["r_db_sum"] += r_db
            metrics["logical_links"] += 1
            index += 1
            continue

        distances = []
        terminal_index = index
        while terminal_index < len(path) - 1:
            u = path[terminal_index]
            v = path[terminal_index + 1]
            distances.append(float(graph[u][v]["true_weight"]))
            terminal_index += 1
            if path[terminal_index] not in compromised:
                break

        if path[terminal_index] in compromised or len(distances) not in (2, 3, 4):
            raise ValueError(f"Invalid routed swap segment: {path[index:terminal_index + 1]}")
        rate, r_db = optimize_live_swap_skr(distances)
        if rate <= 0.0:
            raise ValueError("Routed swap link has nonpositive key rate")

        links = len(distances)
        metrics["distance"] += sum(distances)
        metrics["swap_links"] += 1
        metrics[f"swap_{links}_links"] += 1
        metrics["r_db_sum"] += r_db
        metrics["logical_links"] += 1
        index = terminal_index
    return metrics

# Return an empty routing-metrics record.
def zero_route_metrics():
    return {
        "paths_found": 0,
        "K_sum": 0.0,
        "K_mean": 0.0,
        "n_mean": 0.0,
        "n_mean_including_endpoints": 0.0,
        "nodes_unique_total": 0,
        "nodes_unique_internal": 0,
        "path_distance_sum": 0.0,
        "path_distance_mean": 0.0,
        "p2p_links": 0,
        "swap_links": 0,
        "swap_2_links": 0,
        "swap_3_links": 0,
        "swap_4_links": 0,
        "avg_r_db": 0.0,
    }

# Find multiple resource-disjoint routes and summarize them.
def evaluate_multipath(context, route_function, max_paths, weights=None):
    if context["source"] is None or context["target"] is None:
        return zero_route_metrics()

    removed_nodes = set()
    removed_edges = set()
    path_rates = []
    node_counts = []
    node_counts_including = []
    unique_nodes = set()
    distance_sum = 0.0
    p2p_links = 0
    swap_links = 0
    swap_2_links = 0
    swap_3_links = 0
    swap_4_links = 0
    r_db_sum = 0.0
    logical_links = 0

    for _ in range(int(max_paths)):
        try:
            path, rate = route_function(
                context, removed_nodes, removed_edges, weights
            )
        except nx.NetworkXNoPath:
            break

        if rate <= 0.0:
            raise ValueError("Routing returned a nonpositive path key rate")

        path_metrics = evaluate_path_metrics(context, path)
        path_rates.append(float(rate))
        node_counts.append(
            float(len(path) if COUNT_ENDPOINTS_IN_N else max(len(path) - 2, 0))
        )
        node_counts_including.append(float(len(path)))
        unique_nodes.update(path)
        distance_sum += path_metrics["distance"]
        p2p_links += path_metrics["p2p_links"]
        swap_links += path_metrics["swap_links"]
        swap_2_links += path_metrics["swap_2_links"]
        swap_3_links += path_metrics["swap_3_links"]
        swap_4_links += path_metrics["swap_4_links"]
        r_db_sum += path_metrics["r_db_sum"]
        logical_links += path_metrics["logical_links"]
        remove_path_resources(
            path,
            context["source"],
            context["target"],
            removed_nodes,
            removed_edges,
        )

    if not path_rates:
        return zero_route_metrics()

    internal_unique = unique_nodes.difference(
        {context["source"], context["target"]}
    )
    return {
        "paths_found": len(path_rates),
        "K_sum": float(sum(path_rates)),
        "K_mean": float(np.mean(path_rates)),
        "n_mean": float(np.mean(node_counts)),
        "n_mean_including_endpoints": float(np.mean(node_counts_including)),
        "nodes_unique_total": len(unique_nodes),
        "nodes_unique_internal": len(internal_unique),
        "path_distance_sum": float(distance_sum),
        "path_distance_mean": float(distance_sum / len(path_rates)),
        "p2p_links": int(p2p_links),
        "swap_links": int(swap_links),
        "swap_2_links": int(swap_2_links),
        "swap_3_links": int(swap_3_links),
        "swap_4_links": int(swap_4_links),
        "avg_r_db": float(r_db_sum / logical_links) if logical_links else 0.0,
    }

# Evaluate the A* multipath baseline.
def evaluate_a_star(context, max_paths):
    return evaluate_multipath(context, a_star_route, max_paths)

# Evaluate QUR with a supplied weight vector.
def evaluate_fsqur(context, weights, max_paths):
    return evaluate_multipath(context, fsqur_route, max_paths, weights)

# Score QUR using key rate and node efficiency.
def score_fsqur(n_mean, k_mean, a_star_k_mean, penalty_strength=A_STAR_PENALTY_STRENGTH):
    if n_mean < 0.0 or k_mean <= 0.0:
        return -math.inf
    if math.isclose(k_mean, 1.0, rel_tol=0.0, abs_tol=1e-15):
        return -math.inf
    if n_mean == 0.0:
        base_score = math.inf
    else:
        base_score = -1.0 / (float(n_mean) * math.log10(float(k_mean)))

    relative_deficit = max(
        0.0,
        (float(a_star_k_mean) - float(k_mean)) / float(a_star_k_mean),
    )
    penalty = math.exp(-float(penalty_strength) * relative_deficit)
    return float(base_score * penalty)

# Sample unique QUR weights from the configured grid.
def sample_random_weights(iterations, random_seed):
    scale = 10 ** int(WEIGHT_DECIMALS)
    low = int(round(WEIGHT_MIN * scale))
    high = int(round(WEIGHT_MAX * scale))
    grid_size = high - low + 1
    total_configurations = grid_size**6
    if int(iterations) > total_configurations:
        raise ValueError("Requested more unique weights than the grid contains")

    rng = np.random.default_rng(int(random_seed))
    flat_indices = rng.choice(
        total_configurations, size=int(iterations), replace=False
    )
    sampled = np.empty((int(iterations), 6), dtype=float)
    values = flat_indices.copy()
    for column in range(5, -1, -1):
        sampled[:, column] = (values % grid_size + low) / scale
        values //= grid_size

    rate_column = np.full((int(iterations), 1), RATE_WEIGHT_FIXED, dtype=float)
    return np.hstack((rate_column, sampled))

# Apply deterministic tie-breaking between weight samples.
def candidate_is_better(score, metrics, sample_index, best):
    if best is None:
        return True
    if score > best["score"] + 1e-15:
        return True
    if not math.isclose(score, best["score"], rel_tol=1e-12, abs_tol=1e-15):
        return False
    if metrics["K_mean"] > best["metrics"]["K_mean"] + 1e-15:
        return True
    if not math.isclose(
        metrics["K_mean"],
        best["metrics"]["K_mean"],
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return False
    if metrics["n_mean"] < best["metrics"]["n_mean"] - 1e-15:
        return True
    if not math.isclose(
        metrics["n_mean"],
        best["metrics"]["n_mean"],
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return False
    return sample_index < best["sample_index"]

# Flatten the routing graph for fast Numba evaluation.
def build_numba_model(context):
    if context["source"] is None or context["target"] is None:
        return None
    if "numba_model" in context:
        return context["numba_model"]

    graph = context["graph"]
    number_of_nodes = graph.number_of_nodes()
    if set(graph.nodes()) != set(range(number_of_nodes)):
        raise ValueError("Numba routing requires node labels 0,1,...,N-1")

    for node in sorted(context["safe_nodes"]):
        get_logical_transitions(context, int(node))

    physical_edge_to_id = {
        edge_key(u, v): edge_id
        for edge_id, (u, v) in enumerate(graph.edges())
    }
    number_of_edges = len(physical_edge_to_id)
    number_of_transitions = sum(
        len(context["logical_adjacency"].get(node, ()))
        for node in range(number_of_nodes)
    )

    adjacency_offsets = np.zeros(number_of_nodes + 1, dtype=np.int64)
    next_nodes = np.empty(number_of_transitions, dtype=np.int32)
    rates = np.empty(number_of_transitions, dtype=np.float64)
    cost_bases = np.empty((number_of_transitions, 6), dtype=np.float64)

    max_segment_nodes = MAX_SWAP_LINKS + 1
    max_segment_edges = MAX_SWAP_LINKS
    segment_nodes = np.full(
        (number_of_transitions, max_segment_nodes), -1, dtype=np.int32
    )
    segment_lengths = np.zeros(number_of_transitions, dtype=np.int8)
    transition_edge_ids = np.full(
        (number_of_transitions, max_segment_edges), -1, dtype=np.int32
    )
    transition_edge_lengths = np.zeros(number_of_transitions, dtype=np.int8)

    cursor = 0
    for node in range(number_of_nodes):
        adjacency_offsets[node] = cursor
        for transition in context["logical_adjacency"].get(node, ()):
            next_nodes[cursor] = int(transition["next_node"])
            rates[cursor] = float(transition["rate"])
            cost_bases[cursor, :] = np.asarray(
                transition["cost_bases"], dtype=np.float64
            )

            segment = transition["segment"]
            segment_length = len(segment)
            if segment_length > max_segment_nodes:
                raise ValueError("Logical transition exceeds MAX_SWAP_LINKS + 1")
            segment_lengths[cursor] = segment_length
            segment_nodes[cursor, :segment_length] = np.asarray(
                segment, dtype=np.int32
            )

            physical_edges = transition["physical_edges"]
            edge_length = len(physical_edges)
            if edge_length > max_segment_edges:
                raise ValueError("Logical transition exceeds MAX_SWAP_LINKS")
            transition_edge_lengths[cursor] = edge_length
            for local_index, physical_edge in enumerate(physical_edges):
                transition_edge_ids[cursor, local_index] = physical_edge_to_id[
                    edge_key(*physical_edge)
                ]
            cursor += 1
    adjacency_offsets[number_of_nodes] = cursor

    radius = float(context["descriptor"]["radius"])
    target = int(context["target"])
    heuristic_bases = np.zeros(number_of_nodes, dtype=np.float64)
    for node in range(number_of_nodes):
        heuristic_bases[node] = (
            euclidean_distance(node, target, context["positions"]) / radius
        )

    model = {
        "adjacency_offsets": np.ascontiguousarray(adjacency_offsets),
        "next_nodes": np.ascontiguousarray(next_nodes),
        "rates": np.ascontiguousarray(rates),
        "cost_bases": np.ascontiguousarray(cost_bases),
        "segment_nodes": np.ascontiguousarray(segment_nodes),
        "segment_lengths": np.ascontiguousarray(segment_lengths),
        "transition_edge_ids": np.ascontiguousarray(transition_edge_ids),
        "transition_edge_lengths": np.ascontiguousarray(transition_edge_lengths),
        "heuristic_bases": np.ascontiguousarray(heuristic_bases),
        "source": int(context["source"]),
        "target": target,
        "number_of_edges": int(number_of_edges),
    }
    context["numba_model"] = model
    return model

# Compare floating-point values inside Numba code.
@njit(cache=True, inline="always")
def _numba_isclose(a, b):
    tolerance = max(1e-15, 1e-9 * max(abs(a), abs(b)))
    return abs(a - b) <= tolerance

# Check transition availability inside Numba routing.
@njit(cache=True, inline="always")
def _numba_transition_available(
    transition_id,
    removed_nodes,
    removed_edges,
    segment_nodes,
    segment_lengths,
    transition_edge_ids,
    transition_edge_lengths,
):
    for index in range(int(segment_lengths[transition_id])):
        node = int(segment_nodes[transition_id, index])
        if removed_nodes[node]:
            return False

    for index in range(int(transition_edge_lengths[transition_id])):
        edge_id = int(transition_edge_ids[transition_id, index])
        if removed_edges[edge_id]:
            return False
    return True

# Run one compiled QUR route search.
@njit(cache=True)
def _numba_route_once(
    weights,
    adjacency_offsets,
    next_nodes,
    rates,
    cost_bases,
    segment_nodes,
    segment_lengths,
    transition_edge_ids,
    transition_edge_lengths,
    heuristic_bases,
    source,
    target,
    removed_nodes,
    removed_edges,
    count_endpoints,
):
    number_of_nodes = adjacency_offsets.shape[0] - 1
    best_g = np.full(number_of_nodes, np.inf, dtype=np.float64)
    bottleneck = np.zeros(number_of_nodes, dtype=np.float64)
    predecessor_node = np.full(number_of_nodes, -1, dtype=np.int32)
    predecessor_transition = np.full(number_of_nodes, -1, dtype=np.int32)

    best_g[source] = 0.0
    bottleneck[source] = np.inf

    queue = [(0.0, 0.0, 0.0, 0)]
    queue.pop()
    heapq.heappush(
        queue,
        (weights[6] * heuristic_bases[source], 0.0, -np.inf, source),
    )

    found = False
    while len(queue) > 0:
        _, current_g, _, current = heapq.heappop(queue)
        if current_g > best_g[current] + 1e-15:
            continue
        if current == target:
            found = True
            break

        begin = int(adjacency_offsets[current])
        end = int(adjacency_offsets[current + 1])
        for transition_id in range(begin, end):
            if not _numba_transition_available(
                transition_id,
                removed_nodes,
                removed_edges,
                segment_nodes,
                segment_lengths,
                transition_edge_ids,
                transition_edge_lengths,
            ):
                continue

            edge_cost = 0.0
            for component in range(6):
                edge_cost += weights[component] * cost_bases[
                    transition_id, component
                ]
            if not math.isfinite(edge_cost):
                raise ValueError("Nonfinite FS-QUR edge cost")

            next_node = int(next_nodes[transition_id])
            new_g = current_g + edge_cost
            new_bottleneck = min(bottleneck[current], rates[transition_id])
            previous_g = best_g[next_node]
            previous_bottleneck = bottleneck[next_node]

            better = new_g < previous_g - 1e-15
            tie_better = (
                _numba_isclose(new_g, previous_g)
                and new_bottleneck > previous_bottleneck
            )
            if better or tie_better:
                best_g[next_node] = new_g
                bottleneck[next_node] = new_bottleneck
                predecessor_node[next_node] = current
                predecessor_transition[next_node] = transition_id
                heuristic_cost = weights[6] * heuristic_bases[next_node]
                heapq.heappush(
                    queue,
                    (
                        new_g + heuristic_cost,
                        new_g,
                        -new_bottleneck,
                        np.int64(next_node),
                    ),
                )

    if not found:
        return 0.0, 0.0, 0

    physical_node_count = 1
    current = target
    reconstruction_steps = 0
    while current != source:
        transition_id = int(predecessor_transition[current])
        previous = int(predecessor_node[current])
        if transition_id < 0 or previous < 0:
            raise RuntimeError("Invalid Numba predecessor chain")

        physical_node_count += int(segment_lengths[transition_id]) - 1

        for index in range(int(transition_edge_lengths[transition_id])):
            edge_id = int(transition_edge_ids[transition_id, index])
            removed_edges[edge_id] = True

        if REMOVE_USED_NODES:
            for index in range(int(segment_lengths[transition_id])):
                node = int(segment_nodes[transition_id, index])
                if node != source and node != target:
                    removed_nodes[node] = True

        current = previous
        reconstruction_steps += 1
        if reconstruction_steps > number_of_nodes:
            raise RuntimeError("Cyclic Numba predecessor chain")

    if not count_endpoints:
        physical_node_count = max(physical_node_count - 2, 0)

    return float(physical_node_count), float(bottleneck[target]), 1

# Evaluate one weight vector across multiple paths.
@njit(cache=True)
def _numba_evaluate_one(
    weights,
    max_paths,
    adjacency_offsets,
    next_nodes,
    rates,
    cost_bases,
    segment_nodes,
    segment_lengths,
    transition_edge_ids,
    transition_edge_lengths,
    heuristic_bases,
    source,
    target,
    number_of_edges,
    count_endpoints,
):
    number_of_nodes = adjacency_offsets.shape[0] - 1
    removed_nodes = np.zeros(number_of_nodes, dtype=np.bool_)
    removed_edges = np.zeros(number_of_edges, dtype=np.bool_)

    total_nodes = 0.0
    total_rate = 0.0
    paths_found = 0

    for _ in range(max_paths):
        node_count, path_rate, success = _numba_route_once(
            weights,
            adjacency_offsets,
            next_nodes,
            rates,
            cost_bases,
            segment_nodes,
            segment_lengths,
            transition_edge_ids,
            transition_edge_lengths,
            heuristic_bases,
            source,
            target,
            removed_nodes,
            removed_edges,
            count_endpoints,
        )
        if success == 0:
            break
        if path_rate <= 0.0:
            raise ValueError("Numba routing returned a nonpositive key rate")
        total_nodes += node_count
        total_rate += path_rate
        paths_found += 1

    if paths_found == 0:
        return 0.0, 0.0, 0
    return (
        total_nodes / paths_found,
        total_rate / paths_found,
        paths_found,
    )

# Evaluate a batch of QUR weight vectors in Numba.
@njit(cache=True, parallel=NUMBA_PARALLEL_WITHIN_JOB)
def _numba_evaluate_batch(
    weight_matrix,
    max_paths,
    adjacency_offsets,
    next_nodes,
    rates,
    cost_bases,
    segment_nodes,
    segment_lengths,
    transition_edge_ids,
    transition_edge_lengths,
    heuristic_bases,
    source,
    target,
    number_of_edges,
    count_endpoints,
):
    output = np.empty((weight_matrix.shape[0], 3), dtype=np.float64)
    for sample_index in prange(weight_matrix.shape[0]):
        n_mean, k_mean, paths_found = _numba_evaluate_one(
            weight_matrix[sample_index],
            max_paths,
            adjacency_offsets,
            next_nodes,
            rates,
            cost_bases,
            segment_nodes,
            segment_lengths,
            transition_edge_ids,
            transition_edge_lengths,
            heuristic_bases,
            source,
            target,
            number_of_edges,
            count_endpoints,
        )
        output[sample_index, 0] = n_mean
        output[sample_index, 1] = k_mean
        output[sample_index, 2] = paths_found
    return output

# Evaluate many QUR weights using the compiled model.
def evaluate_fsqur_weight_batch(context, weight_matrix, max_paths):
    model = build_numba_model(context)
    if model is None:
        return np.zeros((len(weight_matrix), 3), dtype=np.float64)

    configure_numba_threads(NUMBA_THREADS_PER_JOB)
    return _numba_evaluate_batch(
        np.ascontiguousarray(weight_matrix, dtype=np.float64),
        int(max_paths),
        model["adjacency_offsets"],
        model["next_nodes"],
        model["rates"],
        model["cost_bases"],
        model["segment_nodes"],
        model["segment_lengths"],
        model["transition_edge_ids"],
        model["transition_edge_lengths"],
        model["heuristic_bases"],
        int(model["source"]),
        int(model["target"]),
        int(model["number_of_edges"]),
        bool(COUNT_ENDPOINTS_IN_N),
    )

# Compile and sanity-check the Numba kernels.
def warmup_numba(thread_count=None):
    if thread_count is not None:
        configure_numba_threads(thread_count)
    adjacency_offsets = np.asarray([0, 1, 1], dtype=np.int64)
    next_nodes = np.asarray([1], dtype=np.int32)
    rates = np.asarray([0.1], dtype=np.float64)
    cost_bases = np.asarray([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    segment_nodes = np.asarray([[0, 1, -1, -1, -1]], dtype=np.int32)
    segment_lengths = np.asarray([2], dtype=np.int8)
    transition_edge_ids = np.asarray([[0, -1, -1, -1]], dtype=np.int32)
    transition_edge_lengths = np.asarray([1], dtype=np.int8)
    heuristic_bases = np.asarray([1.0, 0.0], dtype=np.float64)
    weights = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    configure_numba_threads(NUMBA_THREADS_PER_JOB)
    result = _numba_evaluate_batch(
        weights,
        1,
        adjacency_offsets,
        next_nodes,
        rates,
        cost_bases,
        segment_nodes,
        segment_lengths,
        transition_edge_ids,
        transition_edge_lengths,
        heuristic_bases,
        0,
        1,
        1,
        False,
    )
    if result[0, 2] != 1.0:
        raise RuntimeError("Numba warm-up failed")

# Search sampled weights and return the best QUR result.
def optimize_weights(
    context,
    max_paths,
    a_star_metrics,
    iterations=OPTIMIZATION_ITERATIONS,
    random_seed=OPTIMIZATION_RANDOM_SEED,
    penalty_strength=A_STAR_PENALTY_STRENGTH,
):
    if a_star_metrics["paths_found"] == 0:
        return {
            "weights": np.array([RATE_WEIGHT_FIXED, 0, 0, 0, 0, 0, 0], dtype=float),
            "score": 0.0,
            "sample_index": -1,
            "metrics": zero_route_metrics(),
            "iterations": int(iterations),
            "random_seed": int(random_seed),
            "optimization_seconds": 0.0,
            "weight_trials_per_second": 0.0,
            "numba_threads": int(NUMBA_THREADS_PER_JOB),
        }

    sampled_weights = sample_random_weights(iterations, random_seed)
    optimization_start = time.perf_counter()
    batch_results = evaluate_fsqur_weight_batch(
        context, sampled_weights, max_paths
    )
    optimization_seconds = time.perf_counter() - optimization_start
    weight_trials_per_second = (
        float(iterations) / optimization_seconds
        if optimization_seconds > 0.0
        else math.inf
    )

    best = None
    for sample_index in range(len(sampled_weights)):
        metrics = {
            "paths_found": int(batch_results[sample_index, 2]),
            "n_mean": float(batch_results[sample_index, 0]),
            "K_mean": float(batch_results[sample_index, 1]),
        }
        if metrics["paths_found"] == 0:
            score = -math.inf
        else:
            score = score_fsqur(
                metrics["n_mean"],
                metrics["K_mean"],
                a_star_metrics["K_mean"],
                penalty_strength,
            )
        if candidate_is_better(score, metrics, sample_index, best):
            best = {
                "weights": sampled_weights[sample_index].copy(),
                "score": float(score),
                "sample_index": int(sample_index),
                "metrics": metrics,
                "iterations": int(iterations),
                "random_seed": int(random_seed),
            }

    if best is None:
        raise RuntimeError("The optimizer evaluated no weight configurations")

    best["optimization_seconds"] = float(optimization_seconds)
    best["weight_trials_per_second"] = float(weight_trials_per_second)
    best["numba_threads"] = int(NUMBA_THREADS_PER_JOB)

    full_metrics = evaluate_fsqur(context, best["weights"], max_paths)
    compiled_metrics = best["metrics"]
    if int(full_metrics["paths_found"]) != int(compiled_metrics["paths_found"]):
        raise RuntimeError("Numba/Python FS-QUR mismatch in paths_found")
    if not math.isclose(
        float(full_metrics["K_mean"]),
        float(compiled_metrics["K_mean"]),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Numba/Python FS-QUR mismatch in K_mean")
    if not math.isclose(
        float(full_metrics["n_mean"]),
        float(compiled_metrics["n_mean"]),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Numba/Python FS-QUR mismatch in n_mean")

    best["metrics"] = full_metrics
    return best

# Map the QUR weight vector to named values.
def weights_to_dict(weights):
    return {name: float(value) for name, value in zip(WEIGHT_NAMES, weights)}

# Parse arguments and run the workflow.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_path")
    parser.add_argument("--degree-threshold", type=int, default=10)
    parser.add_argument("--prob-comp", type=float, default=0.0)
    parser.add_argument("--max-paths", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=OPTIMIZATION_ITERATIONS)
    parser.add_argument("--random-seed", type=int, default=OPTIMIZATION_RANDOM_SEED)
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=NUMBA_THREADS_STANDALONE,
    )
    args = parser.parse_args()

    configure_numba_threads(args.numba_threads)
    warmup_numba()
    load_p2p_data()
    context = prepare_network_context(
        args.cache_path, args.prob_comp, args.degree_threshold
    )
    a_star_metrics = evaluate_a_star(context, args.max_paths)
    result = optimize_weights(
        context,
        args.max_paths,
        a_star_metrics,
        args.iterations,
        args.random_seed,
    )
    output = {
        "weights": weights_to_dict(result["weights"]),
        "score": result["score"],
        "sample_index": result["sample_index"],
        "optimization_seconds": result["optimization_seconds"],
        "weight_trials_per_second": result["weight_trials_per_second"],
        "numba_threads": result["numba_threads"],
        "a_star": a_star_metrics,
        "fsqur": result["metrics"],
    }
    print(json.dumps(output, indent=2))

# Run the command-line entry point.
if __name__ == "__main__":
    main()
