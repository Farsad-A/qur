
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
})

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

import optimize_fsqur as opt

# Configure model and runtime parameters.
GLOBAL_RANDOM_SEED = 100
NUM_BATCH_RUNS = 50

USE_MULTIPROCESSING = True
MAX_WORKERS = 4
NUMBA_THREADS_PER_WORKER = 1
MP_CONTEXT = mp.get_context("spawn")


NUM_NODES = 1000
RADIUS = 50.0
R0_RATIO = 0.07
BETA_WAXMAN = 1.0
PROB_COMP = 0.0
DEGREE_THRESHOLD = 10
MAX_PATHS = 5


RADII_SCAN = list(range(50, 300, 50))
MAX_PATHS_SCAN = [1, 2, 3, 5, 10]
DEGREE_THRESHOLD_SCAN = list(range(14, 2, -1))
NUM_NODES_SCAN = list(range(200, 1000, 100))

ENABLED_SCANS = [
    "radius",
    "max_paths",
    "degree_threshold",
    "num_nodes",
]

SCAN_VALUE_MAP = {
    "radius": RADII_SCAN,
    "max_paths": MAX_PATHS_SCAN,
    "degree_threshold": DEGREE_THRESHOLD_SCAN,
    "num_nodes": NUM_NODES_SCAN,
}

SCAN_LABEL_MAP = {
    "radius": "Network radius R (km)",
    "max_paths": "Maximum allowed paths",
    "degree_threshold": "Degree compromise threshold",
    "num_nodes": "Number of nodes",
}
FIXED_WEIGHT_SCOPE = "per_scan_point"
WEIGHT_TRIALS = opt.OPTIMIZATION_ITERATIONS
TOP_CANDIDATES_PER_GRAPH = 500
EXTRA_FIXED_CANDIDATES = 10_000
ML_CANDIDATE_BATCH_SIZE = 5_000
MODEL_RANDOM_SEED = 20260811
MODEL_MAX_ITER = 300
MODEL_LEARNING_RATE = 0.06
MODEL_MAX_LEAF_NODES = 31
MODEL_MIN_SAMPLES_LEAF = 20
MODEL_L2_REGULARIZATION = 1.0
CLEAR_GRAPH_CACHES_AFTER_EACH_JOB = True
RUN_GC_AFTER_EACH_JOB = True
RETURN_FREE_HEAP_TO_OS = True
RESULTS_CSV = Path("qkd_network_scan_results_4alg.csv")
FIXED_WEIGHTS_CSV = Path("learned_fixed_weights_by_configuration.csv")
LANDSCAPE_CACHE_DIR = Path("scan_ml_landscape_cache")
FIXED_EVAL_CACHE_DIR = Path("scan_fixed_eval_cache")
FIGURE_DIR = Path("four_algorithm_figures")
EXPERIMENT_VERSION = 1
LATEX_VALUES_FILE = Path(
    "qur_claims_values.tex"
)
ALGORITHM_ORDER = [
    "A*",
    "Max-Min",
    "QUR-Optimized",
    "QUR-Fixed-Learned",
]
DISPLAY = {
    "A*": "A*",
    "Max-Min": "Max--Min",
    "QUR-Optimized": "QUR optimized per graph",
    "QUR-Fixed-Learned": "QUR fixed learned weights",
}
COLORS = {
    "A*": "#1F77B4",
    "Max-Min": "#2CA02C",
    "QUR-Optimized": "#FF7F0E",
    "QUR-Fixed-Learned": "#D62728",
}
MARKERS = {
    "A*": "o",
    "Max-Min": "s",
    "QUR-Optimized": "^",
    "QUR-Fixed-Learned": "D",
}
BAR_REFERENCE_COLORS = {
    "A*": COLORS["A*"],
    "Max-Min": COLORS["Max-Min"],
}
BAR_OPTIMIZED_ALPHA = 0.90
BAR_FIXED_ALPHA = 0.40
OVERALL_SOURCE_SCAN = "degree_threshold"
REUSE_COMPLETE_RESULTS = True
RESULTS_SEARCH_SPACE_JSON = Path(
    str(RESULTS_CSV) + ".search_space.json"
)
PLOT_DPI = 300
PLOT_PAD_INCHES = 0.35
REUSE_TOLERANCE = 1e-9

# Build a network settings dictionary.
def make_settings(**overrides) -> dict:
    settings = {
        "num_nodes": int(NUM_NODES),
        "radius": float(RADIUS),
        "r0_ratio": float(R0_RATIO),
        "beta_waxman": float(BETA_WAXMAN),
        "prob_comp": float(PROB_COMP),
        "degree_threshold": int(DEGREE_THRESHOLD),
        "max_paths": int(MAX_PATHS),
    }
    settings.update(overrides)
    return settings

# Apply one scanned value to the default settings.
def settings_for_scan(scan_name: str, value) -> dict:
    settings = make_settings()
    if scan_name == "radius":
        settings["radius"] = float(value)
    elif scan_name == "max_paths":
        settings["max_paths"] = int(value)
    elif scan_name == "degree_threshold":
        settings["degree_threshold"] = int(value)
    elif scan_name == "num_nodes":
        settings["num_nodes"] = int(value)
    else:
        raise ValueError(f"Unknown scan: {scan_name}")
    return settings

# Convert settings into a topology cache.
def cache_task_from_settings(seed: int, settings: dict) -> tuple:
    r0 = float(settings["r0_ratio"]) * float(settings["radius"])
    return (
        int(seed),
        int(settings["num_nodes"]),
        float(settings["beta_waxman"]),
        float(r0),
        float(settings["radius"]),
    )

# Create a topology cache through the optimizer module.
def create_cache_task(task: tuple) -> str:
    return opt.create_network_cache(*task)

# Create all graph jobs required by the enabled scans.
def build_scan_jobs() -> tuple[list[dict], list[tuple]]:
    jobs = []
    cache_tasks = set()
    for scan_name in ENABLED_SCANS:
        for x_value in SCAN_VALUE_MAP[scan_name]:
            settings = settings_for_scan(scan_name, x_value)
            for batch_index in range(NUM_BATCH_RUNS):
                seed = GLOBAL_RANDOM_SEED + batch_index
                cache_task = cache_task_from_settings(seed, settings)
                cache_tasks.add(cache_task)
                jobs.append(
                    {
                        "scan_name": scan_name,
                        "x_value": float(x_value),
                        "seed": int(seed),
                        "settings": settings,
                        "cache_path": opt.network_cache_path(*cache_task),
                    }
                )
    return jobs, sorted(cache_tasks)

# Build the fixed weight grouping key for a job.
def group_key(job: dict) -> tuple[str, float]:
    return job["scan_name"], round(float(job["x_value"]), 10)

# Readable label for a scan group.
def group_label(key: tuple[str, float]) -> str:
    scan_name, x_value = key
    return f"{scan_name}={x_value:g}"

# Derive a reproducible random seed.
def seed_from_parts(*parts) -> int:
    digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32 - 1)

# Hash the optimizer source for cache validation.
def optimizer_source_hash() -> str:
    path = Path(inspect.getfile(opt)).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Fingerprint the search space
def experiment_fingerprint() -> str:
    payload = {
        "experiment_version": int(EXPERIMENT_VERSION),
        "optimizer_sha256": optimizer_source_hash(),
        "global_seed": int(GLOBAL_RANDOM_SEED),
        "num_batch_runs": int(NUM_BATCH_RUNS),
        "weight_trials": int(WEIGHT_TRIALS),
        "weights": {
            "names": list(opt.WEIGHT_NAMES),
            "min": float(opt.WEIGHT_MIN),
            "max": float(opt.WEIGHT_MAX),
            "decimals": int(opt.WEIGHT_DECIMALS),
            "rate_fixed": float(opt.RATE_WEIGHT_FIXED),
            "a_star_penalty": float(opt.A_STAR_PENALTY_STRENGTH),
        },
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

# Generate the per graph optimizer seed.
def optimizer_seed(job: dict) -> int:
    settings = job["settings"]
    key = (
        job["scan_name"],
        round(float(job["x_value"]), 10),
        int(job["seed"]),
        int(settings["num_nodes"]),
        round(float(settings["radius"]), 10),
        int(settings["degree_threshold"]),
        int(settings["max_paths"]),
    )
    return seed_from_parts("per-seed-optimizer", key)

# Initialize lookup data and threads in each worker.
def worker_initializer() -> None:
    opt.configure_numba_threads(NUMBA_THREADS_PER_WORKER)
    opt.load_p2p_data()

# Compute a finite ratio
def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not math.isfinite(float(denominator)):
        return 0.0
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else 0.0

# Return basic summary statistics for an array.
def array_stats(values: Iterable[float]) -> tuple[float, float, float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.median(arr)),
        float(np.max(arr)),
    )

# Return unused memory to the operating system when possible.
def _malloc_trim() -> None:
    if not RETURN_FREE_HEAP_TO_OS:
        return
    try:
        libc = ctypes.CDLL(None)
        fn = getattr(libc, "malloc_trim", None)
        if fn is not None:
            fn.argtypes = [ctypes.c_size_t]
            fn.restype = ctypes.c_int
            fn(0)
    except Exception:
        pass

# Clear caches and reclaim worker memory.
def cleanup_worker_memory() -> None:
    if CLEAR_GRAPH_CACHES_AFTER_EACH_JOB:
        opt.OPTIMIZED_SWAP_CACHE.clear()
        opt.LIVE_SWAP_RDB_CACHE.clear()
    if RUN_GC_AFTER_EACH_JOB:
        gc.collect()
    _malloc_trim()

# Find the path with the largest bottleneck key rate.
def max_min_route(context, removed_nodes, removed_edges, weights=None):
    del weights
    source = context["source"]
    target = context["target"]
    if source is None or target is None:
        raise nx.NetworkXNoPath
    queue = [(-math.inf, source)]
    best_bottleneck = {source: math.inf}
    came_from = {}
    while queue:
        negative_value, current = heapq.heappop(queue)
        current_value = -negative_value
        if current_value < best_bottleneck.get(current, -math.inf) - 1e-15:
            continue
        if current == target:
            return (
                opt.reconstruct_physical_path(came_from, target),
                best_bottleneck[target],
            )
        for transition in opt.get_logical_transitions(context, current):
            if not opt.transition_available(transition, removed_nodes, removed_edges):
                continue
            next_node = transition["next_node"]
            candidate = min(current_value, transition["rate"])
            if candidate > best_bottleneck.get(next_node, -math.inf) + 1e-15:
                best_bottleneck[next_node] = candidate
                came_from[next_node] = (current, transition["segment"])
                heapq.heappush(queue, (-candidate, next_node))
    raise nx.NetworkXNoPath

# Evaluate the Max-Min multipath baseline.
def evaluate_max_min(context, max_paths: int) -> dict:
    return opt.evaluate_multipath(context, max_min_route, max_paths)

# Extract topology features for fixed weight learning.
def extract_graph_features(context: dict, a_star_metrics: dict, settings: dict) -> dict[str, float]:
    graph = context["graph"]
    source = context["source"]
    target = context["target"]
    positions = context["positions"]
    safe_nodes = context["safe_nodes"]
    compromised_nodes = context["compromised_nodes"]
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    radius = float(settings["radius"])
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    edge_lengths = np.asarray(
        [float(data["true_weight"]) for _, _, data in graph.edges(data=True)],
        dtype=float,
    )
    degree_mean, degree_std, degree_min, degree_median, degree_max = array_stats(degrees)
    edge_mean, edge_std, edge_min, edge_median, edge_max = array_stats(edge_lengths)
    graph_density = (
        2.0 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    )
    if source is None or target is None:
        source_target_distance = 0.0
        source_degree = 0.0
        target_degree = 0.0
        source_transitions = []
    else:
        source_target_distance = opt.euclidean_distance(source, target, positions)
        source_degree = float(graph.degree(source))
        target_degree = float(graph.degree(target))
        source_transitions = opt.get_logical_transitions(context, source)
    transition_rates = [t["rate"] for t in source_transitions]
    transition_distances = [t["distance"] for t in source_transitions]
    source_p2p = sum(t["swap_relays"] == 0 for t in source_transitions)
    source_swap = sum(t["swap_relays"] > 0 for t in source_transitions)
    rate_mean, rate_std, rate_min, rate_median, rate_max = array_stats(transition_rates)
    tr_mean, tr_std, tr_min, tr_median, tr_max = array_stats(transition_distances)
    a_paths = float(a_star_metrics["paths_found"])
    a_k = float(a_star_metrics["K_mean"])
    features = {
        "cfg_num_nodes": float(settings["num_nodes"]),
        "cfg_radius": float(settings["radius"]),
        "cfg_degree_threshold": float(settings["degree_threshold"]),
        "cfg_max_paths": float(settings["max_paths"]),
        "graph_edges": float(n_edges),
        "graph_density": float(graph_density),
        "degree_mean": degree_mean,
        "degree_std": degree_std,
        "degree_min": degree_min,
        "degree_median": degree_median,
        "degree_max": degree_max,
        "edge_length_mean_over_R": safe_ratio(edge_mean, radius),
        "edge_length_std_over_R": safe_ratio(edge_std, radius),
        "edge_length_min_over_R": safe_ratio(edge_min, radius),
        "edge_length_median_over_R": safe_ratio(edge_median, radius),
        "edge_length_max_over_R": safe_ratio(edge_max, radius),
        "trusted_fraction": safe_ratio(len(safe_nodes), n_nodes),
        "compromised_fraction": safe_ratio(len(compromised_nodes), n_nodes),
        "source_target_distance_over_R": safe_ratio(source_target_distance, radius),
        "source_degree": source_degree,
        "target_degree": target_degree,
        "source_transition_count": float(len(source_transitions)),
        "source_p2p_transition_fraction": safe_ratio(source_p2p, len(source_transitions)),
        "source_swap_transition_fraction": safe_ratio(source_swap, len(source_transitions)),
        "source_transition_rate_mean": rate_mean,
        "source_transition_rate_std": rate_std,
        "source_transition_rate_min": rate_min,
        "source_transition_rate_median": rate_median,
        "source_transition_rate_max": rate_max,
        "source_transition_distance_mean_over_R": safe_ratio(tr_mean, radius),
        "source_transition_distance_std_over_R": safe_ratio(tr_std, radius),
        "source_transition_distance_min_over_R": safe_ratio(tr_min, radius),
        "source_transition_distance_median_over_R": safe_ratio(tr_median, radius),
        "source_transition_distance_max_over_R": safe_ratio(tr_max, radius),
        "astar_paths_found": a_paths,
        "astar_K_mean": a_k,
        "astar_log10_K_mean": math.log10(a_k) if a_k > 0.0 else -20.0,
        "astar_K_sum": float(a_star_metrics["K_sum"]),
        "astar_n_mean": float(a_star_metrics["n_mean"]),
        "astar_path_distance_mean_over_R": safe_ratio(
            float(a_star_metrics["path_distance_mean"]), radius
        ),
        "astar_p2p_links_per_path": safe_ratio(
            float(a_star_metrics["p2p_links"]), a_paths
        ),
        "astar_swap_links_per_path": safe_ratio(
            float(a_star_metrics["swap_links"]), a_paths
        ),
        "astar_avg_r_db": float(a_star_metrics["avg_r_db"]),
    }
    for name, value in features.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite graph feature {name}={value}")
    return features

# Evaluate the exact QUR optimization score.
def exact_optimizer_score(n_mean: float, k_mean: float, paths_found: int, a_star_k: float) -> float:
    if int(paths_found) <= 0:
        return -math.inf
    return float(
        opt.score_fsqur(
            float(n_mean),
            float(k_mean),
            float(a_star_k),
            penalty_strength=opt.A_STAR_PENALTY_STRENGTH,
        )
    )

# Transform an exact score into the ML regression target.
def score_to_ml_target(score: float) -> float:
    if score == math.inf:
        return float(math.log1p(1.0e6))
    if not math.isfinite(score) or score <= 0.0:
        return 0.0
    return float(math.log1p(score))

# Select the best exactly evaluated QUR weight sample.
def choose_best_sample(
    sampled_weights: np.ndarray,
    batch_results: np.ndarray,
    a_star_k_mean: float,
) -> tuple[int, float]:
    best = None
    best_index = -1
    best_score = -math.inf
    for sample_index in range(len(sampled_weights)):
        metrics = {
            "paths_found": int(batch_results[sample_index, 2]),
            "n_mean": float(batch_results[sample_index, 0]),
            "K_mean": float(batch_results[sample_index, 1]),
        }
        score = exact_optimizer_score(
            metrics["n_mean"],
            metrics["K_mean"],
            metrics["paths_found"],
            a_star_k_mean,
        )
        if opt.candidate_is_better(score, metrics, sample_index, best):
            best = {
                "score": float(score),
                "metrics": metrics,
                "sample_index": int(sample_index),
            }
            best_index = int(sample_index)
            best_score = float(score)
    if best_index < 0:
        raise RuntimeError("No QUR weight sample was evaluated")
    return best_index, best_score

# Verify that two routing metric records agree.
def assert_same_metrics(first: dict, second: dict) -> None:
    integer_fields = [
        "paths_found",
        "nodes_unique_total",
        "nodes_unique_internal",
        "p2p_links",
        "swap_links",
        "swap_2_links",
        "swap_3_links",
        "swap_4_links",
    ]
    float_fields = [
        "K_sum",
        "K_mean",
        "n_mean",
        "n_mean_including_endpoints",
        "path_distance_sum",
        "path_distance_mean",
        "avg_r_db",
    ]
    for field in integer_fields:
        if int(first[field]) != int(second[field]):
            raise RuntimeError(f"QUR verification failed for {field}")
    for field in float_fields:
        if not math.isclose(
            float(first[field]),
            float(second[field]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"QUR verification failed for {field}")

# Build one standardized output row for an algorithm.
def algorithm_row(
    job: dict,
    context: dict,
    algorithm: str,
    metrics: dict,
    a_star_metrics: dict,
    weights: np.ndarray | None = None,
    optimizer_info: dict | None = None,
    fixed_learning_info: dict | None = None,
) -> dict:
    settings = job["settings"]
    source = context["source"] if context["source"] is not None else 0
    target = context["target"] if context["target"] is not None else 0
    source_target_distance = (
        opt.euclidean_distance(source, target, context["positions"])
        if context["source"] is not None and context["target"] is not None
        else 0.0
    )
    score = (
        exact_optimizer_score(
            metrics["n_mean"],
            metrics["K_mean"],
            metrics["paths_found"],
            a_star_metrics["K_mean"],
        )
        if algorithm.startswith("QUR") and a_star_metrics["K_mean"] > 0.0
        else None
    )
    row = {
        "data_kind": "composable_p2p_live_mdi_relay",
        "scan_name": job["scan_name"],
        "x_value": job["x_value"],
        "x_label": SCAN_LABEL_MAP[job["scan_name"]],
        "seed": job["seed"],
        "algorithm": algorithm,
        "cache_file": os.path.basename(job["cache_path"]),
        "num_nodes": int(settings["num_nodes"]),
        "graph_edges": int(context["graph"].number_of_edges()),
        "radius": float(settings["radius"]),
        "r0_ratio": float(settings["r0_ratio"]),
        "r0_waxman": float(settings["r0_ratio"]) * float(settings["radius"]),
        "beta_waxman": float(settings["beta_waxman"]),
        "prob_comp": float(settings["prob_comp"]),
        "degree_threshold": int(settings["degree_threshold"]),
        "max_paths": int(settings["max_paths"]),
        "remove_used_nodes": int(opt.REMOVE_USED_NODES),
        "count_endpoints_in_n": int(opt.COUNT_ENDPOINTS_IN_N),
        "trusted_nodes": int(len(context["safe_nodes"])),
        "compromised_nodes": int(len(context["compromised_nodes"])),
        "source": int(source),
        "target": int(target),
        "source_target_distance": float(source_target_distance),
        "success": int(metrics["paths_found"] > 0),
        "paths_found": int(metrics["paths_found"]),
        "K_sum": float(metrics["K_sum"]),
        "K_mean": float(metrics["K_mean"]),
        "n_mean": float(metrics["n_mean"]),
        "n_mean_including_endpoints": float(metrics["n_mean_including_endpoints"]),
        "nodes_unique_total": int(metrics["nodes_unique_total"]),
        "nodes_unique_internal": int(metrics["nodes_unique_internal"]),
        "path_distance_sum": float(metrics["path_distance_sum"]),
        "path_distance_mean": float(metrics["path_distance_mean"]),
        "p2p_links": int(metrics["p2p_links"]),
        "swap_links": int(metrics["swap_links"]),
        "swap_2_links": int(metrics["swap_2_links"]),
        "swap_3_links": int(metrics["swap_3_links"]),
        "swap_4_links": int(metrics["swap_4_links"]),
        "avg_r_db": float(metrics["avg_r_db"]),
        "A_star_K_sum": float(a_star_metrics["K_sum"]),
        "A_star_K_mean": float(a_star_metrics["K_mean"]),
        "qur_score": score,
        "optimizer_iterations": None,
        "optimizer_random_seed": None,
        "optimizer_sample_index": None,
        "fixed_weight_scope": None,
        "fixed_predicted_mean_retention": None,
    }
    for name in opt.WEIGHT_NAMES:
        row[f"weight_{name}"] = None
    if weights is not None:
        for name, value in opt.weights_to_dict(weights).items():
            row[f"weight_{name}"] = float(value)
    if optimizer_info is not None:
        row["optimizer_iterations"] = int(optimizer_info["iterations"])
        row["optimizer_random_seed"] = int(optimizer_info["random_seed"])
        row["optimizer_sample_index"] = int(optimizer_info["sample_index"])
    if fixed_learning_info is not None:
        row["fixed_weight_scope"] = str(FIXED_WEIGHT_SCOPE)
        row["fixed_predicted_mean_retention"] = float(
            fixed_learning_info["predicted_mean_retention"]
        )
    return row

# Build the cache path for one exact weight landscape.
def landscape_cache_path(job: dict, fingerprint: str) -> Path:
    key = (
        fingerprint,
        job["scan_name"],
        round(float(job["x_value"]), 10),
        int(job["seed"]),
        tuple(sorted((str(k), repr(v)) for k, v in job["settings"].items())),
    )
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:20]
    return LANDSCAPE_CACHE_DIR / f"landscape_{digest}.npz"

# Save an NPZ cache atomically.
def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)

# Evaluate baselines and optimized QUR for one graph.
def _simulate_base_job_impl(job: dict) -> str:
    output_path = Path(job["landscape_path"])
    if output_path.exists():
        return str(output_path)
    settings = job["settings"]
    context = opt.prepare_network_context(
        job["cache_path"],
        settings["prob_comp"],
        settings["degree_threshold"],
    )
    a_star = opt.evaluate_a_star(context, settings["max_paths"])
    max_min = evaluate_max_min(context, settings["max_paths"])
    features = extract_graph_features(context, a_star, settings)
    if a_star["paths_found"] <= 0 or a_star["K_mean"] <= 0.0:
        sampled_weights = np.empty((0, len(opt.WEIGHT_NAMES)), dtype=np.float64)
        batch_results = np.empty((0, 3), dtype=np.float64)
        exact_scores = np.empty(0, dtype=np.float64)
        ml_targets = np.empty(0, dtype=np.float64)
        best_weights = np.asarray(
            [opt.RATE_WEIGHT_FIXED, 0, 0, 0, 0, 0, 0], dtype=np.float64
        )
        best_index = -1
        best_score = -math.inf
        optimized_metrics = opt.zero_route_metrics()
    else:
        random_seed = optimizer_seed(job)
        sampled_weights = opt.sample_random_weights(WEIGHT_TRIALS, random_seed)
        batch_results = opt.evaluate_fsqur_weight_batch(
            context, sampled_weights, settings["max_paths"]
        )

        exact_scores = np.empty(len(sampled_weights), dtype=np.float64)
        ml_targets = np.empty(len(sampled_weights), dtype=np.float64)
        for i in range(len(sampled_weights)):
            score = exact_optimizer_score(
                batch_results[i, 0],
                batch_results[i, 1],
                int(batch_results[i, 2]),
                a_star["K_mean"],
            )
            exact_scores[i] = score
            ml_targets[i] = score_to_ml_target(score)

        best_index, best_score = choose_best_sample(
            sampled_weights,
            batch_results,
            a_star["K_mean"],
        )
        best_weights = sampled_weights[best_index].copy()
        optimized_metrics = opt.evaluate_fsqur(
            context, best_weights, settings["max_paths"]
        )
        compiled_summary = {
            "paths_found": int(batch_results[best_index, 2]),
            "n_mean": float(batch_results[best_index, 0]),
            "K_mean": float(batch_results[best_index, 1]),
        }
        if int(optimized_metrics["paths_found"]) != compiled_summary["paths_found"]:
            raise RuntimeError("Numba/Python mismatch in paths_found")
        if not math.isclose(
            optimized_metrics["n_mean"], compiled_summary["n_mean"],
            rel_tol=1e-12, abs_tol=1e-15,
        ):
            raise RuntimeError("Numba/Python mismatch in n_mean")
        if not math.isclose(
            optimized_metrics["K_mean"], compiled_summary["K_mean"],
            rel_tol=1e-12, abs_tol=1e-15,
        ):
            raise RuntimeError("Numba/Python mismatch in K_mean")
    optimizer_info = {
        "iterations": int(WEIGHT_TRIALS),
        "random_seed": int(optimizer_seed(job)),
        "sample_index": int(best_index),
        "score": float(best_score),
    }
    rows = [
        algorithm_row(job, context, "A*", a_star, a_star),
        algorithm_row(job, context, "Max-Min", max_min, a_star),
        algorithm_row(
            job,
            context,
            "QUR-Optimized",
            optimized_metrics,
            a_star,
            weights=best_weights,
            optimizer_info=optimizer_info,
        ),
    ]
    atomic_save_npz(
        output_path,
        seed=np.asarray([job["seed"]], dtype=np.int64),
        scan_name=np.asarray([job["scan_name"]], dtype="U40"),
        x_value=np.asarray([job["x_value"]], dtype=np.float64),
        feature_names=np.asarray(list(features.keys()), dtype="U96"),
        feature_values=np.asarray(list(features.values()), dtype=np.float64),
        sampled_weights=np.asarray(sampled_weights, dtype=np.float64),
        batch_results=np.asarray(batch_results, dtype=np.float64),
        exact_scores=np.asarray(exact_scores, dtype=np.float64),
        ml_targets=np.asarray(ml_targets, dtype=np.float64),
        best_weights=np.asarray(best_weights, dtype=np.float64),
        best_index=np.asarray([best_index], dtype=np.int64),
        best_score=np.asarray([best_score], dtype=np.float64),
        a_star_json=np.asarray([json.dumps(a_star)]),
        base_rows_json=np.asarray([json.dumps(rows)]),
    )
    return str(output_path)

# Run one base graph job with memory cleanup.
def simulate_base_job(job: dict) -> str:
    try:
        return _simulate_base_job_impl(job)
    finally:
        cleanup_worker_memory()

# Load one cached optimization landscape.
def load_landscape(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "seed": int(data["seed"][0]),
            "scan_name": str(data["scan_name"][0]),
            "x_value": float(data["x_value"][0]),
            "feature_names": [str(x) for x in data["feature_names"].tolist()],
            "feature_values": np.asarray(data["feature_values"], dtype=np.float64),
            "sampled_weights": np.asarray(data["sampled_weights"], dtype=np.float64),
            "batch_results": np.asarray(data["batch_results"], dtype=np.float64),
            "exact_scores": np.asarray(data["exact_scores"], dtype=np.float64),
            "ml_targets": np.asarray(data["ml_targets"], dtype=np.float64),
            "best_weights": np.asarray(data["best_weights"], dtype=np.float64),
            "best_index": int(data["best_index"][0]),
            "best_score": float(data["best_score"][0]),
            "a_star": json.loads(str(data["a_star_json"][0])),
            "base_rows": json.loads(str(data["base_rows_json"][0])),
        }

# Create any missing topology caches.
def prepare_network_caches(cache_tasks: list[tuple]) -> None:
    missing = [task for task in cache_tasks if not Path(opt.network_cache_path(*task)).exists()]
    print(
        f"Topology caches: {len(cache_tasks):,} required; "
        f"{len(missing):,} missing"
    )
    if not missing:
        return

    if not USE_MULTIPROCESSING:
        for task in tqdm(missing, desc="Topology caches", dynamic_ncols=True):
            create_cache_task(task)
        return

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=MP_CONTEXT,
    ) as executor:
        futures = [executor.submit(create_cache_task, task) for task in missing]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Topology caches",
            dynamic_ncols=True,
        ):
            future.result()

# Run or reuse the per-graph baseline and QUR jobs.
def run_base_jobs(jobs: list[dict], fingerprint: str) -> dict[int, Path]:
    LANDSCAPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path_by_job_index: dict[int, Path] = {}
    pending = []

    for index, job in enumerate(jobs):
        path = landscape_cache_path(job, fingerprint)
        path_by_job_index[index] = path
        job["landscape_path"] = str(path)
        if not path.exists():
            pending.append((index, job))

    print(
        f"Exact graph landscapes: {len(jobs):,} total; "
        f"{len(jobs)-len(pending):,} cached; {len(pending):,} remaining"
    )

    if not pending:
        return path_by_job_index

    if not USE_MULTIPROCESSING:
        worker_initializer()
        for _, job in tqdm(
            pending,
            desc="A* / Max-Min / per-seed QUR",
            dynamic_ncols=True,
        ):
            simulate_base_job(job)
        return path_by_job_index

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=worker_initializer,
        mp_context=MP_CONTEXT,
    ) as executor:
        futures = {
            executor.submit(simulate_base_job, job): index
            for index, job in pending
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="A* / Max-Min / per-seed QUR",
            dynamic_ncols=True,
        ):
            future.result()

    return path_by_job_index

# Create the gradient boosted surrogate model.
def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=MODEL_LEARNING_RATE,
        max_iter=MODEL_MAX_ITER,
        max_leaf_nodes=MODEL_MAX_LEAF_NODES,
        min_samples_leaf=MODEL_MIN_SAMPLES_LEAF,
        l2_regularization=MODEL_L2_REGULARIZATION,
        random_state=MODEL_RANDOM_SEED,
    )

# Build the candidate fixed weight search pool.
def build_candidate_pool(records: list[dict], scope_label: str) -> np.ndarray:
    pieces = []

    for record in records:
        weights = record["sampled_weights"]
        scores = record["exact_scores"]
        if len(weights) == 0:
            continue

        if TOP_CANDIDATES_PER_GRAPH is None:
            pieces.append(weights)
        else:
            k = min(int(TOP_CANDIDATES_PER_GRAPH), len(weights))
            sortable = np.nan_to_num(
                scores,
                nan=-np.inf,
                posinf=np.finfo(np.float64).max,
                neginf=-np.inf,
            )
            top_indices = np.argpartition(sortable, -k)[-k:]
            pieces.append(weights[top_indices])


        pieces.append(record["best_weights"][None, :])

    if EXTRA_FIXED_CANDIDATES > 0:
        extra_seed = seed_from_parts("fixed-extra-candidates", scope_label, experiment_fingerprint())
        pieces.append(opt.sample_random_weights(EXTRA_FIXED_CANDIDATES, extra_seed))

    if not pieces:
        return np.asarray(
            [[opt.RATE_WEIGHT_FIXED, 0, 0, 0, 0, 0, 0]], dtype=np.float64
        )

    pool = np.unique(np.vstack(pieces), axis=0)
    if not np.allclose(pool[:, 0], opt.RATE_WEIGHT_FIXED):
        raise RuntimeError("Fixed candidate pool violates fixed rate weight")
    return pool

# Predict surrogate scores for candidate weights.
def predict_candidate_targets(
    model,
    graph_features: np.ndarray,
    candidate_pool: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(candidate_pool), dtype=np.float64)
    for start in range(0, len(candidate_pool), ML_CANDIDATE_BATCH_SIZE):
        stop = min(start + ML_CANDIDATE_BATCH_SIZE, len(candidate_pool))
        weights = candidate_pool[start:stop, 1:]
        graph_block = np.repeat(graph_features[None, :], len(weights), axis=0)
        X = np.hstack([graph_block, weights])
        output[start:stop] = model.predict(X)
    return output

# Learn one fixed QUR weight vector from graph landscapes.
def learn_fixed_weights_from_paths(
    paths: list[Path],
    scope_label: str,
) -> tuple[np.ndarray, dict]:
    records = [load_landscape(path) for path in paths]
    usable = [r for r in records if len(r["sampled_weights"]) > 0]
    if not usable:
        weights = np.asarray(
            [opt.RATE_WEIGHT_FIXED, 0, 0, 0, 0, 0, 0], dtype=np.float64
        )
        return weights, {
            "predicted_mean_retention": 0.0,
            "training_graphs": 0,
            "training_trials": 0,
            "candidate_pool_size": 1,
        }

    canonical_names = usable[0]["feature_names"]
    X_parts = []
    y_parts = []

    for record in usable:
        if record["feature_names"] != canonical_names:
            raise RuntimeError("ML feature mismatch across graph landscapes")
        weights = record["sampled_weights"]
        repeated = np.repeat(record["feature_values"][None, :], len(weights), axis=0)
        X_parts.append(np.hstack([repeated, weights[:, 1:]]))
        y_parts.append(record["ml_targets"])

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    model = make_model()
    model.fit(X, y)

    candidate_pool = build_candidate_pool(usable, scope_label)
    retention_sum = np.zeros(len(candidate_pool), dtype=np.float64)
    contributing = 0

    for record in tqdm(
        usable,
        desc=f"Fixed-weight search [{scope_label}]",
        leave=False,
        dynamic_ncols=True,
    ):
        pred_log = predict_candidate_targets(
            model,
            record["feature_values"],
            candidate_pool,
        )
        pred_score = np.expm1(np.maximum(pred_log, 0.0))
        best = float(np.max(pred_score))
        if best <= 0.0 or not math.isfinite(best):
            continue
        retention_sum += pred_score / best
        contributing += 1

    if contributing == 0:

        mean_prediction = np.zeros(len(candidate_pool), dtype=np.float64)
        for record in usable:
            mean_prediction += predict_candidate_targets(
                model, record["feature_values"], candidate_pool
            )
        chosen_index = int(np.argmax(mean_prediction))
        predicted_mean_retention = 0.0
    else:
        mean_retention = retention_sum / contributing
        chosen_index = int(np.argmax(mean_retention))
        predicted_mean_retention = float(mean_retention[chosen_index])

    chosen = candidate_pool[chosen_index].copy()

    info = {
        "predicted_mean_retention": float(predicted_mean_retention),
        "training_graphs": int(len(usable)),
        "training_trials": int(len(y)),
        "candidate_pool_size": int(len(candidate_pool)),
    }

    del X, y, model, candidate_pool, X_parts, y_parts
    gc.collect()
    _malloc_trim()
    return chosen, info

# Learn fixed weights for every configured scan group.
def learn_all_fixed_weights(
    jobs: list[dict],
    path_by_job_index: dict[int, Path],
) -> tuple[dict[tuple[str, float], np.ndarray], dict[tuple[str, float], dict], pd.DataFrame]:
    grouped_indices: dict[tuple[str, float], list[int]] = {}
    for index, job in enumerate(jobs):
        grouped_indices.setdefault(group_key(job), []).append(index)

    weights_by_group: dict[tuple[str, float], np.ndarray] = {}
    info_by_group: dict[tuple[str, float], dict] = {}
    rows = []

    if FIXED_WEIGHT_SCOPE == "global":
        all_paths = [path_by_job_index[i] for i in range(len(jobs))]
        weights, info = learn_fixed_weights_from_paths(all_paths, "GLOBAL")
        for key in grouped_indices:
            weights_by_group[key] = weights.copy()
            info_by_group[key] = dict(info)
        rows.append(
            {
                "scan_name": "GLOBAL",
                "x_value": math.nan,
                "scope": "global",
                **info,
                **{f"weight_{k}": v for k, v in opt.weights_to_dict(weights).items()},
            }
        )

    elif FIXED_WEIGHT_SCOPE == "per_scan_point":
        for key in tqdm(
            sorted(grouped_indices),
            desc="Learn fixed weights by configuration",
            dynamic_ncols=True,
        ):
            paths = [path_by_job_index[i] for i in grouped_indices[key]]
            weights, info = learn_fixed_weights_from_paths(paths, group_label(key))
            weights_by_group[key] = weights
            info_by_group[key] = info
            rows.append(
                {
                    "scan_name": key[0],
                    "x_value": key[1],
                    "scope": "per_scan_point",
                    **info,
                    **{
                        f"weight_{name}": value
                        for name, value in opt.weights_to_dict(weights).items()
                    },
                }
            )
    else:
        raise ValueError(
            "FIXED_WEIGHT_SCOPE must be 'per_scan_point' or 'global'"
        )

    fixed_df = pd.DataFrame(rows)
    fixed_df.to_csv(FIXED_WEIGHTS_CSV, index=False)
    return weights_by_group, info_by_group, fixed_df

# Build the cache path for a fixed-weight evaluation.
def fixed_eval_cache_path(job: dict, weights: np.ndarray, fingerprint: str) -> Path:
    weight_key = tuple(round(float(v), opt.WEIGHT_DECIMALS) for v in weights)
    key = (
        fingerprint,
        job["scan_name"],
        round(float(job["x_value"]), 10),
        int(job["seed"]),
        weight_key,
    )
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:20]
    return FIXED_EVAL_CACHE_DIR / f"fixed_{digest}.json"

# Write a JSON cache atomically.
def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temp, path)

# Evaluate learned fixed weights on one graph.
def _evaluate_fixed_job_impl(job: dict) -> str:
    output_path = Path(job["fixed_eval_path"])
    if output_path.exists():
        return str(output_path)

    weights = np.asarray(job["fixed_weights"], dtype=np.float64)
    learning_info = dict(job["fixed_learning_info"])
    base_record = load_landscape(Path(job["landscape_path"]))
    a_star = base_record["a_star"]

    settings = job["settings"]
    context = opt.prepare_network_context(
        job["cache_path"],
        settings["prob_comp"],
        settings["degree_threshold"],
    )
    metrics = opt.evaluate_fsqur(context, weights, settings["max_paths"])
    row = algorithm_row(
        job,
        context,
        "QUR-Fixed-Learned",
        metrics,
        a_star,
        weights=weights,
        fixed_learning_info=learning_info,
    )
    atomic_write_json(output_path, {"row": row})
    return str(output_path)

# Run one fixed-weight graph job with cleanup.
def evaluate_fixed_job(job: dict) -> str:
    try:
        return _evaluate_fixed_job_impl(job)
    finally:
        cleanup_worker_memory()

# Evaluate learned fixed weights across all graph jobs.
def run_fixed_evaluations(
    jobs: list[dict],
    path_by_job_index: dict[int, Path],
    weights_by_group: dict[tuple[str, float], np.ndarray],
    info_by_group: dict[tuple[str, float], dict],
    fingerprint: str,
) -> dict[int, Path]:
    FIXED_EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_paths: dict[int, Path] = {}
    pending = []

    for index, job in enumerate(jobs):
        key = group_key(job)
        weights = weights_by_group[key]
        path = fixed_eval_cache_path(job, weights, fingerprint)
        result_paths[index] = path
        if not path.exists():
            fixed_job = dict(job)
            fixed_job["landscape_path"] = str(path_by_job_index[index])
            fixed_job["fixed_eval_path"] = str(path)
            fixed_job["fixed_weights"] = weights.tolist()
            fixed_job["fixed_learning_info"] = info_by_group[key]
            pending.append((index, fixed_job))

    print(
        f"Fixed-QUR evaluations: {len(jobs):,} total; "
        f"{len(jobs)-len(pending):,} cached; {len(pending):,} remaining"
    )

    if not pending:
        return result_paths

    if not USE_MULTIPROCESSING:
        worker_initializer()
        for _, job in tqdm(
            pending,
            desc="QUR fixed learned weights",
            dynamic_ncols=True,
        ):
            evaluate_fixed_job(job)
        return result_paths

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=worker_initializer,
        mp_context=MP_CONTEXT,
    ) as executor:
        futures = {
            executor.submit(evaluate_fixed_job, job): index
            for index, job in pending
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="QUR fixed learned weights",
            dynamic_ncols=True,
        ):
            future.result()

    return result_paths

# Assemble and save the complete result table.
def build_final_results(
    jobs: list[dict],
    path_by_job_index: dict[int, Path],
    fixed_path_by_job_index: dict[int, Path],
) -> pd.DataFrame:
    rows = []
    for index in tqdm(
        range(len(jobs)),
        desc="Assembling final CSV",
        dynamic_ncols=True,
    ):
        base = load_landscape(path_by_job_index[index])
        rows.extend(base["base_rows"])
        fixed_payload = json.loads(
            fixed_path_by_job_index[index].read_text(encoding="utf-8")
        )
        rows.append(fixed_payload["row"])

    df = pd.DataFrame(rows)
    algorithm_rank = {name: i for i, name in enumerate(ALGORITHM_ORDER)}
    df["_algorithm_rank"] = df["algorithm"].map(algorithm_rank)
    df = df.sort_values(
        ["scan_name", "x_value", "seed", "_algorithm_rank"],
        kind="stable",
    ).drop(columns="_algorithm_rank")
    df.to_csv(RESULTS_CSV, index=False)
    return df

# Divide arrays while protecting invalid denominators.
def safe_divide_array(numerator, denominator) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)

    output = np.zeros_like(num, dtype=float)

    valid = (
        np.isfinite(num)
        & np.isfinite(den)
        & (den > 0.0)
    )

    output[valid] = num[valid] / den[valid]

    return output

# Derive graph-level metrics used by manuscript figures.
def add_manuscript_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    numeric_columns = [
        "x_value",
        "seed",
        "num_nodes",
        "radius",
        "degree_threshold",
        "max_paths",
        "compromised_nodes",
        "success",
        "paths_found",
        "K_sum",
        "K_mean",
        "n_mean",
        "nodes_unique_internal",
    ]

    for column in numeric_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(
                work[column],
                errors="coerce",
            )

    if "count_endpoints_in_n" in work.columns:

        endpoint_flag = pd.to_numeric(
            work["count_endpoints_in_n"],
            errors="coerce",
        ).fillna(0.0)

        internal_mean = np.where(
            endpoint_flag.to_numpy(dtype=float) > 0.0,
            np.maximum(
                work["n_mean"].to_numpy(dtype=float) - 2.0,
                0.0,
            ),
            work["n_mean"].to_numpy(dtype=float),
        )

    else:

        internal_mean = work["n_mean"].to_numpy(
            dtype=float
        )

    work["internal_mean"] = internal_mean


    work["internal_occurrences"] = (
        work["paths_found"].to_numpy(dtype=float)
        * internal_mean
    )





    raw_reuse = (
        work["internal_occurrences"].to_numpy(dtype=float)
        - work["nodes_unique_internal"].to_numpy(dtype=float)
    )

    work["reuse_excess"] = np.where(
        raw_reuse > REUSE_TOLERANCE,
        raw_reuse,
        0.0,
    )

    work["reuse_fraction_pct_job"] = np.where(
        work["internal_occurrences"].to_numpy(dtype=float) > 0.0,
        (
            100.0
            * work["reuse_excess"].to_numpy(dtype=float)
            / work["internal_occurrences"].to_numpy(dtype=float)
        ),
        0.0,
    )





    work["eta_internal"] = safe_divide_array(
        work["K_mean"],
        work["internal_mean"],
    )

    work["eta_unique"] = safe_divide_array(
        work["K_sum"],
        work["nodes_unique_internal"],
    )

    work["eta_average"] = 0.5 * (
        work["eta_internal"]
        + work["eta_unique"]
    )





    work["compromised_percent"] = (
        100.0
        * work["compromised_nodes"].to_numpy(dtype=float)
        / work["num_nodes"].to_numpy(dtype=float)
    )

    return work

# Compute the mean and standard error of a series.
def mean_and_sem(
    values: pd.Series,
) -> tuple[float, float]:

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return math.nan, math.nan

    return (
        float(numeric.mean()),
        float(numeric.sem()),
    )

# Aggregate graph results for one parameter scan.
def aggregate_manuscript_scan(
    df: pd.DataFrame,
    scan_name: str,
) -> pd.DataFrame:

    scan = df[
        df["scan_name"] == scan_name
    ].copy()

    if scan.empty:
        raise ValueError(
            f"No {scan_name!r} scan was found."
        )

    rows = []

    for (
        x_value,
        algorithm,
    ), group in scan.groupby(
        [
            "x_value",
            "algorithm",
        ],
        observed=True,
    ):

        k_mean, k_sem = mean_and_sem(
            group["K_mean"]
        )

        n_mean, n_sem = mean_and_sem(
            group["internal_mean"]
        )

        eta_mean, eta_sem = mean_and_sem(
            group["eta_average"]
        )



        total_occurrences = float(
            group["internal_occurrences"].sum()
        )

        total_reuse = float(
            group["reuse_excess"].sum()
        )

        reuse_fraction = (
            100.0
            * total_reuse
            / total_occurrences
            if total_occurrences > 0.0
            else 0.0
        )

        rows.append(
            {
                "x_value": float(x_value),

                "algorithm": str(
                    algorithm
                ),

                "compromised_percent": float(
                    group[
                        "compromised_percent"
                    ].mean()
                ),

                "success_percent": float(
                    100.0
                    * group[
                        "success"
                    ].mean()
                ),

                "K_mean": k_mean,
                "K_sem": k_sem,

                "K_sum": float(
                    group[
                        "K_sum"
                    ].mean()
                ),

                "paths_found": float(
                    group[
                        "paths_found"
                    ].mean()
                ),

                "n_mean": n_mean,
                "n_sem": n_sem,

                "nodes_unique_internal": float(
                    group[
                        "nodes_unique_internal"
                    ].mean()
                ),

                "eta_internal": float(
                    group[
                        "eta_internal"
                    ].mean()
                ),

                "eta_unique": float(
                    group[
                        "eta_unique"
                    ].mean()
                ),

                "eta_average": eta_mean,
                "eta_average_sem": eta_sem,

                "reuse_fraction_pct":
                    reuse_fraction,

                "jobs": int(
                    len(group)
                ),
            }
        )

    return pd.DataFrame(rows)

# Select matched default network rows across algorithms.
def select_default_jobs(
    df: pd.DataFrame,
) -> pd.DataFrame:

    work = df[
        (
            df["scan_name"]
            == OVERALL_SOURCE_SCAN
        )
        & (
            df["num_nodes"]
            == int(NUM_NODES)
        )
        & np.isclose(
            df["radius"],
            float(RADIUS),
        )
        & (
            df["degree_threshold"]
            == int(DEGREE_THRESHOLD)
        )
        & (
            df["max_paths"]
            == int(MAX_PATHS)
        )
    ].copy()

    if work.empty:

        raise ValueError(
            "No rows match the default "
            "network configuration in scan "
            f"{OVERALL_SOURCE_SCAN!r}."
        )



    work = (
        work.sort_values(
            [
                "seed",
                "algorithm",
            ],
            kind="stable",
        )
        .drop_duplicates(
            [
                "seed",
                "algorithm",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


    seed_sets = {
        algorithm: set(
            work.loc[
                work["algorithm"]
                == algorithm,
                "seed",
            ].astype(int)
        )
        for algorithm
        in ALGORITHM_ORDER
    }

    common_seeds = set.intersection(
        *(
            seed_sets[
                algorithm
            ]
            for algorithm
            in ALGORITHM_ORDER
        )
    )

    if not common_seeds:

        raise ValueError(
            "The default rows contain no "
            "seeds shared by all four "
            "algorithms."
        )

    work = work[
        work["seed"]
        .astype(int)
        .isin(common_seeds)
    ].copy()

    print(
        "\nOverall manuscript comparison:"
    )

    print(
        f"  source scan : "
        f"{OVERALL_SOURCE_SCAN}"
    )

    print(
        f"  matched seeds: "
        f"{len(common_seeds)}"
    )

    return work

# Calculate percentage change relative to a reference.
def percentage_change(
    candidate: float,
    reference: float,
) -> float:

    if (
        not np.isfinite(reference)
        or reference == 0.0
    ):
        return math.nan

    return (
        100.0
        * (
            candidate
            / reference
            - 1.0
        )
    )

# Calculate percentage saving relative to a reference.
def percentage_saving(
    candidate: float,
    reference: float,
) -> float:

    if (
        not np.isfinite(reference)
        or reference == 0.0
    ):
        return math.nan

    return (
        100.0
        * (
            1.0
            - candidate
            / reference
        )
    )

# Build the overall QUR-versus-baseline comparison.
def build_overall_fixed_comparison(
    default_df: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    pd.DataFrame,
]:

    columns = [
        "K_mean",
        "internal_mean",
        "nodes_unique_internal",
        "paths_found",
        "success",
        "eta_internal",
        "eta_unique",
        "eta_average",
    ]

    means = (
        default_df.groupby(
            "algorithm",
            observed=True,
        )[columns]
        .mean()
    )

    candidates = [
        "QUR-Optimized",
        "QUR-Fixed-Learned",
    ]

    references = [
        "A*",
        "Max-Min",
    ]
    output = {}
    table_rows = []

    for candidate in candidates:

        candidate_values = means.loc[
            candidate
        ]

        output[candidate] = {}

        for reference in references:

            ref = means.loc[
                reference
            ]

            internal_node_saving = percentage_saving(
                float(
                    candidate_values[
                        "internal_mean"
                    ]
                ),
                float(
                    ref[
                        "internal_mean"
                    ]
                ),
            )

            unique_node_saving = percentage_saving(
                float(
                    candidate_values[
                        "nodes_unique_internal"
                    ]
                ),
                float(
                    ref[
                        "nodes_unique_internal"
                    ]
                ),
            )

            average_node_saving = 0.5 * (
                internal_node_saving
                + unique_node_saving
            )





            internal_efficiency_change = percentage_change(
                float(
                    candidate_values[
                        "eta_internal"
                    ]
                ),
                float(
                    ref[
                        "eta_internal"
                    ]
                ),
            )

            unique_efficiency_change = percentage_change(
                float(
                    candidate_values[
                        "eta_unique"
                    ]
                ),
                float(
                    ref[
                        "eta_unique"
                    ]
                ),
            )

            average_efficiency_change = 0.5 * (
                internal_efficiency_change
                + unique_efficiency_change
            )





            values = {

                "mean_path_skr_change":
                    percentage_change(
                        float(
                            candidate_values[
                                "K_mean"
                            ]
                        ),
                        float(
                            ref[
                                "K_mean"
                            ]
                        ),
                    ),

                "internal_node_saving":
                    internal_node_saving,

                "unique_node_saving":
                    unique_node_saving,

                "average_node_saving":
                    average_node_saving,

                "paths_found_change":
                    percentage_change(
                        float(
                            candidate_values[
                                "paths_found"
                            ]
                        ),
                        float(
                            ref[
                                "paths_found"
                            ]
                        ),
                    ),

                "internal_efficiency_change":
                    internal_efficiency_change,

                "unique_efficiency_change":
                    unique_efficiency_change,

                "average_efficiency_change":
                    average_efficiency_change,
            }

            output[
                candidate
            ][
                reference
            ] = values

            table_rows.append(
                {
                    "candidate":
                        candidate,

                    "reference":
                        reference,

                    **values,
                }
            )

    return (
        output,
        pd.DataFrame(
            table_rows
        ),
    )

# Export manuscript statistics as LaTeX commands.
def export_latex_values(
    default_df: pd.DataFrame,
    comparison:
        dict[
            str,
            dict[
                str,
                dict[str, float],
            ],
        ],
    degree: pd.DataFrame,
) -> None:

    def latex_one_decimal(
        name: str,
        value: float,
    ) -> str:

        return (
            f"\\newcommand{{\\{name}}}"
            f"{{{float(value):.1f}}}"
        )

    def latex_two_decimals(
        name: str,
        value: float,
    ) -> str:

        return (
            f"\\newcommand{{\\{name}}}"
            f"{{{float(value):.2f}}}"
        )

    def get_degree_row(
        threshold: float,
        algorithm: str,
    ) -> pd.Series:

        rows = degree[
            np.isclose(
                degree["x_value"],
                float(threshold),
            )
            & (
                degree["algorithm"]
                == algorithm
            )
        ]

        if len(rows) != 1:

            raise ValueError(
                "Expected exactly one aggregated row for "
                f"degree threshold={threshold:g}, "
                f"algorithm={algorithm!r}; "
                f"found {len(rows)}."
            )

        return rows.iloc[0]

    def average_node_saving(
        candidate: pd.Series,
        reference: pd.Series,
    ) -> float:

        internal_saving = percentage_saving(
            float(candidate["n_mean"]),
            float(reference["n_mean"]),
        )

        unique_saving = percentage_saving(
            float(
                candidate[
                    "nodes_unique_internal"
                ]
            ),
            float(
                reference[
                    "nodes_unique_internal"
                ]
            ),
        )

        return 0.5 * (
            internal_saving
            + unique_saving
        )

    def same_paths_percent(
            candidate_algorithm: str,
            reference_algorithm: str,
    ) -> float:

        candidate = (
            default_df[
                default_df["algorithm"]
                == candidate_algorithm
                ][
                [
                    "seed",
                    "paths_found",
                ]
            ]
            .rename(
                columns={
                    "paths_found":
                        "candidate_paths",
                }
            )
        )

        reference = (
            default_df[
                default_df["algorithm"]
                == reference_algorithm
                ][
                [
                    "seed",
                    "paths_found",
                ]
            ]
            .rename(
                columns={
                    "paths_found":
                        "reference_paths",
                }
            )
        )

        matched = candidate.merge(
            reference,
            on="seed",
            how="inner",
        )

        if matched.empty:
            return math.nan

        same = np.isclose(
            matched[
                "candidate_paths"
            ].to_numpy(dtype=float),

            matched[
                "reference_paths"
            ].to_numpy(dtype=float),
        )

        return (
                100.0
                * float(
            np.mean(same)
        )
        )

        reference = (
            default_df[
                default_df["algorithm"]
                == reference_algorithm
            ][
                [
                    "seed",
                    "paths_found",
                ]
            ]
            .rename(
                columns={
                    "paths_found":
                        "reference_paths",
                }
            )
        )

        matched = qur.merge(
            reference,
            on="seed",
            how="inner",
        )

        if matched.empty:

            return math.nan

        same = np.isclose(
            matched[
                "qur_paths"
            ].to_numpy(dtype=float),

            matched[
                "reference_paths"
            ].to_numpy(dtype=float),
        )

        return (
            100.0
            * float(
                np.mean(same)
            )
        )

    qur_astar = comparison[
        "QUR-Optimized"
    ][
        "A*"
    ]

    qur_max_min = comparison[
        "QUR-Optimized"
    ][
        "Max-Min"
    ]
    fixed_astar = comparison[
        "QUR-Fixed-Learned"
    ][
        "A*"
    ]

    fixed_max_min = comparison[
        "QUR-Fixed-Learned"
    ][
        "Max-Min"
    ]
    default_graph_count = int(
        default_df[
            "seed"
        ].nunique()
    )

    same_paths_astar = same_paths_percent(
        "QUR-Optimized",
        "A*",
    )

    same_paths_max_min = same_paths_percent(
        "QUR-Optimized",
        "Max-Min",
    )

    fixed_same_paths_astar = same_paths_percent(
        "QUR-Fixed-Learned",
        "A*",
    )

    fixed_same_paths_max_min = same_paths_percent(
        "QUR-Fixed-Learned",
        "Max-Min",
    )
    qur_d4 = get_degree_row(
        4,
        "QUR-Optimized",
    )
    fixed_d4 = get_degree_row(
        4,
        "QUR-Fixed-Learned",
    )
    astar_d4 = get_degree_row(
        4,
        "A*",
    )

    max_min_d4 = get_degree_row(
        4,
        "Max-Min",
    )

    d4_skr_change_astar = (
        percentage_change(
            float(
                qur_d4[
                    "K_mean"
                ]
            ),
            float(
                astar_d4[
                    "K_mean"
                ]
            ),
        )
    )

    d4_node_saving_astar = (
        average_node_saving(
            qur_d4,
            astar_d4,
        )
    )

    d4_node_saving_max_min = (
        average_node_saving(
            qur_d4,
            max_min_d4,
        )
    )
    fixed_d4_skr_change_astar = (
        percentage_change(
            float(
                fixed_d4[
                    "K_mean"
                ]
            ),
            float(
                astar_d4[
                    "K_mean"
                ]
            ),
        )
    )
    fixed_d4_node_saving_astar = (
        average_node_saving(
            fixed_d4,
            astar_d4,
        )
    )

    fixed_d4_node_saving_max_min = (
        average_node_saving(
            fixed_d4,
            max_min_d4,
        )
    )

    qur_d6 = get_degree_row(
        6,
        "QUR-Optimized",
    )
    fixed_d6 = get_degree_row(
        6,
        "QUR-Fixed-Learned",
    )
    astar_d6 = get_degree_row(
        6,
        "A*",
    )

    max_min_d6 = get_degree_row(
        6,
        "Max-Min",
    )

    d6_skr_change_astar = (
        percentage_change(
            float(
                qur_d6[
                    "K_mean"
                ]
            ),
            float(
                astar_d6[
                    "K_mean"
                ]
            ),
        )
    )

    d6_node_saving_astar = (
        average_node_saving(
            qur_d6,
            astar_d6,
        )
    )

    d6_node_saving_max_min = (
        average_node_saving(
            qur_d6,
            max_min_d6,
        )
    )
    fixed_d6_skr_change_astar = (
        percentage_change(
            float(
                fixed_d6[
                    "K_mean"
                ]
            ),
            float(
                astar_d6[
                    "K_mean"
                ]
            ),
        )
    )

    fixed_d6_node_saving_astar = (
        average_node_saving(
            fixed_d6,
            astar_d6,
        )
    )

    fixed_d6_node_saving_max_min = (
        average_node_saving(
            fixed_d6,
            max_min_d6,
        )
    )


    lines = [

        (
            "% Automatically generated by "
            "network_scan_four_algorithms_ml_fixed.py"
        ),

        (
            "% Do not edit these values by hand."
        ),

        (
            f"\\newcommand{{\\DefaultGraphCount}}"
            f"{{{default_graph_count}}}"
        ),

        latex_one_decimal(
            "QurSkrIncreaseAstar",
            qur_astar[
                "mean_path_skr_change"
            ],
        ),



        latex_one_decimal(
            "QurSkrShortfallMaxMin",
            -qur_max_min[
                "mean_path_skr_change"
            ],
        ),




        latex_one_decimal(
            "QurFixedSkrIncreaseAstar",
            fixed_astar[
                "mean_path_skr_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedSkrShortfallMaxMin",
            -fixed_max_min[
                "mean_path_skr_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedAverageNodeSavingAstar",
            fixed_astar[
                "average_node_saving"
            ],
        ),

        latex_one_decimal(
            "QurFixedAverageNodeSavingMaxMin",
            fixed_max_min[
                "average_node_saving"
            ],
        ),

        latex_one_decimal(
            "QurFixedPathsChangeAstar",
            fixed_astar[
                "paths_found_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedPathsIncreaseMaxMin",
            fixed_max_min[
                "paths_found_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedAverageEfficiencyIncreaseAstar",
            fixed_astar[
                "average_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedAverageEfficiencyIncreaseMaxMin",
            fixed_max_min[
                "average_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedInternalEfficiencyIncreaseAstar",
            fixed_astar[
                "internal_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedUniqueEfficiencyIncreaseAstar",
            fixed_astar[
                "unique_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedInternalEfficiencyIncreaseMaxMin",
            fixed_max_min[
                "internal_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedUniqueEfficiencyIncreaseMaxMin",
            fixed_max_min[
                "unique_efficiency_change"
            ],
        ),
        latex_one_decimal(
            "QurAverageNodeSavingAstar",
            qur_astar[
                "average_node_saving"
            ],
        ),

        latex_one_decimal(
            "QurAverageNodeSavingMaxMin",
            qur_max_min[
                "average_node_saving"
            ],
        ),

        latex_one_decimal(
            "QurPathsChangeAstar",
            qur_astar[
                "paths_found_change"
            ],
        ),

        latex_one_decimal(
            "QurPathsIncreaseMaxMin",
            qur_max_min[
                "paths_found_change"
            ],
        ),

        latex_one_decimal(
            "QurAverageEfficiencyIncreaseAstar",
            qur_astar[
                "average_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurAverageEfficiencyIncreaseMaxMin",
            qur_max_min[
                "average_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurInternalEfficiencyIncreaseAstar",
            qur_astar[
                "internal_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurUniqueEfficiencyIncreaseAstar",
            qur_astar[
                "unique_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurInternalEfficiencyIncreaseMaxMin",
            qur_max_min[
                "internal_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurUniqueEfficiencyIncreaseMaxMin",
            qur_max_min[
                "unique_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurSamePathsAstar",
            same_paths_astar,
        ),

        latex_one_decimal(
            "QurSamePathsMaxMin",
            same_paths_max_min,
        ),




        latex_one_decimal(
            "QurFixedSkrIncreaseAstar",
            fixed_astar[
                "mean_path_skr_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedSkrShortfallMaxMin",
            -fixed_max_min[
                "mean_path_skr_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedAverageNodeSavingAstar",
            fixed_astar[
                "average_node_saving"
            ],
        ),

        latex_one_decimal(
            "QurFixedAverageNodeSavingMaxMin",
            fixed_max_min[
                "average_node_saving"
            ],
        ),

        latex_one_decimal(
            "QurFixedPathsChangeAstar",
            fixed_astar[
                "paths_found_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedPathsIncreaseMaxMin",
            fixed_max_min[
                "paths_found_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedAverageEfficiencyIncreaseAstar",
            fixed_astar[
                "average_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedAverageEfficiencyIncreaseMaxMin",
            fixed_max_min[
                "average_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedInternalEfficiencyIncreaseAstar",
            fixed_astar[
                "internal_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedUniqueEfficiencyIncreaseAstar",
            fixed_astar[
                "unique_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedInternalEfficiencyIncreaseMaxMin",
            fixed_max_min[
                "internal_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedUniqueEfficiencyIncreaseMaxMin",
            fixed_max_min[
                "unique_efficiency_change"
            ],
        ),

        latex_one_decimal(
            "QurFixedSamePathsAstar",
            fixed_same_paths_astar,
        ),

        latex_one_decimal(
            "QurFixedSamePathsMaxMin",
            fixed_same_paths_max_min,
        ),




        latex_one_decimal(
            "DFourCompromised",
            qur_d4[
                "compromised_percent"
            ],
        ),

        latex_one_decimal(
            "DFourQurSuccess",
            qur_d4[
                "success_percent"
            ],
        ),

        latex_two_decimals(
            "DFourQurPaths",
            qur_d4[
                "paths_found"
            ],
        ),

        latex_one_decimal(
            "DFourQurSkrChangeAstar",
            d4_skr_change_astar,
        ),

        latex_one_decimal(
            "DFourQurAverageNodeSavingAstar",
            d4_node_saving_astar,
        ),

        latex_one_decimal(
            "DFourQurAverageNodeSavingMaxMin",
            d4_node_saving_max_min,
        ),
        latex_one_decimal(
            "DFourFixedQurSuccess",
            fixed_d4[
                "success_percent"
            ],
        ),

        latex_two_decimals(
            "DFourFixedQurPaths",
            fixed_d4[
                "paths_found"
            ],
        ),

        latex_one_decimal(
            "DFourFixedQurSkrChangeAstar",
            fixed_d4_skr_change_astar,
        ),

        latex_one_decimal(
            "DFourFixedQurAverageNodeSavingAstar",
            fixed_d4_node_saving_astar,
        ),

        latex_one_decimal(
            "DFourFixedQurAverageNodeSavingMaxMin",
            fixed_d4_node_saving_max_min,
        ),




        latex_one_decimal(
            "DSixCompromised",
            qur_d6[
                "compromised_percent"
            ],
        ),

        latex_one_decimal(
            "DSixQurSuccess",
            qur_d6[
                "success_percent"
            ],
        ),

        latex_two_decimals(
            "DSixQurPaths",
            qur_d6[
                "paths_found"
            ],
        ),

        latex_one_decimal(
            "DSixQurSkrChangeAstar",
            d6_skr_change_astar,
        ),

        latex_one_decimal(
            "DSixQurAverageNodeSavingAstar",
            d6_node_saving_astar,
        ),

        latex_one_decimal(
            "DSixQurAverageNodeSavingMaxMin",
            d6_node_saving_max_min,
        ),

        latex_one_decimal(
            "DSixFixedQurSuccess",
            fixed_d6[
                "success_percent"
            ],
        ),

        latex_two_decimals(
            "DSixFixedQurPaths",
            fixed_d6[
                "paths_found"
            ],
        ),

        latex_one_decimal(
            "DSixFixedQurSkrChangeAstar",
            fixed_d6_skr_change_astar,
        ),

        latex_one_decimal(
            "DSixFixedQurAverageNodeSavingAstar",
            fixed_d6_node_saving_astar,
        ),

        latex_one_decimal(
            "DSixFixedQurAverageNodeSavingMaxMin",
            fixed_d6_node_saving_max_min,
        ),
    ]

    LATEX_VALUES_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"Saved LaTeX values: "
        f"{LATEX_VALUES_FILE}"
    )

# Create legend handles for the routing algorithms.
def manuscript_algorithm_handles(
) -> list[Line2D]:

    return [
        Line2D(
            [0],
            [0],

            color=COLORS[
                algorithm
            ],

            marker=MARKERS[
                algorithm
            ],

            linewidth=1.6,
            markersize=6.5,

            label=DISPLAY[
                algorithm
            ],
        )

        for algorithm
        in ALGORITHM_ORDER
    ]

# Create legend handles for plotted metric styles.
def manuscript_style_handles(
) -> list[Line2D]:

    return [

        Line2D(
            [0],
            [0],

            color="black",

            linewidth=1.6,
            linestyle="-",

            label="Main metric",
        ),

        Line2D(
            [0],
            [0],

            color="black",

            linewidth=1.6,
            linestyle="--",

            label=(
                "Repeated nodes (%)"
            ),
        ),
    ]

# Save and close a manuscript figure.
def save_manuscript_png(
    fig: plt.Figure,
    filename: str,
) -> None:

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIR
        / filename
    )
    fig.savefig(
        output_path,

        dpi=PLOT_DPI,

        facecolor="white",

        bbox_inches="tight",

        pad_inches=PLOT_PAD_INCHES,
    )

    plt.close(
        fig
    )

    print(
        f"Saved figure: "
        f"{output_path}"
    )

# Add a percentage annotation to a plot.
def add_percentage_label(
    ax: plt.Axes,
    value: float,
    y: float,
    color,
) -> None:

    if not np.isfinite(
        value
    ):
        return

    span = (
        ax.get_xlim()[1]
        - ax.get_xlim()[0]
    )

    offset = max(
        0.8,
        0.012 * span,
    )

    if value >= 0.0:

        x = value + offset
        ha = "left"

    else:

        x = value - offset
        ha = "right"

    ax.text(
        x,
        y,

        f"{value:+.1f}%",

        va="center",
        ha=ha,

        fontsize=14,

        color=color,
    )

# Plot the overall QUR comparison against both baselines.
def plot_overall_qur_comparison(
    comparison:
        dict[
            str,
            dict[
                str,
                dict[str, float],
            ],
        ],
) -> None:

    metric_labels = [
        "Mean path SKR",
        "Average node saving",
        "Paths found",
        "Average SKR per node",
    ]

    keys = [
        "mean_path_skr_change",
        "average_node_saving",
        "paths_found_change",
        "average_efficiency_change",
    ]

    fig, ax = plt.subplots(
        figsize=(
            15.0,
            7.5,
        ),
        layout="constrained",
    )

    y_positions = np.arange(
        len(metric_labels),
        dtype=float,
    )

    offsets = {

        (
            "QUR-Optimized",
            "A*",
        ):
            -0.30,

        (
            "QUR-Fixed-Learned",
            "A*",
        ):
            -0.10,

        (
            "QUR-Optimized",
            "Max-Min",
        ):
            0.10,

        (
            "QUR-Fixed-Learned",
            "Max-Min",
        ):
            0.30,
    }

    bar_height = 0.16

    bar_specs = [
        (
            "QUR-Optimized",
            "A*",
            BAR_REFERENCE_COLORS[
                "A*"
            ],
            BAR_OPTIMIZED_ALPHA,
            "QUR optimized per graph vs A*",
        ),

        (
            "QUR-Fixed-Learned",
            "A*",
            BAR_REFERENCE_COLORS[
                "A*"
            ],
            BAR_FIXED_ALPHA,
            "QUR fixed learned weights vs A*",
        ),
        (
            "QUR-Optimized",
            "Max-Min",
            BAR_REFERENCE_COLORS[
                "Max-Min"
            ],
            BAR_OPTIMIZED_ALPHA,
            "QUR optimized per graph vs Max-Min",
        ),

        (
            "QUR-Fixed-Learned",
            "Max-Min",
            BAR_REFERENCE_COLORS[
                "Max-Min"
            ],
            BAR_FIXED_ALPHA,
            "QUR fixed learned weights vs Max-Min",
        ),
    ]

    all_values = []
    for (
        candidate,
        reference,
        color,
        alpha,
        legend_label,
    ) in bar_specs:

        values = [
            float(
                comparison[
                    candidate
                ][
                    reference
                ][
                    key
                ]
            )
            for key in keys
        ]

        all_values.extend(
            values
        )

        ax.barh(

            y_positions
            + offsets[
                (
                    candidate,
                    reference,
                )
            ],

            values,

            height=bar_height,

            color=color,

            alpha=alpha,

            label=legend_label,
        )

    finite = np.asarray(
        all_values,
        dtype=float,
    )

    finite = finite[
        np.isfinite(
            finite
        )
    ]

    minimum = (
        min(
            0.0,
            float(
                finite.min()
            ),
        )
        if finite.size
        else -1.0
    )

    maximum = (
        max(
            0.0,
            float(
                finite.max()
            ),
        )
        if finite.size
        else 1.0
    )

    span = (
        maximum
        - minimum
    )

    padding = (
        0.24
        * span
        if span > 0.0
        else 5.0
    )

    ax.set_xlim(
        minimum - padding,
        maximum + padding,
    )

    for (
        candidate,
        reference,
        color,
        alpha,
        legend_label,
    ) in bar_specs:

        for (
            index,
            key,
        ) in enumerate(
            keys
        ):

            value = float(
                comparison[
                    candidate
                ][
                    reference
                ][
                    key
                ]
            )

            add_percentage_label(
                ax,
                value,

                y_positions[
                    index
                ]
                + offsets[
                    (
                        candidate,
                        reference,
                    )
                ],

                color,
            )

    ax.axvline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        metric_labels
    )

    ax.invert_yaxis()

    ax.set_title(
        " ",
        pad=14,
    )

    ax.set_xlabel(
        "QUR relative to the reference algorithm (%)"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,

        loc="upper center",

        bbox_to_anchor=(
            0.5,
            -0.13,
        ),

        ncol=2,
    )

    save_manuscript_png(
        fig,
        "overall_qur_comparison.png",
    )

# Plot routing performance as compromise increases.
def plot_compromised_network_resilience(
    degree: pd.DataFrame,
) -> None:

    reuse_max = float(
        degree[
            "reuse_fraction_pct"
        ].max()
    )

    reuse_ylim = (
        0.0,
        max(
            5.0,
            1.10 * reuse_max,
        ),
    )

    fig = plt.figure(
        figsize=(
            15.6,
            7.0,
        ),
        layout="constrained",
    )

    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[
            5.2,
            1.65,
        ],
    )

    ax = fig.add_subplot(
        grid[
            0,
            0,
        ]
    )

    legend_ax = fig.add_subplot(
        grid[
            0,
            1,
        ]
    )

    legend_ax.axis(
        "off"
    )

    reuse_ax = ax.twinx()

    reuse_x_values = np.sort(
        degree[
            "compromised_percent"
        ].dropna().unique()
    )

    omit_last_reuse_x = None

    if len(reuse_x_values) > 0:

        last_reuse_x = float(
            reuse_x_values[-1]
        )

        last_rows = degree[
            np.isclose(
                degree[
                    "compromised_percent"
                ],
                last_reuse_x,
            )
        ]

        if (
            not last_rows.empty
            and len(last_rows) >= len(ALGORITHM_ORDER)
            and np.all(
                np.isclose(
                    last_rows[
                        "reuse_fraction_pct"
                    ].to_numpy(
                        dtype=float
                    ),
                    0.0,
                )
            )
        ):

            omit_last_reuse_x = (
                last_reuse_x
            )

    for algorithm in ALGORITHM_ORDER:

        subset = degree[
            degree[
                "algorithm"
            ]
            == algorithm
        ].sort_values(
            "compromised_percent"
        )

        valid = subset[
            np.isfinite(
                subset[
                    "eta_average"
                ]
            )
            & (
                subset[
                    "eta_average"
                ]
                > 0.0
            )
        ]

        ax.errorbar(
            valid[
                "compromised_percent"
            ],

            valid[
                "eta_average"
            ],

            yerr=valid[
                "eta_average_sem"
            ].fillna(
                0.0
            ),

            color=COLORS[
                algorithm
            ],

            linestyle="-",

            linewidth=1.5,

            marker=MARKERS[
                algorithm
            ],

            markersize=5.7,

            capsize=2.5,

            zorder=3,
        )

        reuse_subset = (
            subset.copy()
        )

        if omit_last_reuse_x is not None:

            reuse_subset = reuse_subset[
                ~np.isclose(
                    reuse_subset[
                        "compromised_percent"
                    ],
                    omit_last_reuse_x,
                )
            ]

        reuse_ax.plot(
            reuse_subset[
                "compromised_percent"
            ],

            reuse_subset[
                "reuse_fraction_pct"
            ],

            color=COLORS[
                algorithm
            ],

            linestyle="--",

            linewidth=1.4,

            marker=MARKERS[
                algorithm
            ],

            markersize=3.8,


            alpha=0.60,
        )

    ax.set_yscale(
        "log"
    )

    ax.set_title(
        (
            " "
        ),
        pad=14,
    )

    ax.set_xlabel(
        "Untrusted nodes (%)"
    )

    ax.set_ylabel(
        (
            r"Average SKR per node ($\eta_{\text{avg}}$)"
        )
    )

    reuse_ax.set_ylabel(
        (
            "Repeated internal nodes (%)"
        )
    )

    reuse_ax.set_ylim(
        *reuse_ylim
    )

    ax.grid(
        alpha=0.25,
        which="both",
    )

    algorithm_legend = (
        legend_ax.legend(
            handles=(
                manuscript_algorithm_handles()
            ),

            title=(
                "Routing algorithm"
            ),

            frameon=False,

            loc="upper left",
        )
    )

    legend_ax.add_artist(
        algorithm_legend
    )

    style_legend = (
        legend_ax.legend(
            handles=(
                manuscript_style_handles()
            ),

            title=(
                "Line style"
            ),

            frameon=False,

            loc="center left",
        )
    )

    legend_ax.add_artist(
        style_legend
    )


    save_manuscript_png(
        fig,
        "compromised_network_resilience.png",
    )

# Draw one dual-axis manuscript data series.
def draw_manuscript_dual_axis_series(
    ax: plt.Axes,
    aggregate: pd.DataFrame,
    x_column: str,
    left_column: str,
    left_sem_column: str | None,
    annotate_paths: bool,
    right_axis_label: str | None,
    show_right_tick_labels: bool,
    reuse_ylim:
        tuple[float, float],
) -> plt.Axes:

    reuse_ax = ax.twinx()

    reuse_x_values = np.sort(
        aggregate[
            x_column
        ].dropna().unique()
    )

    omit_last_reuse_x = None

    if len(reuse_x_values) > 0:

        last_reuse_x = float(
            reuse_x_values[-1]
        )

        last_rows = aggregate[
            np.isclose(
                aggregate[
                    x_column
                ],
                last_reuse_x,
            )
        ]

        if (
            not last_rows.empty
            and len(last_rows) >= len(ALGORITHM_ORDER)
            and np.all(
                np.isclose(
                    last_rows[
                        "reuse_fraction_pct"
                    ].to_numpy(
                        dtype=float
                    ),
                    0.0,
                )
            )
        ):

            omit_last_reuse_x = (
                last_reuse_x
            )

    for algorithm in ALGORITHM_ORDER:

        subset = aggregate[
            aggregate[
                "algorithm"
            ]
            == algorithm
        ].sort_values(
            x_column
        )

        valid = subset[
            np.isfinite(
                subset[
                    left_column
                ]
            )
        ]

        if left_sem_column is None:

            yerr = None

        else:

            yerr = valid[
                left_sem_column
            ].fillna(
                0.0
            )

        ax.errorbar(
            valid[
                x_column
            ],

            valid[
                left_column
            ],

            yerr=yerr,

            color=COLORS[
                algorithm
            ],

            linestyle="-",

            linewidth=1.5,

            marker=MARKERS[
                algorithm
            ],

            markersize=5.6,

            capsize=2.3,

            zorder=3,
        )

        if (
            annotate_paths
            and not valid.empty
        ):

            left_row = valid.iloc[
                [0]
            ]

            nonzero_path_rows = valid[
                valid[
                    "paths_found"
                ] > 0.0
                ]

            if nonzero_path_rows.empty:



                annotation_rows = (
                    left_row
                )

            else:

                right_row = (
                    nonzero_path_rows.iloc[
                        [-1]
                    ]
                )

                annotation_rows = pd.concat(
                    [
                        left_row,
                        right_row,
                    ]
                )



                annotation_rows = (
                    annotation_rows
                    .drop_duplicates()
                )

            for _, row in annotation_rows.iterrows():

                ax.annotate(
                    f"{row['paths_found']:.2f}",

                    xy=(
                        row[
                            x_column
                        ],

                        row[
                            left_column
                        ],
                    ),

                    xytext=(
                        0,
                        7,
                    ),

                    textcoords=(
                        "offset points"
                    ),

                    ha="center",

                    va="bottom",

                    fontsize=14,

                    color=COLORS[
                        algorithm
                    ],
                )





        reuse_subset = (
            subset.copy()
        )

        if omit_last_reuse_x is not None:

            reuse_subset = reuse_subset[
                ~np.isclose(
                    reuse_subset[
                        x_column
                    ],
                    omit_last_reuse_x,
                )
            ]

        reuse_ax.plot(
            reuse_subset[
                x_column
            ],

            reuse_subset[
                "reuse_fraction_pct"
            ],

            color=COLORS[
                algorithm
            ],

            linestyle="--",

            linewidth=1.35,

            marker=MARKERS[
                algorithm
            ],

            markersize=3.6,



            alpha=0.60,
        )

    reuse_ax.set_ylim(
        *reuse_ylim
    )

    if right_axis_label is not None:

        reuse_ax.set_ylabel(
            right_axis_label
        )

    reuse_ax.tick_params(
        axis="y",

        labelright=(
            show_right_tick_labels
        ),
    )

    ax.grid(
        alpha=0.25,
        which="both",
    )

    return reuse_ax

# Plot multipath performance and node reuse scaling.
def plot_multipath_and_node_reuse_scaling(
    degree: pd.DataFrame,
    radius: pd.DataFrame,
) -> None:

    reuse_max = float(
        pd.concat(
            [
                degree[
                    "reuse_fraction_pct"
                ],

                radius[
                    "reuse_fraction_pct"
                ],
            ],

            ignore_index=True,
        ).max()
    )

    reuse_ylim = (
        0.0,

        max(
            5.0,
            1.10
            * reuse_max,
        ),
    )

    fig = plt.figure(
        figsize=(
            18.8,
            10.8,
        ),

        layout="constrained",
    )


    grid = fig.add_gridspec(
        2,
        3,

        width_ratios=[
            1.0,
            1.0,
            0.43,
        ],
    )

    ax_top_left = fig.add_subplot(
        grid[
            0,
            0,
        ]
    )

    ax_top_right = fig.add_subplot(
        grid[
            0,
            1,
        ]
    )

    ax_bottom_left = fig.add_subplot(
        grid[
            1,
            0,
        ]
    )

    ax_bottom_right = fig.add_subplot(
        grid[
            1,
            1,
        ]
    )

    legend_ax = fig.add_subplot(
        grid[
            :,
            2,
        ]
    )

    legend_ax.axis(
        "off"
    )
    draw_manuscript_dual_axis_series(
        ax_top_left,

        degree,

        "compromised_percent",

        "K_mean",

        "K_sem",

        False,

        None,

        False,

        reuse_ylim,
    )
    draw_manuscript_dual_axis_series(
        ax_top_right,

        radius,

        "x_value",

        "K_mean",

        "K_sem",

        False,

        (
            "Repeated internal nodes (%)"
        ),

        True,

        reuse_ylim,
    )

    draw_manuscript_dual_axis_series(
        ax_bottom_left,

        degree,

        "x_value",

        "n_mean",

        "n_sem",

        False,

        None,

        False,

        reuse_ylim,
    )
    draw_manuscript_dual_axis_series(
        ax_bottom_right,

        radius,

        "x_value",

        "n_mean",

        "n_sem",

        False,

        (
            "Repeated internal nodes (%)"
        ),

        True,

        reuse_ylim,
    )


    ax_top_left.set_yscale(
        "log"
    )

    ax_top_right.set_yscale(
        "log"
    )
    ax_top_left.set_title(
        (
            "Mean path SKR versus "
            "network compromise"
        ),

        pad=10,
    )
    ax_top_right.set_title(
        (
            "Mean path SKR versus "
            "network radius"
        ),

        pad=10,
    )

    ax_bottom_left.set_title(
        (
            "Mean internal nodes versus "
            "degree threshold"
        ),

        pad=10,
    )
    ax_bottom_right.set_title(
        (
            "Mean internal nodes versus "
            "network radius"
        ),

        pad=10,
    )
    ax_top_left.set_xlabel(
        "Untrusted nodes (%)"
    )

    ax_top_right.set_xlabel(
        "Network radius (km)"
    )

    ax_bottom_left.set_xlabel(
        r"Degree threshold $\mathcal{D}$"
    )

    ax_bottom_right.set_xlabel(
        "Network radius (km)"
    )

    ax_top_left.set_ylabel(
        (
            "Mean composable SKR "
            "per path"
        )
    )

    ax_bottom_left.set_ylabel(
        (
            "Mean internal nodes "
            "per path"
        )
    )

    algorithm_legend = (
        legend_ax.legend(
            handles=(
                manuscript_algorithm_handles()
            ),

            title=(
                "Routing algorithm"
            ),

            frameon=False,

            loc="upper left",
        )
    )

    legend_ax.add_artist(
        algorithm_legend
    )

    style_legend = (
        legend_ax.legend(
            handles=(
                manuscript_style_handles()
            ),

            title="Line style",

            frameon=False,

            loc="center left",
        )
    )

    legend_ax.add_artist(
        style_legend
    )

    save_manuscript_png(
        fig,

        (
            "multipath_and_node_"
            "reuse_scaling.png"
        ),
    )

# Generate all manuscript figures from the result table.
def make_plots(
    df: pd.DataFrame,
) -> None:

    print(
        "\nGenerating manuscript figures..."
    )

    work = add_manuscript_columns(
        df
    )
    default_df = select_default_jobs(
        work
    )

    (
        comparison,
        overall_table,
    ) = build_overall_fixed_comparison(
        default_df
    )
    degree = aggregate_manuscript_scan(
        work,

        "degree_threshold",
    )

    radius = aggregate_manuscript_scan(
        work,

        "radius",
    )
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_table.to_csv(
        FIGURE_DIR
        / "manuscript_overall_metrics.csv",

        index=False,
    )

    degree.to_csv(
        FIGURE_DIR
        / "manuscript_degree_threshold_metrics.csv",

        index=False,
    )

    radius.to_csv(
        FIGURE_DIR
        / "manuscript_radius_metrics.csv",

        index=False,
    )
    export_latex_values(
        default_df,
        comparison,
        degree,
    )
    plot_overall_qur_comparison(
        comparison
    )

    plot_compromised_network_resilience(
        degree
    )

    plot_multipath_and_node_reuse_scaling(
        degree,
        radius,
    )

# Fingerprint the completed-results search space.
def results_search_space_fingerprint() -> str:
    payload = {

        "global_seed":
            int(
                GLOBAL_RANDOM_SEED
            ),

        "num_batch_runs":
            int(
                NUM_BATCH_RUNS
            ),

        "default_configuration": {

            "num_nodes":
                int(
                    NUM_NODES
                ),

            "radius":
                float(
                    RADIUS
                ),

            "r0_ratio":
                float(
                    R0_RATIO
                ),

            "beta_waxman":
                float(
                    BETA_WAXMAN
                ),

            "prob_comp":
                float(
                    PROB_COMP
                ),

            "degree_threshold":
                int(
                    DEGREE_THRESHOLD
                ),

            "max_paths":
                int(
                    MAX_PATHS
                ),
        },

        "enabled_scans":
            list(
                ENABLED_SCANS
            ),

        "scan_values": {

            name: [
                float(value)

                for value
                in SCAN_VALUE_MAP[
                    name
                ]
            ]

            for name
            in ENABLED_SCANS
        },

        "weight_search": {

            "trials":
                int(
                    WEIGHT_TRIALS
                ),

            "names":
                list(
                    opt.WEIGHT_NAMES
                ),

            "min":
                float(
                    opt.WEIGHT_MIN
                ),

            "max":
                float(
                    opt.WEIGHT_MAX
                ),

            "decimals":
                int(
                    opt.WEIGHT_DECIMALS
                ),

            "rate_fixed":
                float(
                    opt.RATE_WEIGHT_FIXED
                ),

            "a_star_penalty":
                float(
                    opt.A_STAR_PENALTY_STRENGTH
                ),
        },

        "fixed_learning": {

            "scope":
                str(
                    FIXED_WEIGHT_SCOPE
                ),

            "top_candidates_per_graph":
                TOP_CANDIDATES_PER_GRAPH,

            "extra_candidates":
                int(
                    EXTRA_FIXED_CANDIDATES
                ),

            "model_random_seed":
                int(
                    MODEL_RANDOM_SEED
                ),

            "model_max_iter":
                int(
                    MODEL_MAX_ITER
                ),

            "model_learning_rate":
                float(
                    MODEL_LEARNING_RATE
                ),

            "model_max_leaf_nodes":
                int(
                    MODEL_MAX_LEAF_NODES
                ),

            "model_min_samples_leaf":
                int(
                    MODEL_MIN_SAMPLES_LEAF
                ),

            "model_l2_regularization":
                float(
                    MODEL_L2_REGULARIZATION
                ),
        },
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()[:20]

# Build the set of result keys expected for this run.
def expected_result_keys(
    jobs: list[dict],
) -> set[tuple]:

    expected = set()

    for job in jobs:

        settings = (
            job[
                "settings"
            ]
        )

        base = (

            str(
                job[
                    "scan_name"
                ]
            ),

            round(
                float(
                    job[
                        "x_value"
                    ]
                ),
                10,
            ),

            int(
                job[
                    "seed"
                ]
            ),

            int(
                settings[
                    "num_nodes"
                ]
            ),

            round(
                float(
                    settings[
                        "radius"
                    ]
                ),
                10,
            ),

            round(
                float(
                    settings[
                        "r0_ratio"
                    ]
                ),
                12,
            ),

            round(
                float(
                    settings[
                        "beta_waxman"
                    ]
                ),
                12,
            ),

            round(
                float(
                    settings[
                        "prob_comp"
                    ]
                ),
                12,
            ),

            int(
                settings[
                    "degree_threshold"
                ]
            ),

            int(
                settings[
                    "max_paths"
                ]
            ),
        )

        for algorithm in ALGORITHM_ORDER:

            expected.add(
                base
                + (
                    algorithm,
                )
            )

    return expected

# Build the set of result keys present in saved results.
def actual_result_keys(
    df: pd.DataFrame,
) -> set[tuple]:

    required = [

        "scan_name",
        "x_value",
        "seed",

        "num_nodes",
        "radius",
        "r0_ratio",

        "beta_waxman",
        "prob_comp",

        "degree_threshold",
        "max_paths",

        "algorithm",
    ]

    if any(
        column
        not in df.columns

        for column
        in required
    ):

        return set()

    keys = set()

    for row in df[
        required
    ].itertuples(
        index=False
    ):

        keys.add(
            (

                str(
                    row.scan_name
                ),

                round(
                    float(
                        row.x_value
                    ),
                    10,
                ),

                int(
                    row.seed
                ),

                int(
                    row.num_nodes
                ),

                round(
                    float(
                        row.radius
                    ),
                    10,
                ),

                round(
                    float(
                        row.r0_ratio
                    ),
                    12,
                ),

                round(
                    float(
                        row.beta_waxman
                    ),
                    12,
                ),

                round(
                    float(
                        row.prob_comp
                    ),
                    12,
                ),

                int(
                    row.degree_threshold
                ),

                int(
                    row.max_paths
                ),

                str(
                    row.algorithm
                ),
            )
        )

    return keys

# Save the search space signature beside the result CSV.
def write_results_search_space_signature(
    signature: str,
) -> None:

    payload = {

        "search_space_fingerprint":
            str(
                signature
            ),

        "results_csv":
            str(
                RESULTS_CSV
            ),
    }
    temp = (
        RESULTS_SEARCH_SPACE_JSON
        .with_name(
            RESULTS_SEARCH_SPACE_JSON.name
            + f".{os.getpid()}.tmp"
        )
    )
    temp.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(
        temp,
        RESULTS_SEARCH_SPACE_JSON,
    )

# Reuse saved results when the full search space matches.
def load_complete_results_if_available(
    jobs: list[dict],
) -> pd.DataFrame | None:

    if not REUSE_COMPLETE_RESULTS:
        return None
    if not RESULTS_CSV.exists():
        return None
    try:

        df = pd.read_csv(
            RESULTS_CSV
        )
    except Exception:

        return None
    expected = expected_result_keys(
        jobs
    )
    actual = actual_result_keys(
        df
    )
    if not expected.issubset(
        actual
    ):
        missing_count = (
            len(
                expected
                - actual
            )
        )
        print(
            "\nExisting final CSV is not complete "
            "for the requested search space."
        )
        print(
            f"Missing result rows: "
            f"{missing_count:,}"
        )
        return None
    current_signature = (
        results_search_space_fingerprint()
    )
    if RESULTS_SEARCH_SPACE_JSON.exists():
        try:
            saved = json.loads(
                RESULTS_SEARCH_SPACE_JSON.read_text(
                    encoding="utf-8"
                )
            )

            saved_signature = (
                saved.get(
                    "search_space_fingerprint"
                )
            )
        except Exception:

            saved_signature = None
        if (
            saved_signature is not None
            and saved_signature
            != current_signature
        ):
            print(
                "\nExisting CSV is complete in terms "
                "of rows, but the calculation/search "
                "settings changed."
            )

            print(
                "Re-entering the calculation pipeline."
            )

            return None
    else:
        write_results_search_space_signature(
            current_signature
        )
    print(
        "\n"
        + "=" * 88
    )

    print(
        "COMPLETE CALCULATIONS FOUND"
    )

    print(
        "=" * 88
    )

    print(
        f"Result CSV: {RESULTS_CSV}"
    )

    print(
        "The current calculation/search space "
        "is already complete."
    )

    print(
        "Skipping:"
    )

    print(
        "  - topology generation"
    )

    print(
        "  - A* calculation"
    )

    print(
        "  - Max-Min calculation"
    )

    print(
        "  - 1,000-trial per-seed QUR optimization"
    )

    print(
        "  - ML surrogate training"
    )

    print(
        "  - learned fixed-weight search"
    )

    print(
        "  - fixed-weight route evaluation"
    )

    print(
        "\nOnly the manuscript figures will be regenerated."
    )

    print(
        "=" * 88
        + "\n"
    )

    return df
# Print a compact summary of the requested workload.
def print_workload(jobs: list[dict], cache_tasks: list[tuple], fingerprint: str) -> None:
    number_of_points = sum(len(SCAN_VALUE_MAP[x]) for x in ENABLED_SCANS)
    print("=" * 88)
    print("Composable CV-QKD scan: four routing curves")
    print(f"Fingerprint             : {fingerprint}")
    print(f"Scans                   : {ENABLED_SCANS}")
    print(f"Scan points             : {number_of_points}")
    print(f"Seeds per point         : {NUM_BATCH_RUNS}")
    print(f"Graph jobs              : {len(jobs):,}")
    print(f"Unique topology caches  : {len(cache_tasks):,}")
    print(f"Weight trials / graph   : {WEIGHT_TRIALS:,}")
    print(f"Fixed-weight scope      : {FIXED_WEIGHT_SCOPE}")
    print(f"Workers                 : {MAX_WORKERS if USE_MULTIPROCESSING else 1}")
    print("Algorithms              : A*, Max-Min, QUR optimized, QUR fixed learned")
    print(f"Results CSV             : {RESULTS_CSV}")
    print(f"Fixed weights CSV       : {FIXED_WEIGHTS_CSV}")
    print(f"Figure directory        : {FIGURE_DIR}")
    print("=" * 88)

# Print the learned fixed weights by scan group.
def print_fixed_weights(fixed_df: pd.DataFrame) -> None:
    print("\nLearned fixed QUR weights")
    print("=" * 88)
    columns = [
        "scan_name",
        "x_value",
        *[f"weight_{name}" for name in opt.WEIGHT_NAMES],
        "predicted_mean_retention",
    ]
    existing = [column for column in columns if column in fixed_df.columns]
    print(
        fixed_df[existing].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

# Parse arguments and run the workflow.
def main() -> None:
    if FIXED_WEIGHT_SCOPE not in {
        "per_scan_point",
        "global",
    }:
        raise ValueError(
            "FIXED_WEIGHT_SCOPE must be "
            "'per_scan_point' or 'global'"
        )
    LANDSCAPE_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIXED_EVAL_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    opt.configure_numba_threads(
        NUMBA_THREADS_PER_WORKER
    )
    opt.warmup_numba()
    opt.load_p2p_data()
    fingerprint = (
        experiment_fingerprint()
    )
    jobs, cache_tasks = (
        build_scan_jobs()
    )
    print_workload(
        jobs,
        cache_tasks,
        fingerprint,
    )
    existing_df = (
        load_complete_results_if_available(
            jobs
        )
    )
    if existing_df is not None:

        make_plots(
            existing_df
        )

        print(
            "\nDone."
        )

        print(
            "Existing calculations were "
            "reused unchanged."
        )

        print(
            f"Results: "
            f"{RESULTS_CSV.resolve()}"
        )

        print(
            f"Figures: "
            f"{FIGURE_DIR.resolve()}"
        )

        return
    prepare_network_caches(
        cache_tasks
    )
    path_by_job_index = (
        run_base_jobs(
            jobs,
            fingerprint,
        )
    )
    (
        weights_by_group,
        info_by_group,
        fixed_df,
    ) = learn_all_fixed_weights(
        jobs,
        path_by_job_index,
    )
    print_fixed_weights(
        fixed_df
    )
    gc.collect()
    _malloc_trim()
    fixed_path_by_job_index = (
        run_fixed_evaluations(
            jobs,
            path_by_job_index,
            weights_by_group,
            info_by_group,
            fingerprint,
        )
    )
    df = build_final_results(
        jobs,
        path_by_job_index,
        fixed_path_by_job_index,
    )
    write_results_search_space_signature(
        results_search_space_fingerprint()
    )
    make_plots(
        df
    )
    print(
        "\nDone."
    )
    print(
        f"Results:       "
        f"{RESULTS_CSV.resolve()}"
    )
    print(
        f"Fixed weights: "
        f"{FIXED_WEIGHTS_CSV.resolve()}"
    )
    print(
        f"Figures:       "
        f"{FIGURE_DIR.resolve()}"
    )
# Run the command-line entry point.
if __name__ == "__main__":
    main()

