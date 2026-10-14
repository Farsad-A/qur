#!/usr/bin/env python3

# Import required libraries.
import argparse
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configure model and runtime parameters.
ALPHA_DB_PER_KM = 0.2
BLOCK_SIZE = int(1e9)
PE_FRACTION = 0.10
BETA = 0.98
FER = 0.10
ALPHABET_SIZE = 2**5
EPS_COR = 1e-10
EPS_SMOOTH = 1e-10
EPS_HASH = 1e-10
EPS_PE = 1e-10

R_DB_VALUES = tuple(range(1, 31))
EXCESS_NOISE_VALUES = tuple(np.round(np.arange(0.01, 0.31, 0.01), 3))
DISTANCES_KM = tuple(np.round(np.arange(0.1, 300.1, 0.1), 2))

OUTPUT_NAME = "p2p_composable_TMSV.txt"
PROCESSES = 4
STOP_AFTER_FIRST_NONPOSITIVE = True
MODEL_VERSION = "p2p_zero_relay_single_link_composable_model_v1"

COLUMNS = (
    "r_db",
    "eps",
    "d_AB",
    "SKR",
    "pr",
    "t_param",
    "N",
    "key_length",
    "raw_key_length_bound",
    "n_PE",
    "n_key",
    "p_EC",
    "T",
    "xi",
    "a",
    "b",
    "c",
    "delta_PE",
    "a_worst",
    "b_worst",
    "c_worst",
    "nu_1",
    "nu_2",
    "nu_cond",
    "I_AB",
    "chi_BE",
    "beta_I_AB",
    "asymptotic_term",
    "Delta_AEP",
    "Theta",
    "eps_cor",
    "eps_smooth",
    "eps_hash",
    "eps_PE",
    "FER",
    "beta",
    "alphabet_size",
    "model_version",
)

# Convert distance to channel transmittance.
def transmittance(distance_km):
    return 10.0 ** (-ALPHA_DB_PER_KM * float(distance_km) / 10.0)

# Evaluate the bosonic entropy term.
def bosonic_entropy(nu):
    nu = float(nu)
    if not math.isfinite(nu) or nu < 1.0 - 1e-8:
        raise ValueError(f"Nonphysical symplectic eigenvalue: {nu}")
    nu = max(nu, 1.0)
    nbar = (nu - 1.0) / 2.0
    if nbar == 0.0:
        return 0.0
    return (nbar + 1.0) * math.log2(nbar + 1.0) - nbar * math.log2(nbar)

# Compute the symplectic eigenvalues
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
        raise ValueError(
            f"Negative symplectic eigenvalue square: {nu1_sq}, {nu2_sq}"
        )

    nu1 = math.sqrt(max(nu1_sq, 0.0))
    nu2 = math.sqrt(max(nu2_sq, 0.0))
    if nu1 < 1.0 - 1e-8 or nu2 < 1.0 - 1e-8:
        raise ValueError(f"Nonphysical covariance matrix: {nu1}, {nu2}")
    return max(nu1, 1.0), max(nu2, 1.0)

# Build the single link covariance parameters
def single_link_covariance(distance_km, r_db, excess_noise):
    r = float(r_db) * math.log(10.0) / 20.0
    mu = math.cosh(2.0 * r)
    T = transmittance(distance_km)

    a = mu
    b = T * mu + 1.0 - T + float(excess_noise)
    c = math.sqrt(T * (mu * mu - 1.0))
    return T, a, b, c

# Evaluate the finite size P2P SKR.
def evaluate_p2p(distance_km, r_db, excess_noise):
    T, a, b, c = single_link_covariance(distance_km, r_db, excess_noise)

    n_pe = int(BLOCK_SIZE * PE_FRACTION)
    n_key = int(BLOCK_SIZE) - n_pe
    p_ec = 1.0 - FER
    if n_pe <= 0 or n_key <= 0:
        raise ValueError("Invalid finite-size block split")
    if not 0.0 < p_ec <= 1.0:
        raise ValueError(f"Invalid error-correction success probability: {p_ec}")

    delta_pe = math.sqrt(4.0 * math.log(8.0 / EPS_PE) / n_pe)
    a_wc = a + delta_pe * (a + 1.0)
    b_wc = b + delta_pe * (b + 1.0)
    c_wc = c - 0.5 * delta_pe * (a + b + 2.0)

    nu1, nu2 = symplectic_eigenvalues(a_wc, b_wc, c_wc)
    nu_cond = a_wc - c_wc * c_wc / (b_wc + 1.0)
    if nu_cond < 1.0 - 1e-8:
        raise ValueError(f"Nonphysical conditional eigenvalue: {nu_cond}")
    nu_cond = max(nu_cond, 1.0)

    i_ab = math.log2((a_wc + 1.0) / (nu_cond + 1.0))
    chi_be = (
        bosonic_entropy(nu1)
        + bosonic_entropy(nu2)
        - bosonic_entropy(nu_cond)
    )
    asymptotic_term = BETA * i_ab - chi_be

    delta_aep = (
        4.0
        * math.log2(math.sqrt(float(ALPHABET_SIZE)) + 2.0)
        * math.sqrt(
            math.log2(18.0 * p_ec ** (-2.0) * EPS_SMOOTH ** (-4.0))
        )
    )
    theta = math.log2(
        p_ec * (1.0 - EPS_SMOOTH**2 / 3.0)
    ) + 2.0 * math.log2(math.sqrt(2.0) * EPS_HASH)

    raw_rate = (n_key * p_ec / BLOCK_SIZE) * (
        asymptotic_term - delta_aep / math.sqrt(n_key) + theta / n_key
    )
    raw_key_length_bound = BLOCK_SIZE * raw_rate
    key_length = max(math.floor(raw_key_length_bound), 0)
    skr = max(float(raw_rate), 0.0)

    return {
        "r_db": int(r_db),
        "eps": float(excess_noise),
        "d_AB": float(distance_km),
        "SKR": skr,
        "pr": 1.0,
        "t_param": float("nan"),
        "N": int(BLOCK_SIZE),
        "key_length": int(key_length),
        "raw_key_length_bound": float(raw_key_length_bound),
        "n_PE": int(n_pe),
        "n_key": int(n_key),
        "p_EC": float(p_ec),
        "T": float(T),
        "xi": float(excess_noise),
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "delta_PE": float(delta_pe),
        "a_worst": float(a_wc),
        "b_worst": float(b_wc),
        "c_worst": float(c_wc),
        "nu_1": float(nu1),
        "nu_2": float(nu2),
        "nu_cond": float(nu_cond),
        "I_AB": float(i_ab),
        "chi_BE": float(chi_be),
        "beta_I_AB": float(BETA * i_ab),
        "asymptotic_term": float(asymptotic_term),
        "Delta_AEP": float(delta_aep),
        "Theta": float(theta),
        "eps_cor": float(EPS_COR),
        "eps_smooth": float(EPS_SMOOTH),
        "eps_hash": float(EPS_HASH),
        "eps_PE": float(EPS_PE),
        "FER": float(FER),
        "beta": float(BETA),
        "alphabet_size": int(ALPHABET_SIZE),
        "model_version": MODEL_VERSION,
    }

# Format one output value for the lookup table.
def format_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"
def make_row(result):
    return ",".join(format_value(result[name]) for name in COLUMNS) + "\n"

# Evaluate one parameter pair across the distance grid.
def run_job(job):
    r_db, excess_noise, distances = job
    rows = []
    for distance in distances:
        result = evaluate_p2p(distance, r_db, excess_noise)
        rows.append(make_row(result))
        if STOP_AFTER_FIRST_NONPOSITIVE and result["SKR"] <= 0.0:
            break
    return "".join(rows)

# Create the output table and header.
def initialize_output(path, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")

# Recover parameter pairs already written to disk for efficiency.
def completed_jobs(path):
    completed = set()
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            fields = line.split(",")
            completed.add((int(float(fields[0])), round(float(fields[1]), 3)))
    return completed

# Generate the lookup table, optionally in parallel.
def run_generation(output_path, processes, overwrite, quick_test):
    initialize_output(output_path, overwrite)
    completed = set() if overwrite else completed_jobs(output_path)

    if quick_test:
        r_values = (5,)
        noise_values = (0.01,)
        distances = (0.1, 5.0, 20.0)
    else:
        r_values = R_DB_VALUES
        noise_values = EXCESS_NOISE_VALUES
        distances = DISTANCES_KM

    jobs = [
        (r_db, round(float(eps), 3), tuple(distances))
        for r_db in r_values
        for eps in noise_values
        if (int(r_db), round(float(eps), 3)) not in completed
    ]

    print("=== P2P composable lookup generator ===")
    print("Model       : zero-relay single thermal-loss link")
    print(f"Block size  : {BLOCK_SIZE:,}")
    print(f"PE fraction : {PE_FRACTION}")
    print(f"Processes   : {processes}")
    print(f"Jobs        : {len(jobs)}")
    print(f"Output      : {output_path}")

    if processes == 1:
        iterator = map(run_job, jobs)
        for text in tqdm(iterator, total=len(jobs), desc="P2P jobs"):
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(text)
        return

    chunksize = max(1, len(jobs) // (int(processes) * 16))
    with mp.Pool(processes=int(processes)) as pool:
        for text in tqdm(
            pool.imap_unordered(run_job, jobs, chunksize=chunksize),
            total=len(jobs),
            desc="P2P jobs",
        ):
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(text)

# Check the P2P model against consistency tests.
def run_self_test():
    result = evaluate_p2p(0.1, 14, 0.005)
    if not math.isfinite(result["SKR"]):
        raise AssertionError("P2P self-test returned a non-finite key rate")

    T, a, b, c = single_link_covariance(10.0, 14, 0.005)
    mu = a
    if not math.isclose(b, T * mu + 1.0 - T + 0.005, rel_tol=1e-14):
        raise AssertionError("Single-link covariance convention mismatch")
    if not math.isclose(c, math.sqrt(T * (mu * mu - 1.0)), rel_tol=1e-14):
        raise AssertionError("Single-link correlation mismatch")

    print("P2P self-test passed.")
    print(f"Representative SKR: {result['SKR']:.6e}")

# Define the command line options
def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a zero-relay P2P composable TMSV lookup"
    )
    parser.add_argument("--output-dir", default="p2p")
    parser.add_argument("--processes", type=int, default=PROCESSES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser

# Parse arguments and run the workflow.
def main():
    args = build_parser().parse_args()
    if args.processes < 1:
        raise ValueError("--processes must be at least 1")

    if args.self_test:
        run_self_test()
        return

    output_path = Path(args.output_dir).expanduser().resolve() / OUTPUT_NAME
    run_generation(
        output_path=output_path,
        processes=args.processes,
        overwrite=args.overwrite,
        quick_test=args.quick_test,
    )
    print("Generation complete.")

# Run the command line entry point.
if __name__ == "__main__":
    main()
