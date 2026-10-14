HOW TO USE THE CV-QKD / QUR CODE
================================

This folder contains three main Python scripts:

1. generate_p2p_composable.py
   Generates the composable point-to-point (P2P) secret-key-rate lookup table.

2. optimize_fsqur.py
   Contains the network model, A* routing, QUR routing, composable swap-link calculations,
   Numba-accelerated weight evaluation, and the standalone QUR weight optimizer.

3. network_scan_four_algorithms_ml_fixed.py
   Runs the complete network study and compares:
      - A*
      - Max-Min
      - QUR-Optimized
      - QUR-Fixed-Learned


1. REQUIREMENTS
===============

Keep these files in the same working folder:

    generate_p2p_composable.py
    optimize_fsqur.py
    network_scan_four_algorithms_ml_fixed.py

The scan script imports optimize_fsqur.py directly so the two files should normally remain in the same directory.


2. NORMAL WORKFLOW
==================

STEP 1 - Generate the P2P lookup table
--------------------------------------
Run:
    python generate_p2p_composable.py

By default this creates:
    p2p/p2p_composable_TMSV.txt

The network code expects this lookup table at that location.

STEP 2 - Run the complete four-algorithm network scan
-----------------------------------------------------

After the P2P lookup exists, run:
    python network_scan_four_algorithms_ml_fixed.py

The script will automatically:

    1. Load the P2P lookup.
    2. Create/reuse network topology caches.
    3. Evaluate A* and Max-Min.
    4. Search QUR weights for each graph.
    5. Train the fixed-weight ML surrogate.
    6. Evaluate the learned fixed QUR weights.
    7. Assemble the result tables.
    8. Generate manuscript plots and Latex numerical values.

Main outputs include:
    qkd_network_scan_results_4alg.csv  which has the complete results for all algorithms, seeds and scan points.

    learned_fixed_weights_by_configuration.csv  which has the learned fixed QUR weights and associated training information.

    qur_claims_values.tex which produces a numerical values exported as Latex commands for the manuscript.

    four_algorithm_figures/ that stores the generated manuscript figures.

The code also creates reusable calculation caches:

    network_topology_cache/
    scan_ml_landscape_cache/
    scan_fixed_eval_cache/


3. MAIN SETTINGS FOR THE NETWORK STUDY
======================================

The main study settings are near the top of:

    network_scan_four_algorithms_ml_fixed.py

Default graph ensemble:

    GLOBAL_RANDOM_SEED = 100
    NUM_BATCH_RUNS = 50

Default network parameters:

    NUM_NODES = 1000
    RADIUS = 50.0
    R0_RATIO = 0.07
    BETA_WAXMAN = 1.0
    PROB_COMP = 0.0
    DEGREE_THRESHOLD = 10
    MAX_PATHS = 5

Enabled scans are controlled by:

    ENABLED_SCANS

Current scan ranges are defined by:

    RADII_SCAN
    MAX_PATHS_SCAN
    DEGREE_THRESHOLD_SCAN
    NUM_NODES_SCAN

For example, to run only radius and degree threshold scans, change:

    ENABLED_SCANS = [
        "radius",
        "degree_threshold",
    ]

Multiprocessing is controlled by:

    USE_MULTIPROCESSING = True
    MAX_WORKERS = 4
    NUMBA_THREADS_PER_WORKER = 1

A safe starting point on a typical desktop is:

    MAX_WORKERS = 4
    NUMBA_THREADS_PER_WORKER = 1

Increase MAX_WORKERS only if sufficient CPU cores and RAM is available.


4. QUR WEIGHT OPTIMIZATION SETTINGS
===================================

The main QUR settings are near the top of optimize_fsqur.py.

Important values include:

    OPTIMIZATION_ITERATIONS = 8000
    OPTIMIZATION_RANDOM_SEED = 100
    WEIGHT_MIN = 0.0
    WEIGHT_MAX = 1.0
    WEIGHT_DECIMALS = 2
    RATE_WEIGHT_FIXED = 1.0
    A_STAR_PENALTY_STRENGTH = 0.5

The seven QUR weights are:

    rate
    distance
    swap_relay
    hop
    swap_away
    swap_detour
    heuristic

The rate weight is fixed at 1.0. The remaining six weights are sampled.

The full network scan uses:

    WEIGHT_TRIALS = opt.OPTIMIZATION_ITERATIONS

Changing OPTIMIZATION_ITERATIONS in optimize_fsqur.py also changes the per-graph QUR search budget used by the scan script.


5. FIXED-LEARNED QUR SETTINGS
=============================

The fixed-weight learning mode is controlled in:

    network_scan_four_algorithms_ml_fixed.py

Default:

    FIXED_WEIGHT_SCOPE = "per_scan_point"

This learns one fixed QUR weight vector from the graph ensemble belonging to each scan point.
For example, all graphs at one radius share one learned fixed vector for that radius.

The ML search is controlled mainly by:

    TOP_CANDIDATES_PER_GRAPH = 500
    EXTRA_FIXED_CANDIDATES = 10000
    ML_CANDIDATE_BATCH_SIZE = 5000

and the HistGradientBoostingRegressor settings beginning with:

    MODEL_...

Normally these do not need to be changed.


6. EXPECTED FOLDER STRUCTURE
=============================

A typical working directory will eventually look like:

    project/
    |
    |-- generate_p2p_composable.py
    |-- optimize_fsqur.py
    |-- network_scan_four_algorithms_ml_fixed.py
    |
    |-- p2p/
    |   `-- p2p_composable_TMSV.txt
    |
    |-- network_topology_cache/
    |-- scan_ml_landscape_cache/
    |-- scan_fixed_eval_cache/
    |-- four_algorithm_figures/
    |
    |-- qkd_network_scan_results_4alg.csv
    |-- learned_fixed_weights_by_configuration.csv
    `-- qur_claims_values.tex
