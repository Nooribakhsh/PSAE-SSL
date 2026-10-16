# PSAE-SSL — Parallel Stacked Autoencoder with Semi-Supervised Learning

MATLAB implementation of PSAE-SSL. Every experiment is a script in `experiments/`.

## Stages (paper §3)

| Folder | Section | Role |
|--------|---------|------|
| `src/stage1_node2vec/` | §3.1 | Node2Vec embeddings |
| `src/stage2_psae/` | §3.2 | Parallel stacked autoencoders + semi-supervised training |
| `src/stage3_ahc/` | §3.3 | Agglomerative hierarchical clustering |
| `src/evaluation/` | — | NMI, Q, P/R/F1, separability |
| `src/utils/` | — | loaders, sampling, `psae_pipeline` |

## Repository structure

```
PSAE-SSL/
├── README.md                                 # This file
├── LICENSE                                   # License
├── data/                                     # Benchmark datasets (provided): edges.txt + labels.txt per dataset
│   ├── README.md                             # Dataset table, file format, DBLP notes
│   ├── karate/                               # Zachary karate club
│   ├── dolphins/                             # Dolphin social network
│   ├── football/                             # American college football
│   ├── polbooks/                             # Political books
│   ├── polblogs/                             # Political blogs (directed)
│   ├── email/                                # Email-Eu-core (directed)
│   ├── cora/                                 # Cora citation network
│   ├── citeseer/                             # CiteSeer citation network (directed)
│   ├── pubmed/                               # PubMed citation network
│   └── dblp/                                 # DBLP co-authorship (317,080 nodes)
├── src/                                      # Source code (added to the MATLAB path automatically by every script)
│   ├── stage1_node2vec/                      # Stage 1: topology-aware node embedding
│   │   ├── node2vec.m                    # Stage 1 entry point: walks + Skip-gram -> embedding Z0
│   │   ├── node2vec_walk.m                   # Biased second-order random walk
│   │   ├── build_transition_probs.m          # Precomputes the walk transition probabilities
│   │   ├── alias_setup.m                     # Alias tables for O(1) sampling
│   │   ├── alias_draw.m                      # Draws one index from an alias table
│   │   └── train_skipgram.m                  # Skip-gram with negative sampling
│   ├── stage2_psae/                          # Stage 2: parallel stacked autoencoder (PSAE)
│   │   ├── train_psae.m                      # Stage 2 entry point: pre-training + fine-tuning (full batch)
│   │   ├── train_psae_minibatch.m            # Mini-batch version of train_psae
│   │   ├── psae_init.m                       # Builds the parallel branches with class-associated masks
│   │   ├── init_branch.m                     # One encoder-decoder branch
│   │   ├── init_he.m                         # He initialization (ReLU layers)
│   │   ├── init_lecun.m                      # LeCun initialization (sigmoid layers)
│   │   ├── psae_forward.m                    # Forward pass through all branches
│   │   ├── psae_backward.m                   # Backpropagation of the total loss
│   │   ├── branch_forward.m                  # Forward pass of one branch
│   │   ├── branch_backward.m                 # Backward pass of one branch
│   │   ├── dense_forward.m                   # Fully-connected layer, forward
│   │   ├── dense_backward.m                  # Fully-connected layer, backward
│   │   ├── relu.m / relu_grad.m              # ReLU activation and its gradient
│   │   ├── sigmoid.m / sigmoid_grad.m        # Sigmoid activation and its gradient
│   │   ├── compute_losses.m                  # Reconstruction + supervised + separation loss
│   │   ├── separation_loss_and_grad.m        # Margin separation and intra-class compactness loss
│   │   ├── generate_pseudo_labels.m          # Pseudo-label generation and refresh
│   │   ├── apply_class_masks.m               # Specializes branch input masks using labeled nodes
│   │   ├── build_class_idx_cell.m            # Class index sets from labeled + pseudo-labeled nodes
│   │   ├── stratified_split.m                # Stratified train/validation split of labeled nodes
│   │   ├── adam_step.m                       # One Adam update for one parameter
│   │   ├── apply_adam_to_model.m             # Adam update for all trainable parameters
│   │   └── init_adam_moments.m               # Zero-initialized Adam moment buffers
│   ├── stage3_ahc/                           # Stage 3: hierarchical clustering and refinement
│   │   ├── ahc.m                         # Stage 3 entry point: agglomerative hierarchical clustering
│   │   ├── cut_ahc_clusters.m                # Cuts the dendrogram at a given number of clusters
│   │   ├── select_k_by_modularity.m          # Chooses the cut with maximum modularity
│   │   ├── cluster_embeddings.m              # Candidate partitions on the embeddings, best by modularity
│   │   ├── post_cut_reassignment.m           # Re-scores each node against all clusters after the cut
│   │   ├── node_cluster_affinity.m           # Node-to-cluster affinity
│   │   ├── reassign_isolates.m               # Reassigns clusters that could not be merged
│   │   ├── label_guided_clustering.m         # Semi-supervised (label-guided) refinement
│   │   ├── select_label_embedding.m          # Chooses the embedding used by label-guided clustering
│   │   ├── compute_modularity.m              # Modularity Q
│   │   ├── modularity_gamma.m                # Per-dataset modularity resolution
│   │   ├── find_root.m                       # Union-find helper
│   │   └── simple_kmeans.m                   # Lloyd k-means (no toolbox needed)
│   ├── evaluation/                           # Evaluation metrics
│   │   ├── evaluate_partition.m              # NMI, Q, precision, recall, F1 of a partition
│   │   ├── compute_nmi.m                     # Normalized mutual information
│   │   ├── compute_prf.m                     # Precision / recall / F1 (majority-vote cluster-to-class map)
│   │   └── separability_metrics.m            # Latent-space separability (FDR, SC, DBI, CHI)
│   └── utils/                                # Pipeline, settings, data handling, statistics
│       ├── get_user_settings.m               # ALL user-adjustable settings (datasets, epochs, hyper-parameters, ...)
│       ├── get_training_budget.m             # Epoch budget for a graph of n nodes (reads usr.epochs)
│       ├── get_dataset_config.m              # Per-dataset architecture and hyper-parameters
│       ├── get_n2v_options.m                 # Node2Vec options with the central epoch budget applied
│       ├── get_results_dir.m                 # results/ or results_cfgXXXXXX/ for changed settings
│       ├── result_is_current.m               # Checks whether a saved result matches the current settings
│       ├── psae_pipeline.m               # One full run: Stage 1 -> 2 -> 3 -> metrics
│       ├── ensure_datasets.m                 # Downloads missing datasets from Zenodo before a run
│       ├── load_dataset.m                    # Loads edges, labels and K of a dataset
│       ├── load_graph.m                      # Edge list -> adjacency list
│       ├── load_labels.m                     # Reads labels.txt
│       ├── sample_labeled_set.m              # Random labeled-node sampling (10%)
│       ├── sample_labeled_set_stratified.m   # Stratified labeled-node sampling
│       ├── adjlist_to_edgelist.m             # Adjacency list -> unique edge list
│       ├── adjlist_to_sparse.m               # Adjacency list -> sparse adjacency matrix
│       ├── adjlist_to_features.m             # Sparse structural features (no-Node2Vec variant)
│       ├── get_peak_mem_mb.m                 # Peak memory of the MATLAB process
│       ├── friedman_test.m                   # Friedman test with Iman-Davenport correction
│       ├── wilcoxon_signed_rank.m            # Paired Wilcoxon signed-rank test
│       ├── holm_adjust.m                     # Holm p-value adjustment
│       ├── avg_rank.m                        # Ranks with average ranks for ties
│       └── paper_baseline_means.m            # Baseline mean scores printed in the manuscript
├── experiments/                              # Scripts to press Run on (order: README_ORDER.txt)
│   ├──                   
│   ├── everything.m                      # Runs all experiments in order
│   ├── all_datasets.m                    # Main results on all datasets (10 runs each)
│   ├── nmi_q_log.m                       # Quick NMI / Q / P / R / F1 log for all datasets
│   ├── ablation_all.m                    # Ablation study (six variants)
│   ├── sensitivity_all.m                 # Runs both sensitivity experiments
│   ├── sensitivity_branches_all.m        # Sensitivity to the number of branches B
│   ├── sensitivity_label_all.m           # Sensitivity to the labeled-node ratio
│   ├── separability_all.m                # Latent-space separability before / after PSAE-SSL
│   ├── scalability_all.m                 # Training time, inference time, peak memory (all datasets)
│   ├── scalability_core.m                # Scalability driver (one fresh MATLAB process per run)
│   ├── scal_worker.m                         # Worker process of one scalability run
│   ├── statistical_tests.m               # Friedman and Wilcoxon tests (30 runs per dataset)
│   ├── collect_results.m                     # Collects the main results into one file
│   ├── make_tables_main.m                    # Main result tables (txt / LaTeX)
│   └── make_tables_rel.m                     # Tables derived from saved results (max values, relative changes)
└── results/                                  # Created at run time (results_cfgXXXXXX/ when the settings change)
```

## Datasets (all 10 included)

karate, dolphins, football, polbooks, polblogs, email, citeseer, cora, pubmed, **dblp**

## Settings and datasets

All run settings live in **`src/utils/get_user_settings.m`**: datasets, label fraction, epoch budgets
(per graph size), patience, learning rate, validation fraction, Adam parameters, pseudo-label parameters,
B, lambda weights, margin, weight decay, Node2Vec p/q, per-dataset architecture (layers, batch size,
Node2Vec d/l/r/w) and the sensitivity grids. Every value is written out explicitly in that file.
Seeds are fixed: `usr.numSeeds` runs everywhere, `usr.statSeeds` for the statistical tests.
Missing datasets are downloaded automatically from Zenodo
(https://doi.org/10.5281/zenodo.18775800) before any experiment starts.
If a result-relevant setting is changed after results exist, new results go to `results_cfgXXXXXX/`
(see `settings_used.txt` inside) so runs with different settings never mix.

## Quick start

```matlab
cd PSAE-SSL/experiments

% --- Easiest: run the complete pipeline in order ---
everything            % master script (all experiments sequentially)

% --- Or run individual blocks ---
all_datasets          % main metrics (Tables 9–13 style)
sensitivity_all       % branches + label sensitivity (unified)
scalability_all       % train/infer time + peak memory (all datasets in usr.datasets)
```

## Full experiment order

See `experiments/README_ORDER.txt`.

**One-shot master script:** `everything`

1. `all_datasets` / `nmi_q_log`
2. `ablation_all`
3. `sensitivity_all`          (branches + label)
4. `separability_all`
5. `scalability_all`
6. `collect_results`
7. `statistical_tests`
8. `make_tables_main`
9. `make_tables_rel`

Results are written under `results/`.

## Requirements

- MATLAB R2019b or newer (tested conceptually against R2023b)
- No toolboxes required beyond base MATLAB for the core pipeline

## Epoch budgets

Epoch budgets are set in `usr.epochs` (`src/utils/get_user_settings.m`), one row per graph-size range.
The same values must be used for `all_datasets` and `scalability_all`.

## Scalability

`scalability_all` measures the datasets in `usr.datasets` and writes `results/scalability_all.mat`
(driver: `scalability_core.m`; one fresh MATLAB process per seed, per-seed cache files make an interrupted
run resumable).  The table is printed at the end of `scalability_core`.
Timing hygiene: close other programs, plug in power, disable sleep; re-run outlier seeds by deleting
`results/scal_psaessl_<ds>_s<k>.mat`.

## Statistical tests

`statistical_tests.m`: Friedman for NMI, Q, Precision, Recall, F1 and Wilcoxon on NMI.
Mode A (per dataset over runs) needs `results/baseline_runs.mat`; Mode B (default) uses the baseline means
printed in the manuscript and ranks over the dataset means. The 30 PSAE-SSL runs per dataset are cached in
`results/stat_runs/` (resumable). Tables derived from saved results: `make_tables_rel.m`.
