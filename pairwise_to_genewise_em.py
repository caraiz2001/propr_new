"""
This script is used to convert pairwise propd results into genewise DE results.
Exact per-gene mean, median, and weighted connectivity from a huge compressed comma-separated file.

Input columns (comma-separated): partner, pair, theta, ..., fdr
- partner: integer gene index (0..NUMBER_OF_GENES-1)
- pair:    integer gene index (0..NUMBER_OF_GENES-1)
- theta:   float
- fdr:     float

Assumptions:
- Each unordered gene pair appears exactly once. That is (g1, g2) is present and (g2, g1) is not.
- A gene g appears in NUMBER_OF_GENES-1 rows overall.

Method:
1) First streaming pass:
   - Count how many times each gene appears. This gives exact per-gene counts.
   - Accumulate exact weighted connectivity: sum of (1 - theta) for rows with fdr < 0.05, added to both genes.
2) Allocate one big memory-mapped array on disk whose length is the sum of all per-gene counts
   (which should equal two times the number of unordered pairs).
   Build per-gene offsets by prefix sums of counts.
3) Second streaming pass:
   - Write every theta into its gene's contiguous slice in the memory-mapped array.
   - Maintain a per-gene write cursor so appends are O(1).
4) After the second pass:
   - For each gene, read its contiguous slice, compute exact mean and exact median.
   - Save a compressed comma-separated output with gene_index, pair_count, theta_mean, theta_median, weighted_connectivity.

This is exact and uses bounded main memory. Disk usage is dominated by the memory-mapped theta store
(~ total_pairs*2 * 8 bytes).
"""

# ----- Imports -----
import argparse
import os
import numpy as np
import pandas as pd
import sys


# ----- Paths -----
#TARGET_GENE = "MED12"
TARGET_GENE = os.getenv("TARGET_GENE")
if not TARGET_GENE:
    print("TARGET_GENE environment variable not set")
    sys.exit(1)
INPUT_PAIRWISE_PATH = f"/users/cn/caraiz/propr_new/results/results_pairwise/{TARGET_GENE}_gpu_results.csv.gz"
OUTPUT_GENEWISE_PATH = f"/users/cn/caraiz/propr_new/results/propd_single/theta_emergent/genewise_metrics/pert_{TARGET_GENE}_genewise_metrics.csv"       # update this
TMP_MEMMAP_PATH = "/users/cn/caraiz/propr_new/results/tmp/theta_by_gene.float64" # folder must exist
ALL_GENE_NAMES_PATH = "/users/cn/caraiz/propr_new/data/all_gene_names_18080.txt"

NUMBER_OF_GENES = 18080
CHUNK_SIZE_ROWS = 5_000_000  # tune to your server
PER_GENE_PAIR_COUNT = NUMBER_OF_GENES - 1  

FDR_SIGNIFICANCE_THRESHOLD = 2 # just to take all of them, as they refer to theta and not theta_e

# Column subset to read
USE_COLUMNS = ["Partner", "Pair", "theta_e", "FDR"]
DTYPE_MAP = {
    "Partner": np.int32,
    "Pair":    np.int32,
    "theta_e":   np.float64,  # keep as float64 for exact mean/median operations
    "FDR":     np.float32,
}

# -------------------------
# FUNCTIONS
# -------------------------


def first_pass_connectivity_only():
    """
    This function computes the weighted connectivity for each gene in a specific experiment. 
    weighted connectivity is defined as the sum of (1 - theta) for all significant connections (fdr < 0.05)
    """
    per_gene_count = np.zeros(NUMBER_OF_GENES, dtype=np.int64)
    per_gene_weighted_connectivity = np.zeros(NUMBER_OF_GENES, dtype=np.float64)

    # Read the file in chunks (keep only necessary columns)
    reader = pd.read_csv(
        INPUT_PAIRWISE_PATH,
        usecols=USE_COLUMNS,
        dtype=DTYPE_MAP,
        chunksize=CHUNK_SIZE_ROWS,
        compression="gzip",
        low_memory=False,
        engine="c",  # fastest
    )

    for chunk in reader:
        partner = chunk["Partner"].to_numpy(copy=False) - 1
        pair = chunk["Pair"].to_numpy(copy=False) - 1
        theta_e = chunk["theta_e"].to_numpy(copy=False)
        fdr = chunk["FDR"].to_numpy(copy=False)

        # Count how many times each gene appears
        np.add.at(per_gene_count, partner, 1)
        np.add.at(per_gene_count, pair, 1)

        # Weighted connectivity: sum over significant connections of theta_e to both genes
        sig = fdr < FDR_SIGNIFICANCE_THRESHOLD  # creates a boolean mask for significant pairs.
        if np.any(sig):
            w = theta_e[sig] # In this case, theta_e close to 1 is more significant, so we use theta_e directly.
            np.add.at(per_gene_weighted_connectivity, partner[sig], w)
            np.add.at(per_gene_weighted_connectivity, pair[sig], w)

        print(f"Weighted connectivity pass: processed chunk with {len(chunk)} rows", flush=True)
    print("First pass complete")

    # Per-gene counts should all equal PER_GENE_PAIR_COUNT
    if not np.all(per_gene_count == PER_GENE_PAIR_COUNT):
        bad = np.where(per_gene_count != PER_GENE_PAIR_COUNT)[0]
        raise RuntimeError(f"Count mismatch for {bad.size} genes")

    return per_gene_weighted_connectivity


def prepare_memmap_and_offsets():
    """
    This function prepares a memory-mapped array to store theta values for each gene.
    """
    # Each gene owns a fixed length slice of length PER_GENE_PAIR_COUNT
    # Offsets are a simple arithmetic progression
    slice_start = np.arange(NUMBER_OF_GENES, dtype=np.int64) * PER_GENE_PAIR_COUNT # array 18080 x 18079
    total_values = int(NUMBER_OF_GENES) * int(PER_GENE_PAIR_COUNT)

    theta_mem = np.memmap(TMP_MEMMAP_PATH, dtype=np.float64, mode="w+", shape=(total_values,))
    next_write_pos = slice_start.copy()
    return theta_mem, slice_start, next_write_pos, total_values


def second_pass_fill(theta_mem, next_write_pos):
    """
    This function fills the memory-mapped array with theta_e values for each gene.
    The array will be used for extracting per-gene mean and median later
    """
    reader = pd.read_csv(
        INPUT_PAIRWISE_PATH,
        usecols=USE_COLUMNS,
        dtype=DTYPE_MAP,
        chunksize=CHUNK_SIZE_ROWS,
        compression="gzip",
        low_memory=False,
        engine="c",
    )

    for chunk in reader:
        partner = chunk["Partner"].to_numpy(copy=False) - 1  # 1-based -> 0-based
        pair = chunk["Pair"].to_numpy(copy=False) - 1
        theta_e = chunk["theta_e"].to_numpy(copy=False)

        for genes in (partner, pair):
            order = np.argsort(genes, kind="mergesort")
            genes_sorted = genes[order]
            theta_es_sorted = theta_e[order]

            if genes_sorted.size == 0:
                continue

            boundaries = np.flatnonzero(np.diff(genes_sorted)) + 1
            starts = np.concatenate(([0], boundaries))
            stops  = np.concatenate((boundaries, [genes_sorted.size]))
            unique_genes = genes_sorted[starts]

            for g, s, t in zip(unique_genes, starts, stops):
                k = t - s
                pos = next_write_pos[g]
                theta_mem[pos:pos + k] = theta_es_sorted[s:t]
                next_write_pos[g] = pos + k

    theta_mem.flush()
    return next_write_pos  # important

# --- new helper: compute actual counts and validate against expectation ---
def compute_actual_counts(next_write_pos, slice_start):
    actual_counts = (next_write_pos - slice_start).astype(np.int64, copy=False)

    # Optional: warn if anything deviates from the expected fixed count
    expected = PER_GENE_PAIR_COUNT
    bad = np.where(actual_counts != expected)[0]
    if bad.size > 0:
        # Print a compact summary, plus first few offending genes (1-based indices for readability)
        print(f"[WARN] {bad.size} genes have unexpected pair counts (expected {expected}).")
        for g in bad[:10]:
            print(f"       gene {g+1}: got {actual_counts[g]}")
        # We will proceed using actual_counts so stats are correct.

    return actual_counts

# -------------------------
# Finalize exact statistics
# -------------------------
def finalize_stats(weighted_connectivity, slice_start):
    theta_mem = np.memmap(TMP_MEMMAP_PATH, dtype=np.float64, mode="r")

    # Creates vectors to store the results
    theta_e_mean = np.empty(NUMBER_OF_GENES, dtype=np.float64)
    theta_e_median = np.empty(NUMBER_OF_GENES, dtype=np.float64)

    count = PER_GENE_PAIR_COUNT

    for g in range(NUMBER_OF_GENES):
        start = slice_start[g]
        stop = start + count
        vals = np.array(theta_mem[start:stop], copy=True)

        # Exact mean
        theta_e_mean[g] = vals.mean(dtype=np.float64)

        # Exact median
        half = count // 2
        if count % 2 == 1:
            theta_e_median[g] = np.partition(vals, half)[half]
        else:
            left = np.partition(vals, half - 1)[half - 1]
            right = np.partition(vals, half)[half]
            theta_e_median[g] = (left + right) / 2.0

    out = pd.DataFrame(
        {
            "gene_index": np.arange(1, NUMBER_OF_GENES + 1, dtype=np.int32),  # back to 1 based
            "pair_count": np.full(NUMBER_OF_GENES, count, dtype=np.int64),
            "theta_e_mean": theta_e_mean,
            "theta_e_median": theta_e_median,
            "weighted_connectivity_e": weighted_connectivity,
        }
    )
    return out

# --- replace your finalize_stats with this robust version ---
def finalize_stats_robust(weighted_connectivity, slice_start, actual_counts):
    theta_mem = np.memmap(TMP_MEMMAP_PATH, dtype=np.float64, mode="r")

    N = NUMBER_OF_GENES
    theta_e_mean   = np.full(N, np.nan, dtype=np.float64)
    theta_e_median = np.full(N, np.nan, dtype=np.float64)

    for g in range(N):
        cnt = int(actual_counts[g])
        if cnt <= 0:
            continue
        start = int(slice_start[g])
        stop  = start + cnt
        vals = np.array(theta_mem[start:stop], copy=True)

        # exact mean
        theta_e_mean[g] = vals.mean(dtype=np.float64)

        # exact median
        half = cnt // 2
        if cnt % 2 == 1:
            theta_e_median[g] = np.partition(vals, half)[half]
        else:
            left  = np.partition(vals, half - 1)[half - 1]
            right = np.partition(vals, half)[half]
            theta_e_median[g] = (left + right) / 2.0

    # Coerce everything to clean 1D NumPy arrays and validate lengths
    gene_index = np.arange(1, N + 1, dtype=np.int32)
    pair_count = np.asarray(actual_counts, dtype=np.int64).reshape(-1)
    wc         = np.asarray(weighted_connectivity, dtype=np.float64).reshape(-1)
    tmn        = np.asarray(theta_e_mean, dtype=np.float64).reshape(-1)
    tmd        = np.asarray(theta_e_median, dtype=np.float64).reshape(-1)

    lengths = [gene_index.size, pair_count.size, wc.size, tmn.size, tmd.size]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All arrays must match in length. Shapes -> "
            f"gene_index:{gene_index.size}, pair_count:{pair_count.size}, "
            f"weighted_connectivity:{wc.size}, theta_e_mean:{tmn.size}, theta_e_median:{tmd.size}"
        )

    out = pd.DataFrame(
        {
            "gene_index": gene_index,
            "pair_count": pair_count,
            "theta_e_mean": tmn,
            "theta_e_median": tmd,
            "weighted_connectivity_e": wc,
        }
    )
    return out


# -------------------------
# Main
# -------------------------
def main():
    # os.makedirs(os.path.dirname(OUTPUT_GENEWISE_PATH), exist_ok=True)
    # os.makedirs(os.path.dirname(TMP_MEMMAP_PATH), exist_ok=True)

    # # First pass: connectivity only
    # weighted_connectivity = first_pass_connectivity_only()
    # print("Computed weighted connectivity for all genes")

    # # Prepare disk layout
    # theta_mem, slice_start, next_write_pos, total_values = prepare_memmap_and_offsets()
    # print(f"Prepared memory map for {total_values} theta values at {TMP_MEMMAP_PATH}")

    # # Second pass: fill per gene slices with exact theta values
    # second_pass_fill(theta_mem, next_write_pos)

    # # Optional integrity check. Comment out if not needed.
    # # Each cursor should equal slice_start[g] + PER_GENE_PAIR_COUNT
    # if not np.all(next_write_pos == slice_start + PER_GENE_PAIR_COUNT):
    #     bad = np.where(next_write_pos != slice_start + PER_GENE_PAIR_COUNT)[0]
    #     raise RuntimeError(f"Write count mismatch for {bad.size} genes")

    # # Finalize exact mean and median and write result
    # result = finalize_stats(weighted_connectivity, slice_start)

    os.makedirs(os.path.dirname(OUTPUT_GENEWISE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TMP_MEMMAP_PATH), exist_ok=True)

    # 1) First pass: connectivity only
    weighted_connectivity = first_pass_connectivity_only()
    print("Computed weighted connectivity for all genes")
    #print(f"shape of weighted_connectivity: {weighted_connectivity.shape}")

    # 2) Prepare fixed layout
    theta_mem, slice_start, next_write_pos, total_values = prepare_memmap_and_offsets()
    print(f"Prepared memory map for {total_values} theta values at {TMP_MEMMAP_PATH}")

    # 3) Second pass: fill and get final write cursors
    next_write_pos = second_pass_fill(theta_mem, next_write_pos)

    # 4) Derive actual per-gene counts from cursors (robust to any missing/extra rows)
    actual_counts = compute_actual_counts(next_write_pos, slice_start)

    # 5) Finalize exact stats and write
    result = finalize_stats_robust(weighted_connectivity, slice_start, actual_counts)
    # result.to_csv(OUTPUT_GENEWISE_PATH, index=False, compression="gzip")
    # print(f"Wrote per gene metrics to {OUTPUT_GENEWISE_PATH}")

    # Load the gene names file and add a new column to the result DataFrame
    gene_names = pd.read_csv(ALL_GENE_NAMES_PATH, header=None, names=["gene_name"])
    result = pd.concat([gene_names, result], axis=1)
    result.to_csv(OUTPUT_GENEWISE_PATH, index=False, compression="gzip")
    print(f"Wrote per gene metrics to {OUTPUT_GENEWISE_PATH}")


if __name__ == "__main__":
    main()
