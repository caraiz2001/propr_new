"""
This script implements a GSEA-like algorithm for gene pairs.
For simplicity, we will not consider weights (just the rank of the pairs).
Input is a csv file with propd results containing the following columns:
- Partner: Index of the partner gene (partner > pair)
- Pair: Index of the pair gene (pair < partner)
- theta: Differential proportionality metric. Between 0 and 1. 
- Other columns are ignored. 

The idea is to compute a genewise 'differential expression - like' score 
based on the pairs that each gene is involved in.
"""


# ----------------------------
# Import libraries
# ----------------------------
import csv
import math
import os
import gzip
import subprocess
import sys
#import tempfile
#from collections import defaultdict
from pathlib import Path
import numpy as np


# ----------------------------
# PATHS
# ----------------------------
#TARGET_GENE = "KLF10" # for testing only
TARGET_GENE = os.getenv("TARGET_GENE")
if not TARGET_GENE:
    print("TARGET_GENE environment variable not set")
    sys.exit(1)

PARENT_DIR = "/users/cn/caraiz/propr_new/"
INPUT_PAIRWISE_PATH = f"{PARENT_DIR}results/{TARGET_GENE}_gpu_results.csv.gz"
OUTPUT_GSEA_PATH = f"{PARENT_DIR}results/gsea_pair2gene/{TARGET_GENE}_gsea_results.csv"
OUTPUT_SORTED_PATH = f"{PARENT_DIR}results/tmp/{TARGET_GENE}_sorted.csv"
TMP_DIR = f"{PARENT_DIR}results/tmp/"
os.makedirs(TMP_DIR, exist_ok=True)



# ----------------------------
# Functions
# ----------------------------

# Sorting function for big propd csv files
def sort_csv_by_score(input_csv: Path, output_sorted_csv: Path):
    """
    Produces a sorted CSV/TSV with the SAME columns ("Partner","Pair","theta","FDR"),
    ordered by the theta column (3rd column), ascending (close to 0 first).
    Detects delimiter (comma or tab) automatically.
    Handles gzip-compressed input files.
    Requires coreutils: tail, head, sort, zcat.
    """
    # # Detect delimiter from header (read first line using gzip)
    # with gzip.open(input_csv, "rt") as f:
    #     header = f.readline()
    #     delimiter = "\t" if "\t" in header else ","
    #     sort_flag = "-t$'\\t'" if delimiter == "\t" else "-t,"

    sort_flag = "-t,"  # assume comma for simplicity

    #tmpdir = tempfile.mkdtemp(prefix="pairgsea_")
    # Intermediate sorted file
    # tmp_nohdr = Path(TMP_DIR) / "nohdr.txt" # File without header
    #tmp_sorted = Path(TMP_DIR) / "sorted.txt" # File with sorted results

    # Sort the file directly, skipping the header (first line) using process substitution
    # This avoids writing a new file for the body
    sort_key = "-k3,3g"
    sort_cmd = f"zcat {input_csv} | sort {sort_flag} {sort_key}"

    # Write the sorted file
    with open(output_sorted_csv, "wb") as out:
        subprocess.run(sort_cmd, shell=True, stdout=out, check=True)

    # # Reattach header (extract header using zcat and head)
    # with open(output_sorted_csv, "wb") as out:
    #     subprocess.run(f"zcat {input_csv} | head -n 1", shell=True, stdout=out, check=True)
    #     subprocess.run(["cat", str(tmp_sorted)], stdout=out, check=True)

# Function to compute enrichment score for each gene
def compute_es_stream(sorted_csv: Path,
                      gene_id: dict, N: int):
    """
    Single streaming pass over the sorted file (rank order).
    Updates each gene's running-sum:
      - misses: constant decrement = 1/(N - k_g)
      - hits:   +1/k_g (unweighted) OR +w_i / sum_w_g (weighted)
    Tracks max and min to get ES+ and ES-.

    Inputs:
        - sorted_csv: path to sorted CSV file
        - gene_id: mapping gene -> integer id (0..G-1)
        - N: total number of pairs (rows)
    """
    G = len(gene_id) # G is the number of genes
    K = G - 1 # K is the 'set size', which equals the number of pairs per gene = G - 1

    if N == (G * (G - 1)) / 2:
        print(f"Input file looks correct: {N} pairs for {G} genes.")
    else:
        raise RuntimeError(f"There are {N} pairs, but expected {G * (G - 1) / 2} for {G} genes.")

    # Precompute increments for misses and hits.
    # As the gene set size is constant, the miss is constant too
    miss_inc = 1.0 / (N - K) # Pmiss = 1 / (total pairs - pairs with gene g) = 1 / [(G*G - G) - (G-1)]
    hit_const = 1.0 / K # Phit = 1 / (pairs with gene g) = 1 / (G-1)

    # States
    last_idx = np.zeros(G, dtype=np.int64)  # last position seen; start at 0
    cur = np.zeros(G, dtype=np.float64) # current running sum
    maxv = np.zeros(G, dtype=np.float64) # max ES value
    minv = np.zeros(G, dtype=np.float64) # min ES value (negative)

    # Detect delimiter from header
    with open(sorted_csv, newline="") as f:
        header = f.readline()
        delimiter = "\t" if "\t" in header else ","
        f.seek(0)
        rdr = csv.DictReader(f, delimiter=delimiter)
        i = 0
        for row in rdr:
            i += 1
            g1 = int(row["Partner"].strip())
            g2 = int(row["Pair"].strip())
            # genes must exist from prepass
            id1 = gene_id[g1]
            id2 = gene_id[g2]

            # Unweighted fast path
            dmiss = (i - last_idx[id1] - 1) * miss_inc  # miss_inc is scalar
            if dmiss:
                cur[id1] -= dmiss
                if cur[id1] < minv[id1]: minv[id1] = cur[id1]
            cur[id1] += hit_const
            if cur[id1] > maxv[id1]: maxv[id1] = cur[id1]
            last_idx[id1] = i
            # gene 2
            dmiss = (i - last_idx[id2] - 1) * miss_inc
            if dmiss:
                cur[id2] -= dmiss
                if cur[id2] < minv[id2]: minv[id2] = cur[id2]
            cur[id2] += hit_const
            if cur[id2] > maxv[id2]: maxv[id2] = cur[id2]
            last_idx[id2] = i

    # Tail drift (after last hit to end of list)
    tail = N - last_idx
    cur -= tail * miss_inc
    # update minima where tail pushes below existing min
    minv = np.minimum(minv, cur)

    # ES selection (direction with larger |ES|)
    pos = maxv
    neg = minv  # negative numbers (or zero)
    es = np.where(np.abs(pos) >= np.abs(neg), pos, neg)
    return es, pos, neg


def write_results(out_csv: Path, gene_id, es, pos, neg, pvals, qvals):
    """
    Write out results to CSV. The output file has the following columns:
    gene, ES, ES_pos, ES_neg, set_size, pval, qval
    """
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "ES", "ES_pos", "ES_neg", "set_size", "pval", "qval"])
        for i, g in enumerate(gene_id):
            w.writerow([g, f"{es[i]:.8g}", f"{pos[i]:.8g}", f"{neg[i]:.8g}",
                        f"{pvals[i]:.6g}", f"{qvals[i]:.6g}"])
def prepass_counts(input_csv: Path):
    """
    Reads CSV once (unsorted), builds:
      - gene -> id mapping (Partner and Pair are numeric IDs)
      - k[g]: number of pairs containing gene g (always G-1)
      - N: total pairs (rows)
    Requires columns "Partner","Pair","theta" (header present).
    """
    gene_ids = set()
    N = 0

    # Detect delimiter from header
    open_func = gzip.open if str(input_csv).endswith(".gz") else open
    with open_func(input_csv, "rt", newline="") as f:
        header = f.readline()
        delimiter = "\t" if "\t" in header else ","
        #delimiter = ","  # assume comma for simplicity
        f.seek(0)
        rdr = csv.DictReader(f, delimiter=delimiter)
        for row in rdr:
            g1 = int(row["Partner"].strip())
            g2 = int(row["Pair"].strip())
            gene_ids.add(g1)
            gene_ids.add(g2)
            N += 1

    if N == 0:
        raise RuntimeError("Input appears empty.")

    gene_ids = sorted(gene_ids)
    gene_id = {g: i for i, g in enumerate(gene_ids)}
    # id_to_gene = gene_ids  # list of integer gene indexes
    G = len(gene_ids)

    if N != (G * (G - 1)) / 2:
        raise RuntimeError(f"There are {N} pairs, but expected {G * (G - 1) / 2} for {G} genes.")

    return gene_id, N

def ks_one_sample_pvalue_asymptotic(D: float, n: int) -> float:
    """
    Asymptotic p-value for one-sample KS test:
    P(D_n >= D) ≈ 2 * sum_{j=1..∞} (-1)^{j-1} exp(-2 j^2 λ^2),
    with λ = D * (sqrt(n) + 0.12 + 0.11 / sqrt(n)).
    Good when n is reasonably large (true here).
    """
    if n <= 0 or D <= 0:
        return 1.0
    sqrtn = math.sqrt(n)
    lam = (sqrtn + 0.12 + 0.11 / sqrtn) * D
    # sum until terms are tiny
    total = 0.0
    j = 1
    while True:
        term = math.exp(-2.0 * (j * j) * (lam * lam))
        add = (1 if j % 2 == 1 else -1) * term
        total += add
        if term < 1e-12:  # tighten if you like
            break
        j += 1
        if j > 1000:  # absolute cap
            break
    p = 2.0 * total
    # numerical guard
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return p

# Benjamini-Hochberg FDR for multiple testing
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR (q-values). Vectorized; stable for ties.
    """
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity
    for i in range(m - 2, -1, -1):
        if q[i] > q[i + 1]:
            q[i] = q[i + 1]
    out = np.empty_like(q)
    out[order] = q
    return out

def main():
    """
    Main function to run the GSEA-like analysis on gene pairs.
    """
    # 1) Pre-pass on unsorted file (fast, streaming)
    print("Pre-pass: counting per-gene set sizes...", file=sys.stderr)
    gene_id, N = prepass_counts(INPUT_PAIRWISE_PATH)

    G = len(gene_id)
    print(f"Found {G} genes; {N} pairs (rows).", file=sys.stderr)

    sort_csv_by_score(INPUT_PAIRWISE_PATH, OUTPUT_SORTED_PATH)
    print("Sorting done.", file=sys.stderr)

    print("Computing enrichment scores (streaming pass)...", file=sys.stderr)
    es, pos, neg = compute_es_stream(OUTPUT_SORTED_PATH, gene_id, N)

    # 4) p-values (one-sample KS) and BH-FDR
    print("Computing p-values and FDR...", file=sys.stderr)
    # KS uses D = |ES| and sample size = k (hits for that gene)
    D = np.abs(es)
    K = G - 1  # set size (pairs per gene)
    pvals = np.array([ks_one_sample_pvalue_asymptotic(D[i], K) for i in range(G)], dtype=np.float64)
    qvals = bh_fdr(pvals)

    # 5) Write results
    write_results(OUTPUT_GSEA_PATH, gene_id, es, pos, neg, pvals, qvals)
    print(f"Done. Results -> {OUTPUT_GSEA_PATH}", file=sys.stderr)

if __name__ == "__main__":
    main()
