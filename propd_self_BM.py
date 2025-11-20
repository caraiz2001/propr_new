"""
This script compares the propd_combined DE results to the propd_single DE results
Significance threshold is set on the average weighted connectivity from propd_single
"""
# ----------------------------
# Import libraries
# ----------------------------

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#TARGET_GENE = "MED12" # for testing only
TARGET_GENE = os.getenv("TARGET_GENE")
if not TARGET_GENE:
    print("TARGET_GENE environment variable not set")
    sys.exit(1)
PROPD_COMBINED_PATH = f"/db/VCC/theta_combined/theta_summed_{TARGET_GENE}.csv.gz"
#WILCOXON_PATH = f"/users/cn/projects/VCC/de_results_per_gene/{TARGET_GENE}_de_genes.tsv"
PROPD_SINGLE_PATH = f"/users/cn/caraiz/propr_new/results/propd_single/genewise_metrics/pert_{TARGET_GENE}_genewise_metrics.csv"
# PROPD_PAIRWISE_PATH = f"/users/cn/caraiz/propr_new/results/{TARGET_GENE}_gpu_results.csv.gz"
PROPD_COMBINED_MATRIX_PATH = f"/db/VCC/theta_combined/theta_{TARGET_GENE}.csv.gz"
OUTPUT_DIR = f"/users/cn/caraiz/propr_new/results/propd_benchmark/{TARGET_GENE}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Functions
# ----------------------------
import numpy as np
import pandas as pd

def compute_theta_enrichment(theta_df: pd.DataFrame, ascending: bool = True):
    """
    Compute per-gene enrichment scores over a flattened, ranked list of all theta values.

    Inputs
    ------
    theta_df : pandas.DataFrame
        Rows are experiments (149). Columns are genes (~18,080). Values are theta.
    ascending : bool
        If True, lower theta values are treated as more significant and placed first.

    Method
    ------
    1) Flatten the entire matrix to a single vector and rank all entries.
    2) For each gene, treat the 149 entries belonging to that gene as the "hit" set.
       All other entries are "misses".
    3) Perform a single streaming pass over the ranked list and update a running sum:
         - miss step: constant decrement = 1 / (N - k)
         - hit step:  constant increment  = 1 / k
       Track per-gene maxima and minima of the running sum to obtain the final score:
         - es_pos = maximum running sum
         - es_neg = minimum running sum
         - es     = es_pos if |es_pos| >= |es_neg| else es_neg

    Outputs
    -------
    pandas.DataFrame with index = gene names and columns:
        es, es_pos, es_neg, direction
        where 'direction' is 'top', 'bottom', or 'neutral'.
    """
    # Shapes
    num_experiments, num_genes = theta_df.shape
    k_hits_per_gene = num_experiments
    total_entries = num_experiments * num_genes

    # Flatten all theta values
    values = theta_df.to_numpy().ravel(order="C")  # row-major
    # Map each flattened entry to its gene column index
    # Row-major ravel means pattern per row: [g0, g1, ..., g_{G-1}], repeated for each experiment
    gene_of_entry = np.tile(np.arange(num_genes, dtype=np.int64), num_experiments)

    # Rank indices (ascending means lower theta first)
    sort_idx = np.argsort(values, kind="mergesort")  # stable sort helps reproducibility
    if not ascending:
        sort_idx = sort_idx[::-1]

    # Precompute constants
    miss_step = 1.0 / (total_entries - k_hits_per_gene)
    hit_step = 1.0 / k_hits_per_gene

    # Per-gene state (int64 for positions, float64 for running sums)
    last_seen_position = np.zeros(num_genes, dtype=np.int64)   # last ranked position we updated for this gene
    running_sum = np.zeros(num_genes, dtype=np.float64)
    max_running = np.zeros(num_genes, dtype=np.float64)
    min_running = np.zeros(num_genes, dtype=np.float64)

    # Streaming pass over the ranked list
    # Ranked positions are 1..N for clarity with the delta formula below
    i = 0
    for flat_idx in sort_idx:
        i += 1
        g = gene_of_entry[flat_idx]

        # Apply accumulated misses since the last time we touched this gene
        delta_misses = (i - last_seen_position[g] - 1)
        if delta_misses:
            dec = delta_misses * miss_step
            val = running_sum[g] - dec
            running_sum[g] = val
            if val < min_running[g]:
                min_running[g] = val

        # Apply this hit
        val = running_sum[g] + hit_step
        running_sum[g] = val
        if val > max_running[g]:
            max_running[g] = val

        # Update last seen position for this gene
        last_seen_position[g] = i

    # Tail drift for each gene from its last hit to the end of the ranked list
    tail_len = total_entries - last_seen_position
    if np.any(tail_len):
        tail_dec = tail_len * miss_step
        running_sum -= tail_dec
        # Update minima where tail pushes below existing minimum
        min_running = np.minimum(min_running, running_sum)

    # Choose direction with larger magnitude
    es_pos = max_running
    es_neg = min_running  # negative or zero
    abs_pos = np.abs(es_pos)
    abs_neg = np.abs(es_neg)
    choose_pos = abs_pos >= abs_neg
    es = np.where(choose_pos, es_pos, es_neg)
    es_abs = np.abs(es)

    # Direction labels
    # positive -> enriched at top of ranked list
    # negative -> enriched at bottom of ranked list
    # near zero -> neutral
    # Threshold can be adjusted; here we call exactly zero neutral.
    direction = np.where(es > 0, "top", np.where(es < 0, "bottom", "neutral"))

    # Package result as a DataFrame indexed by gene names
    result = pd.DataFrame(
        {
            "es": es,
            "es_pos": es_pos,
            "es_neg": es_neg,
            "direction": direction,
            "es_abs": es_abs,
        },
        index=theta_df.columns,
    ).sort_values("es", ascending=False)

    return result



# ----------------------------
# Main code
# ----------------------------

# run with main()
def main():
    propd_comb_de = pd.read_csv(PROPD_COMBINED_PATH, compression="gzip")
    #wilcoxon_de = pd.read_csv(WILCOXON_PATH, sep="\t")
    propd_single_de = pd.read_csv(PROPD_SINGLE_PATH, compression="gzip")
    propd_comb_matrix = pd.read_csv(PROPD_COMBINED_MATRIX_PATH, index_col=0, compression="gzip")



    # Print basic info
    print(f"propd_combined: {propd_comb_de.shape[0]} genes")
    #print(f"wilcoxon: {wilcoxon_de.shape[0]} genes")
    print(f"propd_single: {propd_single_de.shape[0]} genes")
    print(f"propd_combined_matrix: {propd_comb_matrix.shape[0]} experiments, {propd_comb_matrix.shape[1]} genes")
    
    

    # Print columns info
    print("propd_combined columns:", propd_comb_de.columns.tolist())
    #print("wilcoxon columns:", wilcoxon_de.columns.tolist())
    print("propd_single columns:", propd_single_de.columns.tolist())

    es_df = compute_theta_enrichment(propd_comb_matrix, ascending=True)
    #es_df.to_csv(os.path.join(OUTPUT_DIR, f"propd_combined_enrichment_{TARGET_GENE}.csv"))
    print(f"enrichment df: {es_df.shape[0]} genes")
    # Set [gene_name, theta_combined] as column names for propd_comb_de (right now are unnamed)
    propd_comb_de = propd_comb_de.rename(columns={propd_comb_de.columns[0]: "gene_name", propd_comb_de.columns[1]: "theta_combined"})

    # Sort propd_comb_de by theta_combined descending (high to low)
    propd_comb_de = propd_comb_de.sort_values(by="theta_combined", ascending=False).reset_index(drop=True)
    # Add a column named 'ranked_theta_combined' with the rank (1-based)
    propd_comb_de["ranked_theta_combined"] = np.arange(1, propd_comb_de.shape[0] + 1)

    # # Filter wilcoxon_de to keep rows with significant fdr (< 0.05). Add a column named 'significant' with True/False
    # wilcoxon_de["significant"] = wilcoxon_de["fdr"] < 0.05
    # wilcoxon_de_sig = wilcoxon_de[wilcoxon_de["significant"]].copy()
    # print(f"wilcoxon: {wilcoxon_de_sig.shape[0]} significant genes (fdr < 0.05)")

    # Compute average weighted connectivity in propd_single_de
    avg_weighted_connectivity = propd_single_de["weighted_connectivity"].mean()
    print(f"propd_single: average weighted connectivity = {avg_weighted_connectivity:.4f}")
    # Filter propd_single_de to keep rows with weighted_connectivity > average. Add a column named 'significant' with True/False)
    propd_single_de["significant"] = propd_single_de["weighted_connectivity"] > avg_weighted_connectivity
    propd_single_de_sig = propd_single_de[propd_single_de["significant"]].copy()
    num_sig = propd_single_de_sig.shape[0]
    print(f"propd_single: {propd_single_de_sig.shape[0]} significant genes (weighted_connectivity > average)")

    # Merge propd_comb_de and wilcoxon_de_sig on gene_name (keeping order of propd_comb_de)
    merged_comb_single = pd.merge(propd_comb_de, propd_single_de[["gene_name", "weighted_connectivity","significant"]], left_on="gene_name", right_on="gene_name", how="inner")

    # Add a column with the cumulative sum of significant genes (True) in wilcoxon_de_sig
    merged_comb_single["cum_significant_theta_comb"] = merged_comb_single["significant"].cumsum()

    # # Rank propd_single_de by weighted_connectivity descending (high to low)
    # propd_single_de = propd_single_de.sort_values(by="weighted_connectivity", ascending=False).reset_index(drop=True)
    # # Add a column named 'ranked_weighted_connectivity' with the rank (1-based)
    # propd_single_de["ranked_weighted_connectivity"] = np.arange(1, propd_single_de.shape[0] + 1)

    # # Merge merged_comb_single with propd_single_de on gene_name (left join)
    # merged_all = pd.merge(merged_comb_single, propd_single_de[["gene_name", "weighted_connectivity", "ranked_weighted_connectivity"]], on="gene_name", how="left")
    # # Rank merged_all by ranked_weighted_connectivity (ascending)
    # merged_all = merged_all.sort_values(by="ranked_weighted_connectivity", ascending=True).reset_index(drop=True)
    # # Add a column named 'cum_significant_weighted_connectivity' with the cumulative sum of significant genes (True) in wilcoxon_de_sig
    # merged_all["cum_significant_weighted_connectivity"] = merged_all["significant"].cumsum()


    # Set es_df index as gene_name
    es_df = es_df.reset_index().rename(columns={"index": "gene_name"})
    # Add a column named 'es_abs_rank' with the rank (1-based) of es_abs (descending)
    es_df = es_df.sort_values(by="es_abs", ascending=False).reset_index(drop=True)
    es_df["es_abs_rank"] = np.arange(1, es_df.shape[0] + 1)

    merged_all = merged_comb_single.copy()
    
    # Merge merged_all with es_df on gene_name (left join)
    merged_all = pd.merge(merged_all, es_df[["gene_name", "es", "es_abs", "es_pos", "es_neg", "direction", "es_abs_rank"]], on="gene_name", how="left")
    # Sort merged_all by es_abs_rank (ascending)
    merged_all = merged_all.sort_values(by="es_abs_rank", ascending=True).reset_index(drop=True)
    # Add a column named 'cum_significant_theta_es' with the cumulative sum of significant genes (True) in wilcoxon_de_sig
    merged_all["cum_significant_theta_es_abs"] = merged_all["significant"].cumsum()

    # Take the significant column and shuffle it randomly to create a random baseline (significant_random)
    np.random.seed(42)
    merged_all["significant_random"] = np.random.permutation(merged_all["significant"].values)
    # Add a column named 'cum_significant_random' with the cumulative sum of significant_random
    merged_all["cum_significant_random"] = merged_all["significant_random"].cumsum()
    # add a ranked_random column with the rank (1-based) of the index
    merged_all["ranked_random"] = np.arange(1, merged_all.shape[0] + 1)    


    # Plot the cumulative sum of significant genes (y) vs the rank for both propd_combined and propd_single in different colors

    plt.figure(figsize=(10, 6))
    # Rank by theta_combined
    merged_all = merged_all.sort_values(by="ranked_theta_combined", ascending=True).reset_index(drop=True)
    print("Ranked by theta combined:")
    print(merged_all[["theta_combined", "ranked_theta_combined", "cum_significant_theta_comb"]].head(10))
    plt.plot(merged_all["ranked_theta_combined"], merged_all["cum_significant_theta_comb"], label="propd_combined", color="blue")
    
    # merged_all = merged_all.sort_values(by="ranked_weighted_connectivity", ascending=True).reset_index(drop=True)
    # print("Ranked by weighted connectivity:")
    # print(merged_all[["weighted_connectivity","ranked_weighted_connectivity", "cum_significant_weighted_connectivity"]].head(10))
    # plt.plot(merged_all["ranked_weighted_connectivity"], merged_all["cum_significant_weighted_connectivity"], label="propd_single", color="orange")
    
    merged_all = merged_all.sort_values(by="es_abs_rank", ascending=True).reset_index(drop=True)
    print("Ranked by enrichment score:")
    print(merged_all[["es_abs","es_abs_rank", "cum_significant_theta_es_abs"]].head(10))
    plt.plot(merged_all["es_abs_rank"], merged_all["cum_significant_theta_es_abs"], label="propd_combined_enrichment", color="green")

    merged_all = merged_all.sort_values(by="ranked_random", ascending=True).reset_index(drop=True)
    print("Ranked by random:")
    print(merged_all[["ranked_random", "cum_significant_random"]].head(10))
    plt.plot(merged_all["ranked_random"], merged_all["cum_significant_random"], label="random", color="red", linestyle="--")
    plt.xlabel("Rank")
    plt.ylabel(f"Cumulative sum of propd_single significant genes (wc < {avg_weighted_connectivity:.4f})")
    plt.title(f"Cumulative sum of propd_single significant genes ({num_sig})vs Rank for {TARGET_GENE}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, f"GTPD_cumulative_significant_genes_{TARGET_GENE}.png"))
    plt.close()

    # Save merged_all to csv
    merged_all.to_csv(os.path.join(OUTPUT_DIR, f"merged_all_{TARGET_GENE}_GTpropdsingle.csv"), index=False)

    # # Plot scatter plot of ranked_theta_combined (x) vs ranked_weighted_connectivity (y)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(merged_all["ranked_theta_combined"], merged_all["ranked_weighted_connectivity"], alpha=0.5)
    # plt.xlabel("Ranked Theta Combined (propd_combined)")
    # plt.ylabel("Ranked Weighted Connectivity (propd_single)")
    # plt.title(f"Ranked Theta Combined vs Ranked Weighted Connectivity for {TARGET_GENE}")
    # plt.grid()
    # plt.savefig(os.path.join(OUTPUT_DIR, f"GTPD_ranked_theta_vs_weighted_connectivity_{TARGET_GENE}.png"))
    # plt.close()


    # # Plot scatter plot of ranked_theta_combined (x) vs ranked_weighted_connectivity (y)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(merged_all["ranked_theta_combined"], merged_all["es_abs_rank"], alpha=0.5)
    # plt.xlabel("Ranked Theta Combined (propd_combined)")
    # plt.ylabel("Ranked Theta Combined (enrichment)")
    # plt.title(f"Ranked Theta Combined (sum 1-theta) vs Ranked Theta Combined (enrichment) for {TARGET_GENE}")
    # plt.grid()
    # plt.savefig(os.path.join(OUTPUT_DIR, f"ranked_theta_vs_weighted_connectivity_{TARGET_GENE}.png"))
    # plt.close()


    # # Plot histogram of es
    # plt.figure(figsize=(10, 6))
    # plt.hist(merged_all["es"], bins=50, color="green", alpha=0.7)
    # plt.xlabel("Enrichment Score")
    # plt.ylabel("Frequency")
    # plt.title(f"Histogram of Enrichment Scores for {TARGET_GENE}")
    # plt.grid()
    # plt.savefig(os.path.join(OUTPUT_DIR, f"histogram_enrichment_scores_{TARGET_GENE}.png"))
    # plt.close()

    # # Plot histogram of theta_combined
    # plt.figure(figsize=(10, 6))
    # plt.hist(merged_all["theta_combined"], bins=50, color="blue", alpha=0.7)
    # plt.xlabel("Theta Combined")
    # plt.ylabel("Frequency")
    # plt.title(f"Histogram of Theta Combined for {TARGET_GENE}")
    # plt.grid()
    # plt.savefig(os.path.join(OUTPUT_DIR, f"histogram_theta_combined_{TARGET_GENE}.png"))


    # # Plot of the theta_combined values (y) vs the rank (x)
    # plt.figure(figsize=(10, 6))
    # plt.plot(merged_all["ranked_weighted_connectivity"], merged_all["theta_combined"], marker="o", linestyle="", alpha=0.5)
    # plt.xlabel("Rank")
    # plt.ylabel("Theta Combined")
    # plt.title(f"Theta Combined values vs Rank of propd_single for {TARGET_GENE}")
    # plt.grid()
    # plt.savefig(os.path.join(OUTPUT_DIR, f"theta_combined_vs_rank_propd_single_{TARGET_GENE}.png"))
    # plt.close()

    # Plot of the theta_combined values (y) vs the rank (x) in red for significant genes and blue for non-significant genes
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_all[~merged_all["significant"]]["ranked_theta_combined"], merged_all[~merged_all["significant"]]["theta_combined"], color="blue", label="Non-significant", alpha=0.3, s=1)
    plt.scatter(merged_all[merged_all["significant"]]["ranked_theta_combined"], merged_all[merged_all["significant"]]["theta_combined"], color="red", label="Significant (wilcoxon fdr < 0.05)", alpha=0.7, s=1)
    plt.xlabel("Rank")
    plt.ylabel("Theta Combined")
    plt.title(f"Theta Combined values vs Rank of propd_single for {TARGET_GENE}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, f"GTPD_theta_combined_vs_rank_theta_comb_colored_{TARGET_GENE}.png"))

if __name__ == "__main__":
    main()
