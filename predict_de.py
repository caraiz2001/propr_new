"""
We want to select a high confidence set of DE genes based on propd_combined results
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET_GENE = "WSB2"  # for testing only
ALL_METRICS_PATH = f"/users/cn/caraiz/propr_new/results/propd_benchmark/{TARGET_GENE}/merged_all_{TARGET_GENE}.csv"
ALL_PERT_PATH = f"/users/cn/caraiz/propr_new/genes.txt"

merged_all = pd.read_csv(ALL_METRICS_PATH)
all_gene_names = pd.read_csv(ALL_PERT_PATH, header=None)[0].tolist()
# Sort by es_abs_rank
merged_all = merged_all.sort_values(by='es_abs_rank', ascending=False).reset_index(drop=True)

diff_curve = np.cumsum(merged_all['ranked_theta_combined'] - merged_all['es_abs_rank'])
cutoff_rank = np.argmax(np.abs(diff_curve))


# High-confidence set = genes before cutoff
high_conf_genes = merged_all.iloc[:cutoff_rank]

# Save high-confidence genes
# output_path = f"/users/cn/caraiz/propr_new/results/propd_benchmark/{TARGET_GENE}/high_conf_genes_{TARGET_GENE}.csv"
# high_conf_genes.to_csv(output_path, index=False)

# ------ Inspect agreement between rankings ----------------------------
scaler = MinMaxScaler()
scores = pd.DataFrame({
    "enrichment": merged_all['es_abs'],   # per-gene ES values
    "naive": merged_all['theta_combined'],              # per-gene sum(1 - θ)
    "gene_name": merged_all['gene_name']
})

# Set gene_name as index
scores = scores.set_index('gene_name')

scores_scaled = pd.DataFrame(
    scaler.fit_transform(scores),
    columns=scores.columns,
    index=scores.index
)

quantile = 0.1  # top 10%
mask = (
    (scores_scaled["enrichment"] >= scores_scaled["enrichment"].quantile(1 - quantile)) &
    (scores_scaled["naive"] >= scores_scaled["naive"].quantile(1 - quantile))
)
high_confidence = scores_scaled[mask]

print("There are ", len(high_confidence), "high-confidence genes in the top", int(quantile*100), "% of both rankings.")


# ----- which experiment has the highest overlap with this set? -----------------
# For each gene, load the corresponding DE data
de_data = {}
for gene in all_gene_names:
    try:
        de_data[gene] = pd.read_csv(f'/users/cn/projects/VCC/de_results_per_gene/{gene}_de_genes.tsv', sep='\t')
    except FileNotFoundError:
        print(f'File for gene {gene} not found.')


de_gene_lists = {}
for gene, df in de_data.items():
    print(f"Processing gene: {gene}")
    significant_de = df[df['fdr'] < 0.05]
    total_de = len(significant_de)
    print(f"Gene: {gene}, Total DE genes with FDR < 0.05: {total_de}")
    if total_de > 0:
        de_gene_lists[gene] = significant_de['feature'].tolist()
    else:
        de_gene_lists[gene] = []

# Now compute overlap with high-confidence genes
high_conf_gene_set = set(high_confidence.index)
overlap_counts = {}
for gene, de_genes in de_gene_lists.items():
    overlap = high_conf_gene_set.intersection(de_genes)
    overlap_counts[gene] = len(overlap)/len(de_gene_lists[gene])
# Sort by overlap count
sorted_overlap = sorted(overlap_counts.items(), key=lambda x: x[1], reverse=True)

print("Top genes with highest overlap with high-confidence set:")
for gene, count in sorted_overlap[:5]:
    real_overlap = len(set(de_gene_lists[gene]).intersection(set(de_gene_lists[TARGET_GENE])))/len(de_gene_lists[gene])
    print(f"Gene: {gene}, predicted overlap count: {count}, real overlap: {real_overlap}")

print("Bottom genes with lowest overlap with high-confidence set:")
for gene, count in sorted_overlap[-5:]:
    real_overlap = len(set(de_gene_lists[gene]).intersection(set(de_gene_lists[TARGET_GENE])))/len(de_gene_lists[gene])
    print(f"Gene: {gene}, predicted overlap count: {count}, real overlap: {real_overlap}")

# Save results of highest overlap pairs
highest_overlap_pairs = sorted_overlap[:5]
lowest_overlap_pairs = sorted_overlap[-5:]
output_path = f"/users/cn/caraiz/propr_new/results/propd_benchmark/{TARGET_GENE}/top_bottom_5_overlap_pairs_{TARGET_GENE}.csv"

# Save to csv
pd.DataFrame(highest_overlap_pairs + lowest_overlap_pairs, columns=['gene', 'predicted_overlap']).to_csv(output_path, index=False)



# ------ Divergence detection function --------------------------------------------

# def early_divergence_cutoff(naive_rank: pd.Series,
#                             enrich_rank: pd.Series,
#                             baseline_frac: float = 0.10,
#                             k_sigma: float = 3.0,
#                             persistence: int = 50) -> int | None:
#     """
#     Find the first rank where the cumulative difference between methods
#     exceeds a growing threshold k * sigma * sqrt(i) and persists.

#     Parameters
#     ----------
#     naive_rank : pd.Series
#         Rank positions from the naive score (aligned to the same ordering as enrich_rank).
#     enrich_rank : pd.Series
#         Rank positions from enrichment (this should be your reference order).
#     baseline_frac : float
#         Fraction of early steps used to estimate noise (e.g., 0.10 = first 10%).
#     k_sigma : float
#         Sensitivity (3.0 is conservative; lower to detect earlier).
#     persistence : int
#         Require the condition to hold for this many consecutive steps to avoid flicker.

#     Returns
#     -------
#     cutoff_idx : int or None
#         Zero-based index where divergence begins. None if no stable crossing.
#     """
#     assert len(naive_rank) == len(enrich_rank)
#     N = len(enrich_rank)

#     # Step series and cumulative difference
#     delta = (naive_rank.to_numpy() - enrich_rank.to_numpy()).astype(np.float64)
#     diff_curve = np.cumsum(delta)

#     # Noise estimation from the early segment of the *steps*, not the cumulative
#     w = max(50, int(baseline_frac * N))
#     sigma = np.std(delta[:w], ddof=1)
#     if sigma == 0:
#         # If there is zero initial variance, fall back to absolute threshold on cum curve
#         sigma = 1.0

#     # Time-varying threshold for a random-walk: k * sigma * sqrt(i)
#     i = np.arange(1, N + 1, dtype=np.float64)
#     thresh = k_sigma * sigma * np.sqrt(i)

#     # First crossing that PERSISTS for `persistence` steps
#     crossed = np.abs(diff_curve) > thresh
#     if persistence > 1:
#         # rolling "all true" over a window
#         win = np.ones(persistence, dtype=int)
#         stable = np.convolve(crossed.astype(int), win, mode="same") >= persistence
#     else:
#         stable = crossed

#     idx = np.argmax(stable) if stable.any() else None
#     # np.argmax returns 0 even if all False; guard it:
#     if idx == 0 and not stable[0]:
#         idx = None
#     return idx

# # --- Your case ---------------------------------------------------------------
# # You already did:
# # merged_all = merged_all.sort_values(by='es_abs_rank', ascending=False).reset_index(drop=True)
# # So enrich_rank is in reference order. Now run:
# cutoff_idx = early_divergence_cutoff(
#     naive_rank=merged_all['ranked_theta_combined'],
#     enrich_rank=merged_all['es_abs_rank'],
#     baseline_frac=0.20,   # use first 20% to estimate noise
#     k_sigma=4.0,          # 4-sigma; use 2.0 for earlier detection
#     persistence=1000        # require ~1000 consecutive steps beyond threshold
# )

# print("early divergence at index:", cutoff_idx)

# # High-confidence genes = before cutoff (if any)
# if cutoff_idx is not None:
#     high_conf = merged_all.iloc[:cutoff_idx]
# else:
#     high_conf = merged_all  # no stable divergence detected

