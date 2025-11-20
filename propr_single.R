options(propr.use_gpu = TRUE)
Sys.setenv(UV_OFFLINE=1)
library(anndata)
library(propr)
library(limma)


target_gene <- Sys.getenv("TARGET_GENE")
if (target_gene == "") {
  stop("TARGET_GENE environment variable not set")
}

adata_path <- "/users/cn/projects/VCC/vcc_data/adata_Training.h5ad"
out_dir    <- "/users/cn/caraiz/propr_new/results"


adata <- read_h5ad(adata_path)

# 2) extract obs labels and full counts matrix
cell_targets <- adata$obs$target_gene
X_all        <- adata$X             # cells x genes
gene_names   <- colnames(X_all)

ctrl_label <- "non-targeting"
i_pert <- which(cell_targets == target_gene)
i_ctrl <- which(cell_targets == ctrl_label)

# debug prints
cat("Processing:", target_gene, "\n")
cat(" # perturbed cells:", length(i_pert), "\n")
cat(" # control   cells:", length(i_ctrl), "\n")

# subset
counts_sub <- as(X_all[c(i_pert, i_ctrl), , drop = FALSE], "matrix")
grp <- factor(
  c(rep(target_gene, length(i_pert)), rep(ctrl_label, length(i_ctrl))),
  levels = c(ctrl_label, target_gene)
)

#pd <- propd(counts_sub, grp, alpha = 0.5) BEFORE
pd <- propd(counts_sub, grp, weighted = FALSE, alpha = 0.5)

cat("Propd object created. Running updateF...\n")
# pd <- propd(counts_sub, grp)
pd <- updateF(pd)
cat("updateF done \n")

cat("Done:", target_gene, "\n")

# # save(pd, file = file.path(out_dir, paste0(target_gene, ".rdata")))
# out_csv <- file.path(out_dir, paste0(target_gene, "_gpu_results.csv.gz"))
# # write.csv(pd@results, out_csv, row.names = FALSE)


# data.table::fwrite(pd@results,
#   file.path(out_dir, paste0(target_gene, "_gpu_results.csv.gz")),
#   compress = "gzip")


out_gz <- file.path(out_dir, paste0(target_gene, "_gpu_results.csv.gz"))
con <- gzfile(out_gz, compression = 6)  # leave it CLOSED
write.csv(pd@results, con, row.names = FALSE)  # opens & closes internally
# no close(con) here


cat("Wrote results to:", out_gz, "\n")