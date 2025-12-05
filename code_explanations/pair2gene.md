# Debug notes: Explanation of the code login in pairwise2genewise 

This note explains two common gotchas when translating 1-based gene indices (from R) into NumPy code, and how the weighted connectivity update works.

---

## 1) Index in R vs Numpy

### Toy data
We have 4 genes labeled **1..4** (R-style, 1-based indexing), and 5 pair rows. This is how the output of propd looks like, as it is computed in R. 

```python
import numpy as np

partner_small = np.array([2, 3, 3, 4, 4])   # 1-based gene ids
pair_small    = np.array([1, 1, 2, 1, 2])   # 1-based gene ids

```


Key rule
NumPy arrays are 0-based.
If you allocate np.zeros(4), valid indices are 0,1,2,3 (not 1..4).

So you must convert to 0-based indices

```python
per_gene_count = np.zeros(4, dtype=np.int64)  # genes 0..3 internally
partner0 = partner_small - 1
pair0    = pair_small - 1

np.add.at(per_gene_count, partner0, 1)
np.add.at(per_gene_count, pair0, 1)

per_gene_count

```

Result:

pair0 contains [0,0,1,0,1] and partner0 contains [1,2,2,3,3]

counts become [3, 3, 2, 2] meaning:

gene1 appears 3 times

gene2 appears 3 times

gene3 appears 2 times

gene4 appears 2 times

Each gene should appear the same number of times (Number of genes - 1)

## 2) Weighted connectivity block: what it does (exactly)
Weighted connectivity is defined as:

For every significant pair (where fdr < 0.05), add (1 - theta) to both genes in that pair.

Toy example including theta and fdr
```python
FDR_SIGNIFICANCE_THRESHOLD = 0.05

theta_small = np.array([0.2, 0.9, 0.4, 0.1, 0.8])
fdr_small   = np.array([0.01, 0.20, 0.03, 0.001, 0.06])

# Convert indices to 0-based:
partner0 = partner_small - 1
pair0    = pair_small - 1

# Initialize accumulator:
per_gene_weighted_connectivity = np.zeros(4, dtype=float)
```

Line-by-line explanation
a) Select which rows are significant
```python
sig = fdr_small < FDR_SIGNIFICANCE_THRESHOLD
# sig == [True, False, True, True, False]
```
So rows 0, 2, and 3 are significant.

b) Compute weights for significant rows only
```python

w = 1.0 - theta_small[sig]
# theta_small[sig] == [0.2, 0.4, 0.1]
# w == [0.8, 0.6, 0.9]
```
c) Add weights to the “partner” gene of each significant row
```python
np.add.at(per_gene_weighted_connectivity, partner0[sig], w)
partner0[sig] == [1, 2, 3]
```

so we add:

+0.8 to gene index 1

+0.6 to gene index 2

+0.9 to gene index 3

Accumulator becomes:

[0.0, 0.8, 0.6, 0.9]

d) Add the same weights to the “pair” gene of each significant row
```python
np.add.at(per_gene_weighted_connectivity, pair0[sig], w)
pair0[sig] == [0, 1, 0]
```

so we add:

+0.8 to gene index 0

+0.6 to gene index 1

+0.9 to gene index 0

Final accumulator:

[1.7, 1.4, 0.6, 0.9]

Interpret output in 1-based gene ids
Index mapping: 0->gene1, 1->gene2, 2->gene3, 3->gene4

So:

gene1 connectivity = 1.7

gene2 connectivity = 1.4

gene3 connectivity = 0.6

gene4 connectivity = 0.9

Why np.add.at (instead of arr[idx] += w)?
Genes repeat many times. np.add.at guarantees correct accumulation even when indices repeat within the same update (it performs unbuffered indexed adds).


## 3) Computing exact mean and median:

### Why the fancy memmap thing is necessary:
You only need the “fancy memmap thing” because you asked for the exact median per gene.

Mean is easy in one pass: keep sum and count per gene.

Exact median is hard in one pass because the median depends on the sorted list of all theta values for that gene, so you must either:

    a) store all theta values per gene somewhere, or

    b) do something more complex (external sorting or per-gene heaps), which still ends up storing almost everything.

With 18,080 genes and all-vs-all pairs, each gene has 18,079 theta values. That means you have to retain about:

total stored theta values = NUMBER_OF_GENES * (NUMBER_OF_GENES - 1)
= 18080 * 18079 ≈ 326 million

if float64: 326M * 8 bytes ≈ 2.6 GB (on disk)

So memmap is simply: “store the big pile of theta values on disk in a way you can access like an array without loading it all into RAM”.

### 3.1) What prepare_memmap_and_offsets() is doing
The idea:

We create one huge 1D array theta_mem on disk, and we reserve a fixed block for each gene.

Because you have complete all-vs-all, each gene has exactly PER_GENE_PAIR_COUNT = 18079 values.

So layout is:

- gene 0 gets positions [0 ... 18078]

- gene 1 gets positions [18079 ... 2*18079 - 1]

- gene 2 gets positions [2*18079 ... 3*18079 - 1]

Code meaning

```python
slice_start = np.arange(NUMBER_OF_GENES, dtype=np.int64) * PER_GENE_PAIR_COUNT
```

slice_start[g] is the first index in the big array where gene g’s theta values will be written.

slice_start is a 1D vector of length 18080, it contains start positions, not a 2D matrix. 

Then:

```python
total_values = NUMBER_OF_GENES * PER_GENE_PAIR_COUNT
theta_mem = np.memmap(..., shape=(total_values,))
next_write_pos = slice_start.copy()
```

- theta_mem is the big disk-backed array

- next_write_pos[g] is a cursor: “where is the next free slot in gene g’s block?”

At the very beginning, the next free slot is the block start.

### 3.2) What second_pass_fill() is doing

For each row (partner, pair, theta), we need to append theta to: partner’s list and pair’s list

Because your file stores each unordered pair once, each theta contributes to exactly two genes.


If we did this naively row-by-row:

```python

pos = next_write_pos[g]
theta_mem[pos] = theta_value
next_write_pos[g] += 1

```

that would work but be slower due to lots of tiny random writes, so the implementation groups writes by gene inside each chunk.


Inside a chunk, we take all “partner” indices (or all “pair” indices), and we:

1. sort rows by gene id

2. find contiguous runs for the same gene

3. write each run as one contiguous block into the memmap

That is exactly what this accomplishes:

```python
order = np.argsort(genes, kind="mergesort")
genes_sorted = genes[order]
thetas_sorted = theta[order]

```


Now all values belonging to the same gene are next to each other. Then:

```python

boundaries = np.flatnonzero(np.diff(genes_sorted)) + 1
starts = np.concatenate(([0], boundaries))
stops  = np.concatenate((boundaries, [genes_sorted.size]))
unique_genes = genes_sorted[starts]
```


This is just a way to compute run-length encoding:

starts[i] : stops[i] is the slice of rows for one gene unique_genes[i]

Then for each run:

```python

k = t - s                       # how many thetas for this gene in this chunk
pos = next_write_pos[g]          # where to begin writing in gene g's reserved block
theta_mem[pos:pos+k] = thetas_sorted[s:t]
next_write_pos[g] += k

```

This appends a block of values for gene g.

###  Now with the tiny toy example (so the logic clicks)

Let NUMBER_OF_GENES=4, PER_GENE_PAIR_COUNT=3 (toy), so each gene has 3 slots.

Then:

- slice_start = [0, 3, 6, 9]

- gene0 positions are 0..2

- gene1 positions are 3..5

- gene2 positions are 6..8

- gene3 positions are 9..11

Suppose in one chunk the partner0 values are:

- genes = [3, 1, 3, 0] and theta = [0.8, 0.2, 0.5, 0.9]

Sorting by gene gives:

- genes_sorted = [0,1,3,3]

- theta_sorted = [0.9,0.2,0.8,0.5]

Runs are:

- gene0 has theta [0.9]

- gene1 has theta [0.2]

- gene3 has theta [0.8,0.5]

We write each as a contiguous block into each gene’s reserved region, advancing the cursor.
