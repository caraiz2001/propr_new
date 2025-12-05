# Step-by-step explanation of `compute_es_stream` (GSEA-like running-sum) with a toy example

This function computes a GSEA-style enrichment score for **each gene**, using a single streaming pass over a file that is already sorted by some ranking (best-to-worst pairs, like a gene-pair score).

Each gene is treated as a “set” containing exactly the rows where that gene appears.  
As we scan the ranked list from top to bottom, each gene has a **running sum**:

- When the current row **does NOT contain** the gene: the gene gets a **miss** (running sum goes down).
- When the current row **DOES contain** the gene: the gene gets a **hit** (running sum goes up).

For each gene we track:
- `maxv[g]`: the maximum running sum reached (positive enrichment)
- `minv[g]`: the minimum running sum reached (negative enrichment)
- final ES is whichever has larger absolute value

---

## What each variable means in plain terms

Inputs:
- `sorted_csv`: file sorted by rank (row 1 is “best”, row N is “worst”)
- `gene_id`: mapping from gene label (like 1..G) to internal 0-based index
- `N`: total number of rows (pairs)

Derived constants:
- `G = len(gene_id)` number of genes
- `K = G - 1` number of pairs each gene appears in (all-vs-all unordered pairs)

Running state arrays (length G):
- `last_idx[g]`: last position (rank) where gene `g` was seen
- `cur[g]`: current running sum at current rank
- `maxv[g]`: maximum value of `cur[g]` so far
- `minv[g]`: minimum value of `cur[g]` so far

Increments:
- `hit_const = 1/K`  (each hit increases running sum by this amount)
- `miss_inc = 1/(N-K)` (each miss decreases running sum by this amount)

Important: Since `K` is constant for every gene (all-vs-all), **hit and miss increments are constant scalars** here.

---

## Toy example: 4 genes

Let genes be `{1,2,3,4}` so `G = 4`.  
All unordered pairs are:

1. (1,2)
2. (1,3)
3. (1,4)
4. (2,3)
5. (2,4)
6. (3,4)

So:
- `N = 6`
- `K = G - 1 = 3` because each gene participates in 3 pairs

Check:
- expected unordered pairs = `G*(G-1)/2 = 4*3/2 = 6` OK

Compute increments:
- `hit_const = 1/K = 1/3 ≈ 0.3333`
- `miss_inc = 1/(N-K) = 1/(6-3) = 1/3 ≈ 0.3333`

### Suppose the ranked (sorted) file order is this
We will number ranks i = 1..6:

| rank i | Partner | Pair |
|--------|---------|------|
| 1      | 1       | 2    |
| 2      | 3       | 4    |
| 3      | 1       | 3    |
| 4      | 2       | 4    |
| 5      | 1       | 4    |
| 6      | 2       | 3    |

This is “sorted by score” already, so we just stream in this order.

Let internal ids be 0-based:
- gene_id[1]=0, gene_id[2]=1, gene_id[3]=2, gene_id[4]=3

Initialize:
- `last_idx = [0,0,0,0]`
- `cur = [0,0,0,0]`
- `maxv = [0,0,0,0]`
- `minv = [0,0,0,0]`

---

## The key line: how misses are applied

When we see gene `g` at rank `i`, we compute how many ranks we “skipped” since last time:

```python
dmiss = (i - last_idx[g] - 1) * miss_inc
cur[g] -= dmiss
cur[g] += hit_const
last_idx[g] = i
```

Explanation:

If we last saw gene g at rank last_idx[g], then ranks
(last_idx[g] + 1) ... (i - 1) are rows where gene g did not appear.

The number of such missed rows is (i - last_idx[g] - 1).

Each miss decreases by miss_inc, so total miss decrement is dmiss.

Then we apply the hit for this row.

This trick lets us avoid updating every gene at every row.
We only update genes when they occur, but we still account for all the misses in between.

## Walk through a few ranks explicitly
Rank i = 1, row (1,2)

- Gene 1 (id0):

    - last_idx[0]=0

    - missed rows since last hit: 1 - 0 - 1 = 0

    - dmiss = 0 * 1/3 = 0

    - cur += hit_const => cur[0] = 0 + 1/3 = 0.3333

    - maxv[0] becomes 0.3333

    - last_idx[0] = 1

- Gene 2 (id1):

    - last_idx[1]=0

    - missed rows: 1 - 0 - 1 = 0

    - cur[1] = 0 + 1/3 = 0.3333

    - maxv[1] = 0.3333

    - last_idx[1] = 1

- Now:

    - cur = [0.3333, 0.3333, 0, 0]

    - last_idx = [1,1,0,0]

Rank i = 2, row (3,4)

- Gene 3 (id2):

    - last_idx[2]=0 -> missed rows: 2 - 0 - 1 = 1

    - dmiss = 1 * 1/3 = 0.3333

    - cur[2] -= 0.3333 => -0.3333

    - minv[2] becomes -0.3333

    - cur[2] += 0.3333 => back to 0

    - maxv[2] remains 0

    - last_idx[2]=2

- Gene 4 (id3):

    - last_idx[3]=0 -> missed rows: 1

    - cur[3] goes -0.3333 then +0.3333 -> 0

    - minv[3] = -0.3333

    - last_idx[3]=2

- Now:

    - cur = [0.3333, 0.3333, 0, 0]

    - minv = [0, 0, -0.3333, -0.3333]

    - last_idx = [1,1,2,2]

Rank i = 3, row (1,3)

- Gene 1 (id0):

    - last_idx[0]=1 -> missed rows: 3 - 1 - 1 = 1

    cur[0] -= 1/3 => 0.3333 - 0.3333 = 0

    cur[0] += 1/3 => 0.3333

    - maxv[0] stays 0.3333

    - last_idx[0]=3

- Gene 3 (id2):

    - last_idx[2]=2 -> missed rows: 3 - 2 - 1 = 0

    - cur[2] += 1/3 => 0.3333

    - maxv[2] becomes 0.3333

    - last_idx[2]=3

And so on for all rows.

## The important pattern:

A gene that appears frequently near the top gets its running sum pushed upward early (positive ES).

A gene that appears mostly near the bottom will accumulate misses early and hits late, producing a negative dip (negative ES).

Why the function does the “tail drift” at the end

When the loop finishes at rank N, some genes might not have been seen in the last few rows.
Those final rows are all misses for those genes, but we never explicitly subtracted them.

So:

```python
tail = N - last_idx
cur -= tail * miss_inc
minv = np.minimum(minv, cur)
```

Meaning:

If a gene’s last hit was at rank 4 and N=6, then tail=2.

That gene should get 2 additional misses (for ranks 5 and 6).

We subtract that all at once.

Then we update minima because this tail can push the running sum lower.

### How ES is chosen at the end

We have:

- pos = maxv (largest positive deviation)

- neg = minv (most negative deviation)

Final ES is whichever has larger absolute magnitude:

- es = np.where(np.abs(pos) >= np.abs(neg), pos, neg)


So:

- If gene g’s strongest signal is positive, ES is positive.

- If gene g’s strongest signal is negative (bigger absolute dip), ES is negative.

### Why last_idx trick is useful

A naive implementation would update every gene on every row:

subtract miss from ~G genes each row (very slow)

This optimized implementation:

- updates only the two genes present in the row

- accounts for all missed rows since last time using dmiss

So it is O(N) row processing with small constant work per row.
