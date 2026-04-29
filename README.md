# Branch-and-Price for the Generalized Assignment Problem

A from-scratch branch-and-price implementation for GAP, written in modern
AMPL / amplpy style. Master and pricing problems live in `.mod` files;
all orchestration (column generation, branching, tree search, validation)
lives in Python.

The notation matches the lecture (`Branch-and-Price_FlamandLecture.pdf`)
exactly: `i` indexes tasks (`N`), `j` indexes machines (`M`), `p[i,j]`
is profit, `w[i,j]` is weight, `d[j]` is capacity, and the problem is
**profit maximization** throughout — no negation to a min problem.

## Files

| File | Role |
| --- | --- |
| `master.mod`  | Restricted master problem: set-partitioning over per-machine feasible patterns, with Big-M slacks for guaranteed initial feasibility. |
| `pricing.mod` | Per-machine 0-1 knapsack pricing subproblem, with parameterised `fix_in` / `fix_out` task sets to absorb branching decisions. |
| `compact.mod` | Direct GAP MIP, used as a validation oracle for the B&P solution. |
| `bnp.py`      | Driver: column generation, most-fractional branching on `y[i,j]`, DFS B&B tree, demos. |

## Run

```bash
python -m pip install --upgrade amplpy
python -m amplpy.modules install highs
python bnp.py
```

`bnp.py` ships with three demos:

1. **Lecture slide 18 worked example** (2 machines, 3 tasks). Optimum
   `z* = 17` matches slide 23 exactly.
2. **Symmetric instance** designed to defeat the natural strength of
   Dantzig-Wolfe — identical machines and identical tasks force a
   fractional LP root, so branching actually fires.
3. **Larger random instance** where CG closes the gap at the root in
   a single node (no branching).

## Decomposition recap (lecture eq (8) → (10), (11))

GAP (slide 14):

```
max  sum_{i in N, j in M} p[i,j] y[i,j]
s.t. sum_{j in M} y[i,j] = 1                   forall tasks i
     sum_{i in N} w[i,j] y[i,j] <= d[j]        forall machines j
     y[i,j] in {0, 1}
```

Block-decompose by machine. Each machine's polytope of feasible task
subsets becomes the columns of a master:

```
master    max  sum_{j,h} pi_{j,h} x[j,h]
          s.t. sum_{j,h} a_{j,i,h} x[j,h] = 1   forall tasks i      (dual pi[i])
               sum_h     x[j,h]            = 1   forall machines j  (dual mu[j])
               x >= 0

pricing(j)  max  sum_i (p[i,j] - pi[i]) z[i] - mu[j]
            s.t. sum_i w[i,j] z[i] <= d[j]
                 z in {0, 1}             (a 0-1 knapsack)
```

A column is improving when its SP objective is strictly positive.

## Branching (slides 24-27)

Most-fractional rule on the recovered original variable
`y[i,j] = sum_{h: i in column h of machine j} x[j,h]`. The up child
fixes `y[i,j] = 1` (machine `j` must take task `i`); the down child
fixes `y[i,j] = 0` (machine `j` cannot take task `i`). These propagate
as:

| Decision | Existing master columns | Pricing of machine `j` | Pricing of machine `j' != j` |
| --- | --- | --- | --- |
| `y[i,j] = 0` | drop columns of machine `j` containing `i` | `z[i] := 0` | unchanged |
| `y[i,j] = 1` | drop columns of machine `j` *not* containing `i`, and columns of any other machine containing `i` | `z[i] := 1` | `z[i] := 0` |

Master-side: `_apply_node_context` walks the global pattern pool and
fixes incompatible `X` variables to 0 via AMPL's `fix` / `unfix`
(idiomatic runtime variable-bound primitives in AMPL).
Pricing-side: `_machine_branching_constraints` translates the path of
forced / forbidden `(i, j)` decisions into per-machine `fix_in`/`fix_out`
parameters that the `.mod` file consumes via `subject to FixIn` /
`FixOut` constraints.

This branching scheme is "compatible with pricing" in the standard sense:
the constraints added by branching are exactly representable as variable
fixings in the knapsack pricing problem, so pricing remains a clean
knapsack and never needs side constraints from the branching tree.

The lecture (slide 25) also discusses a two-step rule: pick a fractional
master column first, then within it pick the original variable with the
largest weight `w`. The implementation uses the simpler, equivalent
formulation: most-fractional rule directly on the recovered `y[i,j]`.

## Big-M slacks instead of Phase I / artificial variables

`master.mod` carries `Slack[i] >= 0` on each cover constraint with
coefficient `-BigM` in the (max) objective. This:

- Makes the very first LP solve feasible without needing initial
  knapsack-feasible patterns (only empty per-machine patterns are seeded).
- Detects infeasibility at deep nodes: when forced/forbidden decisions
  make the subproblem infeasible, the LP optimum keeps a slack > 0 and
  the node is pruned (`_slack_active` check).

If you prefer a Phase I / Phase II setup, replace the slacks with
artificial variables priced into a Phase I objective and switch to the
real objective once they leave the basis.

## Why the asymmetric demo solves at the root

The DW master LP for GAP is provably at least as tight as the compact
LP relaxation, and for many random GAP instances it is integer at the
root — this is exactly the famous strength of Dantzig-Wolfe for set-
partitioning structure. The symmetric demo is constructed precisely to
defeat this: identical machines and identical tasks force the LP into a
symmetric fractional solution that no integer combination of patterns
can match at the same profit.

## Natural next steps for production use

The implementation is intentionally pedagogical. To scale to larger GAP
benchmarks (Beasley OR-Library types `c`, `d`, `e`), reasonable upgrades
are:

1. **Heuristic primal at root.** Run a greedy / regret-based GAP
   heuristic before the first LP to set a finite incumbent and enable
   bound-pruning at all interior nodes.
2. **Best-first or best-bound search** instead of DFS, especially once
   the incumbent is reasonable (lecture slide 28 contrasts the two).
3. **Strong branching** on candidate `(i, j)` pairs.
4. **Stabilisation** (Wentges, dual-price smoothing, in-out) to cut
   tailing-off in column generation.
5. **Dynamic column pool management.** This implementation never
   removes columns; for long runs you'd evict columns whose reduced
   profits have stayed sufficiently negative across many LP solves.
6. **Lagrangian dual bound.** At each LP, the bound
   `LP_obj + sum_j max(0, knap_j - mu_j)` is a valid Lagrangian upper
   bound that often allows pruning before column generation converges.
7. **Solver swap.** Master is pure LP — HiGHS is fine. Pricing is a
   sequence of small 0-1 knapsacks; for harder instances, a dedicated
   knapsack DP (pseudo-polynomial in capacity) typically beats a MIP
   solver. Drop in a knapsack DP behind the same `_solve_pricing`
   interface.
8. **Ryan-Foster branching** on pairs of tasks (same-machine vs.
   different-machine) as an alternative to branching on `y[i,j]`. Useful
   when many machines are interchangeable.
