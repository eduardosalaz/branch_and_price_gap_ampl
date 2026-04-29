# =====================================================================
# Restricted master problem for branch-and-price on GAP
# (lecture slide 16, eq (10)).
#
#   max  sum_{j in M, h in C_j} pi_{j,h} x[j,h]
#   s.t. sum_{j,h} a_{j,i,h} x[j,h] = 1     for each task i      (dual pi[i])
#        sum_h     x[j,h]            = 1     for each machine j   (dual mu[j])
#        x[j,h] >= 0                                    (LP relaxation)
#
# Notation matches the lecture:
#   N = TASKS,    indexed by i
#   M = MACHINES, indexed by j
#   C_j = feasible patterns (knapsack-feasible task subsets) of machine j
#   a_{j,i,h} = 1 if task i is in column h of machine j
#   pi_{j,h}  = column reward = sum_{i in column} p[i,j]
#   x[j,h]    = 1 if column h of machine j is selected
#
# Patterns from all machines are flattened into a single PATTERNS set
# indexed by a unique id; pat_machine[h] recovers j and pat_covers[h,i]
# recovers a_{j,i,h}.
#
# Big-M slacks on the cover constraints:
#   - guarantee initial LP feasibility (only empty per-machine patterns
#     are seeded at startup),
#   - propagate infeasibility at deep B&P nodes (leftover Slack > 0 in
#     the optimal LP signals that branching has killed the subproblem).
# =====================================================================

set TASKS;                      # N
set MACHINES;                   # M
set PATTERNS default {};        # union of all generated columns (j,h)

param pat_machine {PATTERNS} in MACHINES;              # which machine owns the pattern
param pat_profit  {PATTERNS} >= 0;                     # pi_{j,h}
param pat_covers  {PATTERNS, TASKS} binary, default 0; # a_{j,i,h}

param BigM > 0;

var X     {PATTERNS} >= 0;      # lecture's x[j,h], flattened over patterns
var Slack {TASKS}    >= 0;

maximize MasterProfit:
    sum {h in PATTERNS} pat_profit[h] * X[h]
  - BigM * sum {i in TASKS} Slack[i];

# Cover: each task picked exactly once across all selected columns.
# Dual is pi[i] (passed to pricing as the task multiplier).
subject to CoverTask {i in TASKS}:
    sum {h in PATTERNS: pat_covers[h,i] = 1} X[h] + Slack[i] = 1;

# Convexity: each machine selects exactly one of its columns.
# Dual is mu[j] (the constant in pricing's reduced-profit test).
subject to Convexity {j in MACHINES}:
    sum {h in PATTERNS: pat_machine[h] = j} X[h] = 1;
