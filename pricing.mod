# =====================================================================
# Pricing subproblem for one machine j (lecture slide 17, eq (11)):
#
#   SP(j, pi, mu):  max  sum_{i in N} (p[i,j] - pi[i]) z[i]  -  mu[j]
#                   s.t. sum_{i in N} w[i,j] z[i] <= d[j]
#                        z[i] in {0, 1}
#
# A column (the support of z*) is improving when SP_obj > 0. This file
# omits the constant -mu[j] from the objective; the driver subtracts
# mu[j] after the solve and adds the column when ReducedProfit - mu[j]
# is strictly positive.
#
# Driver supplies (for the fixed machine j):
#   red_profit[i] = p[i,j] - pi[i]    (objective coefficients after duals)
#   w[i]          = w[i,j]
#   d             = d[j]
#   fix_in[i], fix_out[i] from the branching path:
#     y[i,j] = 1 forces z[i] = 1   on machine j  (FixIn)
#     y[i,j] = 0 forces z[i] = 0   on machine j  (FixOut)
#     y[i,j'] = 1 with j' != j also forces z[i] = 0 here, because task i
#       is committed to a different machine (driver folds this into
#       fix_out before passing it in).
# =====================================================================

set TASKS;                      # N

param red_profit {TASKS};       # p[i,j] - pi[i]
param w          {TASKS} >= 0;  # w[i,j]
param d                  >= 0;  # d[j]

param fix_in  {TASKS} binary, default 0;
param fix_out {TASKS} binary, default 0;

var Z {TASKS} binary;           # lecture's z_i

maximize ReducedProfit:
    sum {i in TASKS} red_profit[i] * Z[i];

subject to KnapCap:
    sum {i in TASKS} w[i] * Z[i] <= d;

subject to FixIn  {i in TASKS: fix_in[i]  = 1}: Z[i] = 1;
subject to FixOut {i in TASKS: fix_out[i] = 1}: Z[i] = 0;
