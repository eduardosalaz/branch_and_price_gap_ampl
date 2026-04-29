# =====================================================================
# Compact MIP for the Generalized Assignment Problem (lecture eq (8),
# slide 14). Used as a validation oracle for branch-and-price.
#
# Notation matches the lecture:
#   N = TASKS,    indexed by i
#   M = MACHINES, indexed by j
#   p[i,j] : profit of assigning task i to machine j
#   w[i,j] : capacity of machine j consumed by task i
#   d[j]   : capacity of machine j
#   Y[i,j] : 1 if task i assigned to machine j  (lecture's y_ij)
# =====================================================================

set TASKS;
set MACHINES;

param p {TASKS, MACHINES} >= 0;
param w {TASKS, MACHINES} >= 0;
param d {MACHINES}        >= 0;

var Y {TASKS, MACHINES} binary;

maximize Total:
    sum {i in TASKS, j in MACHINES} p[i,j] * Y[i,j];

# Each task assigned to exactly one machine.
subject to AssignTask {i in TASKS}:
    sum {j in MACHINES} Y[i,j] = 1;

# Each machine respects its capacity.
subject to MachineCap {j in MACHINES}:
    sum {i in TASKS} w[i,j] * Y[i,j] <= d[j];
