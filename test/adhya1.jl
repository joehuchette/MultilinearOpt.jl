using JuMP, MultilinearOpt

# using BARON
# m = Model(with_optimizer(Baron.Optimizer))

using Gurobi
m = Model(with_optimizer(Gurobi.Optimizer))

#
# Problem Topology

I = 1:5 # Input Feed Stocks
J = 1:4 # Products
L = 1:2 # Pools
K = 1:4 # Qualities

# i,l -> Bool
Tx(i,l) = (i,l) in ((1,1),(2,1),(3,1),(4,2),(5,2)) ? true : false
Ty(l,j) = true
Tz(i,j) = false

#
# Problem Parameters

# Product unit price
d = Dict(1=>16, 2=>25, 3=>15, 4=>10)

# Feed unit price
c = Dict(1=>7, 2=>3, 3=>2, 4=>10, 5=>5)

# Maximum available flow
AU(i) = Inf

# Minimum available flow
AL(i) = 0.0

# Pool size
SS(l) = Inf

# Maximum product demand
DU(j) = Dict(1=>10, 2=>25, 3=>30, 4=>10)[j]

# Minimum product demand
DL(j) = 0.0

# Feed concentration
CC = [1.0    6.0  4.0  0.5
      4.0    1.0  3.0  2.0
      4.0    5.5  3.0  0.9
      3.0    3.0  3.0  1.0
      1.0    2.7  4.0  1.6]

# Maximum Allowable Product Concentration
PU(j,k) = [3.00  3.00  3.25  0.75
           4.00  2.50  3.50  1.50
           1.50  5.50  3.90  0.80
           3.00  4.00  4.00  1.80][j,k]

# Minimum Allowable Product Concentration
PL(j,k) = 0.0

#
# Hard Bounds

@variable(m, 0 ≤ q[i in I, l in L] ≤ (Tx(i,l) ? 1 : Inf))
@variable(m, 0 ≤ y[l in L, j in J] ≤ min(SS(l), DU(j), sum(AU(i) for i in I if Tx(i,l))))
@variable(m, 0 ≤ z[i in I, j in J] ≤ min(AU(i), DU(j)))

#
# Standard Pooling Problem

@objective(m, Min, sum(c[i]*q[i,l]*y[l,j] for i in I, l in L, j in J if Tx(i,l) && Ty(l,j)) - sum(d[j]*y[l,j] for l in L, j in J if Ty(l,j)) - sum((d[j]-c[i])*z[i,j] for i in I, j in J if Tz(i,j)))

# Availability bounds
for i in I
    if AL(i) > 0
        @constraint(m, sum(q[i,l]*y[l,j] for j in J, l in L if Tx(i,l) && Ty(l,j)) + sum(z[i,j] for j in J if Tz(i,j)) ≥ AL(i))
    end
    if AU(i) < Inf
        @constraint(m, sum(q[i,l]*y[l,j] for j in J, l in L if Tx(i,l) && Ty(l,j)) + sum(z[i,j] for j in J if Tz(i,j)) ≤ AU(i))
    end
end

# Pool Capacity ========================
for l in L
    if SS(l) < Inf
        @constraint(m, sum(y[l,j] for j in J if Ty(l,j)) ≤ SS(l))
    end
end

# Product Demand =======================
for j in J
    if DL(j) > 0
        @constraint(m, sum(y[l,j] for l in L if Ty(l,j)) + sum(z[i,j] for i in I if Tz(i,j)) ≥ DL(j))
    end
    if DU(j) < Inf
        @constraint(m, sum(y[l,j] for l in L if Ty(l,j)) + sum(z[i,j] for i in I if Tz(i,j)) ≤ DU(j))
    end
end

# Product Quality ======================
for j in J, k in K
    if PL(j,k) > 0
        @constraint(m, PL(j,k)*(sum(y[l,j] for l in L if Ty(l,j)) + sum(z[i,j] for i in I if Tz(i,j))) ≤ sum(CC[i,k]*z[i,j] for i in I if Tz(i,j)) + sum(CC[i,k]*q[i,l]*y[l,j] for i in I, l in L if Tx(i,l) && Ty(l,j)))
    end
    if PU(j,k) < Inf
        @constraint(m, PU(j,k)*(sum(y[l,j] for l in L if Ty(l,j)) + sum(z[i,j] for i in I if Tz(i,j))) ≥ sum(CC[i,k]*z[i,j] for i in I if Tz(i,j)) + sum(CC[i,k]*q[i,l]*y[l,j] for i in I, l in L if Tx(i,l) && Ty(l,j)))
    end
end

# Simplex Constraint ===================
for l in L
    @constraint(m, sum(q[i,l] for i in I if Tx(i,l)) == 1)
end

# PQ Cut ===============================
for l in L, j in J
    @constraint(m, sum(q[i,l]*y[l,j] for i in I if Tx(i,l)) == y[l,j])
end

relaxbilinear!(m)
optimize!(m)
