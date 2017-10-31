using MultilinearOpt, JuMP, Gurobi

N = 1:11
S = 1:5
I = 6:7
T = 8:11

K = 1:4

#    1   2   3   4   5   6   7   8   9   10   11
c = [0   0   0   0   0   7   0   0   0    0    0  # 1
     0   0   0   0   0   3   0   0   0    0    0  # 2
     0   0   0   0   0   0   2   0   0    0    0  # 3
     0   0   0   0   0   0  10   0   0    0    0  # 4
     0   0   0   0   0   0   5   0   0    0    0  # 5
     0   0   0   0   0   0   0 -16 -25  -15  -10  # 6
     0   0   0   0   0   0   0 -16 -25  -15  -10] # 7

# A = zeros(11,11);
# A[1,6] = A[2,6] = A[3,7] = A[4,7] = A[5,7] = A[6,8] = A[6,9] = A[6,10] = A[6,11] = A[7,8]= A[7,9]= A[7,10]= A[7,11] = 1

A = [(1,6),(2,6),(3,7),(4,7),(5,7),(6,8),(6,9),(6,10),(6,11),(7,8),(7,9),(7,10),(7,11)]

N⁺(i) = [j for j in N if (i,j) in A]
N⁻(i) = [j for j in N if (j,i) in A]

q = [1.00    6.00    4.00    0.50
     4.00    1.00    3.00    2.00
     4.00    5.50    3.00    0.90
     3.00    3.00    3.00    1.00
     1.00    2.70    4.00    1.60
     0.00    0.00    0.00    0.00
     0.00    0.00    0.00    0.00
     3.00    3.00    3.25    0.75
     4.00    2.50    3.50    1.50
     1.50    5.50    3.90    0.80
     3.00    4.00    4.00    1.80]

b = [75,75,75,75,75,75,75,10,25,30,10]

m = Model(solver=GurobiSolver())
@variable(m, 0 ≤ f[i in N, j in N; (i,j) in A] ≤ maximum(b))
for i in N
    if i in S
        @constraint(m, sum(f[i,j] for j in N⁺(i)) ≤ b[i])
    else
        @constraint(m, sum(f[j,i] for j in N⁻(i)) ≤ b[i])
    end
    @constraint(m, sum(f[i,j] for j in N⁺(i)) == sum(f[j,i] for j in N⁻(i)))
end
@variable(m, 0 ≤ w[i=N,k=K] ≤ q[i,k])
for i in N, k in K
    if i in S
        @constraint(m, w[i,k] == q[i,k])
    else
        @constraint(m, w[i,k]*sum(f[j,i] for j in N⁻(i)) == sum(w[j,k]*f[j,i] for j in N⁻(i)))
    end
end

@objective(m, Min, sum(c[i,j]*f[i,j] for (i,j) in A))

relaxbilinear!(m, method=:Logarithmic2D)
stat = solve(m)
