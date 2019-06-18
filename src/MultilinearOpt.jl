module MultilinearOpt

import JuMP, MathOptInterface, PiecewiseLinearOpt

using LinearAlgebra: Symmetric, eigvals

export HyperRectangle, MultilinearFunction, Discretization, outerapproximate, relaxbilinear!

const MOI = MathOptInterface
const MOIU = MOI.Utilities

struct HyperRectangle{D}
    l::NTuple{D,Float64}
    u::NTuple{D,Float64}
end

mutable struct MultilinearFunction{D}
    bounds::HyperRectangle{D}
    f::Function
end
(mlf::MultilinearFunction)(args...) = mlf.f(args...)

mutable struct Discretization{D}
    d::NTuple{D,Vector{Float64}}
end
Discretization(d...) = Discretization(map(t->convert(Vector{Float64},t), d))

mutable struct MultilinearData
    product_dict::Dict{JuMP.UnorderedPair{JuMP.VariableRef}, JuMP.VariableRef}

    MultilinearData() = new(Dict{JuMP.UnorderedPair{JuMP.VariableRef}, JuMP.VariableRef}())
end

function outerapproximate(m::JuMP.Model, x::NTuple{D,JuMP.VariableRef}, mlf::MultilinearFunction{D}, disc::Discretization, method) where {D}
    vform = (method in (:Logarithmic1D,:Logarithmic2D,:ZigZag1D,:ZigZag2D,:Unary))
    r = length(disc.d)
    @assert r == 2
    V = collect(Base.product(disc.d...))
    T = map(t -> mlf.f(t...), V)
    z = JuMP.@variable(m, base_name="z")
    if vform
        PiecewiseLinearOpt.initPWL!(m)
        λ = JuMP.@variable(m, [V], lower_bound=0, upper_bound=1, base_name="λ")
        JuMP.@constraint(m, sum(λ) == 1)
        for i in 1:r
            JuMP.@constraint(m, sum(λ[v]*v[i] for v in V) == x[i])
        end
        JuMP.@constraint(m, sum(λ[v]*mlf.f(v...) for v in V)  == z)
        for i in 1:r
            I = disc.d[i]
            n = length(I)-1
            n > 1 || continue
            γ = JuMP.@expression(m, [j=1:(n+1)], sum(λ[v] for v in V if v[i] == I[j]))
            if method == :Logarithmic1D || method == :Logarithmic2D
                PiecewiseLinearOpt.sos2_logarithmic_formulation!(m, γ)
            elseif method == :Unary
                PiecewiseLinearOpt.sos2_mc_formulation!(m, γ) # TODO: where's γ supposed to come from?
            elseif method == :ZigZag1D || method == :ZigZag2D
                PiecewiseLinearOpt.sos2_zigzag_general_integer_formulation!(m, γ)
            else
                throw(ArgumentError("Unrecognized method: $method"))
            end
        end
    else
        # Methods from Misener only work for bilinear terms, and discretized along only one dimension
        @assert minimum(map(length, disc.d)) == 2
        iˣ = length(disc.d[1]) == 2 ? 2 : 1 # direction we discretize along
        iʸ = iˣ == 1 ? 2 : 1
        I = disc.d[iˣ]
        NP = length(I)-1
        xˡ, xᵘ = minimum(I), maximum(I)
        yˡ, yᵘ = minimum(disc.d[iʸ]), maximum(disc.d[iʸ])
        a = I[2] - I[1] # should assert all lengths are the same
        x, y = x[iˣ], x[iʸ]
        if method == :MisenerLinear
            λ  = JuMP.@variable(m, [1:NP], Bin)
            Δy = JuMP.@variable(m, [1:NP], lower_bound=0, upper_bound=yᵘ-yˡ)
            JuMP.@constraint(m, sum(λ) == 1)
            JuMP.@constraints(m, begin
                xˡ + sum(a*(nP-1)*λ[nP] for nP in 1:NP) ≤ x
                x ≤ xˡ + sum(a*nP*λ[nP] for nP in 1:NP)
            end)
            JuMP.@constraints(m, begin
                y == yˡ + sum(Δy[nP] for nP in 1:NP)
                [nP=1:NP], Δy[nP] ≤ (yᵘ-yˡ)*λ[nP]
            end)
            JuMP.@constraints(m, begin
                z ≥ x*yˡ + sum((xˡ+a*(nP-1))* Δy[nP]                for nP in 1:NP)
                z ≥ x*yᵘ + sum((xˡ+a* nP   )*(Δy[nP]-(yᵘ-yˡ)*λ[nP]) for nP in 1:NP)
                z ≤ x*yˡ + sum((xˡ+a* nP   )* Δy[nP]                for nP in 1:NP)
                z ≤ x*yᵘ + sum((xˡ+a*(nP-1))*(Δy[nP]-(yᵘ-yˡ)*λ[nP]) for nP in 1:NP)
            end)
        elseif method == :MisenerLog1
            NL = ceil(Int, log2(NP))
            λ    = JuMP.@variable(m, [1:NL], Bin)
            Δy   = JuMP.@variable(m, [1:NP], lower_bound=0, upper_bound=yᵘ-yˡ)
            λhat = JuMP.@variable(m, [1:NP], lower_bound=0, upper_bound=1)
            JuMP.@constraints(m, begin
                xˡ + sum(2^(NL-nL)*a*λ[nL] for nL in 1:NL)     ≤ x
                xˡ + sum(2^(NL-nL)*a*λ[nL] for nL in 1:NL) + a ≥ x
            end)
            JuMP.@constraint(m, sum(λhat) == 1)
            for nL in 1:NL
                JuMP.@constraints(m, begin
                    sum(λhat[nP] for nP in 1:NP if mod(floor((nP-1)/2^(NL-nL)),2)==0) ≤ 1 - λ[nL]
                    sum(λhat[nP] for nP in 1:NP if mod(floor((nP-1)/2^(NL-nL)),2)==1) ≤     λ[nL]
                end)
            end
            for nP in 1:NP
                JuMP.@constraint(m, Δy[nP] ≤ (yᵘ-yˡ)*λhat[nP])
            end
            JuMP.@constraint(m, y == yˡ + sum(Δy))
            JuMP.@constraints(m, begin
                z ≥ x*yˡ + sum((xˡ+a*(nP-1))* Δy[nP]                   for nP in 1:NP)
                z ≥ x*yᵘ + sum((xˡ+a* nP   )*(Δy[nP]-(yᵘ-yˡ)*λhat[nP]) for nP in 1:NP)
                z ≤ x*yˡ + sum((xˡ+a* nP   )* Δy[nP]                   for nP in 1:NP)
                z ≤ x*yᵘ + sum((xˡ+a*(nP-1))*(Δy[nP]-(yᵘ-yˡ)*λhat[nP]) for nP in 1:NP)
            end)
        elseif method == :MisenerLog2
            NL = ceil(Int, log2(NP))
            λ  = JuMP.@variable(m, [1:NL], Bin)
            Δy = JuMP.@variable(m, [1:NL], lower_bound=0, upper_bound=yᵘ-yˡ)
            s  = JuMP.@variable(m, [1:NL], lower_bound=0, upper_bound=yᵘ-yˡ)
            JuMP.@constraints(m, begin
                xˡ +     sum(2^(nL-1)*a*λ[nL] for nL in 1:NL) ≤ x
                xˡ + a + sum(2^(nL-1)*a*λ[nL] for nL in 1:NL) ≥ x
                xˡ + a + sum(2^(nL-1)*a*λ[nL] for nL in 1:NL) ≤ xᵘ
            end)
            for nL in 1:NL
                JuMP.@constraints(m, begin
                    Δy[nL] ≤ (yᵘ-yˡ)*λ[nL]
                    Δy[nL] == (y-yˡ) - s[nL]
                    s[nL] ≤ (yᵘ-yˡ)*(1-λ[nL])
                end)
            end
            JuMP.@constraints(m, begin
                z ≥ x*yˡ +  xˡ   *(y-yˡ) + sum(a*2^(nL-1)* Δy[nL]             for nL in 1:NL)
                z ≥ x*yᵘ + (xˡ+a)*(y-yᵘ) + sum(a*2^(nL-1)*(Δy[nL]-(yᵘ-yˡ)*λ[nL]) for nL in 1:NL)
                z ≤ x*yˡ + (xˡ+a)*(y-yˡ) + sum(a*2^(nL-1)* Δy[nL]             for nL in 1:NL)
                z ≤ x*yᵘ +  xˡ   *(y-yᵘ) + sum(a*2^(nL-1)*(Δy[nL]-(yᵘ-yˡ)*λ[nL]) for nL in 1:NL)
            end)
        else
            throw(ArgumentError("Unrecognized method: $method"))
        end
    end
    z
end

function gramian(expr::JuMP.GenericQuadExpr{T}) where T
    vars = unique(Iterators.flatten((var1, var2) for (coeff, var1, var2) in JuMP.quad_terms(expr)))
    varindices = Dict(v => i for (i, v) in enumerate(vars))
    n = length(vars)
    gramian = zeros(T, n, n)
    for (coeff, var1, var2) in JuMP.quad_terms(expr)
        ind1, ind2 = varindices[var1], varindices[var2]
        row, col = extrema((ind1, ind2))
        gramian[row, col] += row == col ? coeff : coeff / 2
    end
    Symmetric(gramian), vars
end

ispossemidef(mat) = all(eigvals(mat) .>= -1e-10)
isconvex(x) = error("Could not determine convexity.")
isconvex(expr::JuMP.GenericAffExpr) = true
isconvex(expr::JuMP.GenericQuadExpr) = ispossemidef(first(gramian(expr)))
isconvex(set::MOI.LessThan) = true
isconvex(set::MOI.GreaterThan) = true
isconvex(set::MOI.EqualTo) = true
isconvex(set::MOI.Interval) = true

function isconvex(constr::JuMP.ScalarConstraint)
    f = JuMP.jump_function(constr)
    set = JuMP.moi_set(constr)
    if f isa JuMP.GenericAffExpr
        return isconvex(set)
    elseif f isa JuMP.GenericQuadExpr
        interval = MOI.Interval(set)
        l, u = interval.lower, interval.upper
        if l == -Inf && isfinite(u)
            return isconvex(f)
        elseif isfinite(l) && u == Inf
            return isconvex(-f)
        elseif isfinite(l) && isfinite(u)
            return all(iszero, coeff for (coeff, _, _) in JuMP.quad_terms(f))
        elseif l == -Inf && u == Inf
            return true
        else
            error("Should never get here.")
        end
    else
        error("Function type $(typeof(f)) not recognized.")
    end
end

function relaxbilinear!(m::JuMP.Model; method=:Logarithmic1D, disc_level::Int = 9)
    # replace each bilinear term in (nonconvex) quadratic constraints with outer approx
    product_dict = Dict{JuMP.UnorderedPair{JuMP.VariableRef}, JuMP.VariableRef}() #m.ext[:Multilinear].product_dict
    obj = JuMP.objective_function(m)
    if !isconvex(obj)
        aff = linearize_quadratic!(m, obj, product_dict, method, disc_level)
        JuMP.set_objective_function(m, aff)
    end
    linearized_quad_constrs = JuMP.ConstraintRef[]
    for (F, S) in JuMP.list_of_constraint_types(m)
        if F <: JuMP.GenericQuadExpr
            for constr in JuMP.all_constraints(m, F, S)
                q = JuMP.constraint_object(constr)
                if !isconvex(q)
                    f = JuMP.jump_function(q)
                    aff = linearize_quadratic!(m, f, product_dict, method, disc_level)
                    set = JuMP.moi_set(q)
                    interval = MOI.Interval(set)
                    lb = interval.lower - JuMP.constant(aff)
                    ub = interval.upper - JuMP.constant(aff)
                    aff.constant = 0
                    JuMP.@constraint(m, lb <= aff <= ub)
                    push!(linearized_quad_constrs, constr)
                end
            end
        end
    end
    for constr in linearized_quad_constrs
        JuMP.delete(m, constr)
    end
    nothing
end

function linearize_quadratic!(m::JuMP.Model, t::JuMP.QuadExpr, product_dict::Dict, method::Symbol, disc_level::Int)
    aff = copy(t.aff)
    for (coeff, x, y) in JuMP.quad_terms(t)
        z = get!(product_dict, JuMP.UnorderedPair(x, y)) do
            # TODO: better relaxation for the case x == y
            lˣ, lʸ = JuMP.lower_bound(x), JuMP.lower_bound(y)
            uˣ, uʸ = JuMP.upper_bound(x), JuMP.upper_bound(y)
            @assert isfinite(lˣ) && isfinite(lʸ) && isfinite(uˣ) && isfinite(uʸ)
            hr = HyperRectangle((lˣ,lʸ), (uˣ,uʸ))
            mlf = MultilinearFunction(hr, (a,b) -> a*b)
            disc_levelˣ = lˣ == uˣ ? 1 : disc_level # if variable is fixed, no need to discretize
            disc_levelʸ = lʸ == uʸ ? 1 : (!(method in (:Logarithmic2D,:ZigZag2D)) ? 2 : disc_level)
            # disc = Discretization(range(lˣ, stop=uˣ, length=disc_levelˣ),  range(lʸ, stop=uʸ, length=disc_levelʸ))
            disc = Discretization(range(lˣ, stop=uˣ, length=disc_levelˣ),  range(lʸ, stop=uʸ, length=disc_levelʸ))
            outerapproximate(m, (x, y), mlf, disc, method)
        end
        JuMP.add_to_expression!(aff, coeff * z)
    end
    aff
end

end # module
