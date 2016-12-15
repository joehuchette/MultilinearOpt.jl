module MultilinearOpt

import JuMP, PiecewiseLinear

export HyperRectangle, MultilinearFunction, Discretization, outerapproximate, relaxbilinear!

immutable HyperRectangle{D}
    l::NTuple{D,Float64}
    u::NTuple{D,Float64}
end

type MultilinearFunction{D}
    bounds::HyperRectangle{D}
    f::Function
end
(mlf::MultilinearFunction)(args...) = mlf.f(args...)

type Discretization{D}
    d::NTuple{D,Vector{Float64}}
end
Discretization(d...) = Discretization(map(t->convert(Vector{Float64},t), d))

type MultilinearData
    product_dict::Dict{Tuple{JuMP.Variable,JuMP.Variable},JuMP.Variable}

    MultilinearData() = new(Dict{Tuple{JuMP.Variable,JuMP.Variable},JuMP.Variable}())
end

function outerapproximate{D}(m::JuMP.Model, x::NTuple{D,JuMP.Variable}, mlf::MultilinearFunction{D}, disc::Discretization)
    n = length(disc.d)
    V = collect(Base.product(disc.d...))
    T = map(t -> mlf.f(t...), V)
    λ = JuMP.@variable(m, [V], lowerbound=0, upperbound=1, basename="λ")
    z = JuMP.@variable(m, basename="z")
    JuMP.@constraint(m, sum(λ) == 1)
    for i in 1:n
        JuMP.@constraint(m, sum(λ[v]*v[i] for v in V) == x[i])
    end
    JuMP.@constraint(m, sum(λ[v]*mlf.f(v...) for v in V)  == z)
    for i in 1:n
        I = disc.d[i]
        t = length(I)
        k = ceil(Int, log2(t))
        H = PiecewiseLinear.reflected_gray(k)
        y = JuMP.@variable(m, [1:k], Bin, basename="y")
        JuMP.@expression(m, γ[j=1:t], sum(λ[v] for v in V if v[i] == I[j]))
        for j in 1:k
            JuMP.@constraints(m, begin
                H[1][j]*γ[1] + sum(min(H[v][j],H[v-1][j])*γ[v] for v in 2:n-1) + H[n][j]*γ[n] ≤ y[j]
                H[1][j]*γ[1] + sum(max(H[v][j],H[v-1][j])*γ[v] for v in 2:n-1) + H[n][j]*γ[n] ≥ y[j]
            end)
        end
    end
    z
end

const default_disc_level = 5

function relaxbilinear!(m::JuMP.Model)
    # replace each bilinear term in (nonconvex) quadratic constraints with outer approx
    product_dict = Dict() #m.ext[:Multilinear].product_dict
    nonconvex = true # TODO: check this
    if nonconvex
        aff = linearize_quadratic!(m, m.obj, product_dict)
        m.obj = JuMP.QuadExpr(aff)
    end
    for q in m.quadconstr
        nonconvex = true # TODO: check this and preserve convex quadratics
        if nonconvex
            # TODO: merge terms (i.e. x*y + 2x*y = 3x*y)
            t = q.terms
            aff = linearize_quadratic!(m, t, product_dict)
            lb = q.sense == :<= ? -Inf : -aff.constant
            ub = q.sense == :>= ?  Inf : -aff.constant
            aff.constant = 0
            lc = JuMP.LinearConstraint(aff, lb, ub)
            JuMP.addconstraint(m, lc)
        end
    end
    empty!(m.quadconstr) # TODO: change with support for convex quadratic
    nothing
end

function linearize_quadratic!(m::JuMP.Model, t::JuMP.QuadExpr, product_dict::Dict)
    aff = copy(t.aff)
    for i in 1:length(t.qvars1)
        x, y = t.qvars1[i], t.qvars2[i]
        if haskey(product_dict, (x, y))
            z = product_dict[(x,y)]
        elseif haskey(product_dict, (y, x)) # should be unnecessary branch
            z = product_dict[(y,x)]
        else
            @assert x != y # TODO: support non-bilinear terms
            lˣ, lʸ = JuMP.getlowerbound(x), JuMP.getlowerbound(y)
            uˣ, uʸ = JuMP.getupperbound(x), JuMP.getupperbound(y)
            @assert isfinite(lˣ) && isfinite(lʸ) && isfinite(uˣ) && isfinite(uʸ)
            hr = HyperRectangle((lˣ,lʸ), (uˣ,uʸ))
            mlf = MultilinearFunction(hr, (a,b) -> a*b)
            disc_levelˣ = lˣ == uˣ ? 1 : default_disc_level # if variable is fixed, no need to discretize
            disc_levelʸ = lʸ == uʸ ? 1 : default_disc_level
            disc = Discretization(linspace(lˣ,uˣ,disc_levelˣ), linspace(lʸ,uʸ,disc_levelʸ))
            z = outerapproximate(m, (t.qvars1[i],t.qvars2[i]), mlf, disc)
            product_dict[(x,y)] = z
            product_dict[(y,x)] = z
        end
        append!(aff, t.qcoeffs[i]*z)
    end
    aff
end

end # module
