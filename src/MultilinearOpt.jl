module MultilinearOpt

import JuMP, PiecewiseLinearOpt

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

function outerapproximate{D}(m::JuMP.Model, x::NTuple{D,JuMP.Variable}, mlf::MultilinearFunction{D}, disc::Discretization, method)
    vform = (method in (:Logarithmic1D,:Logarithmic2D,:Unary))
    r = length(disc.d)
    @assert r == 2
    V = collect(Base.product(disc.d...))
    T = map(t -> mlf.f(t...), V)
    z = JuMP.@variable(m, basename="z")
    if vform
        PiecewiseLinearOpt.initPWL!(m)
        λ = JuMP.@variable(m, [V], lowerbound=0, upperbound=1, basename="λ")
        JuMP.@constraint(m, sum(λ) == 1)
        for i in 1:r
            JuMP.@constraint(m, sum(λ[v]*v[i] for v in V) == x[i])
        end
        JuMP.@constraint(m, sum(λ[v]*mlf.f(v...) for v in V)  == z)
        for i in 1:r
            I = disc.d[i]
            n = length(I)-1
            n > 1 || continue
            if method == :Logarithmic1D || method == :Logarithmic2D
                JuMP.@expression(m, γ[j=1:(n+1)], sum(λ[v] for v in V if v[i] == I[j]))
                k = ceil(Int, log2(n))
                H = PiecewiseLinearOpt.reflected_gray_codes(k)
                PiecewiseLinearOpt.sos2_logarthmic_formulation!(m, γ)
            elseif method == :Unary
                PiecewiseLinearOpt.sos2_mc_formulation!(m, γ)
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
            Δy = JuMP.@variable(m, [1:NP], lowerbound=0, upperbound=yᵘ-yˡ)
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
            Δy   = JuMP.@variable(m, [1:NP], lowerbound=0, upperbound=yᵘ-yˡ)
            λhat = JuMP.@variable(m, [1:NP], lowerbound=0, upperbound=1)
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
            Δy = JuMP.@variable(m, [1:NL], lowerbound=0, upperbound=yᵘ-yˡ)
            s  = JuMP.@variable(m, [1:NL], lowerbound=0, upperbound=yᵘ-yˡ)
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

const default_disc_level = 9

function grammian(expr::GenericQuadExpr)
    vars = unique([expr.qvars1; expr.qvars2])
    varindices = Dict((v, i) for (i, v) in enumerate(vars))
    n = length(vars)
    T = eltype(expr.qcoeffs)
    grammian = spzeros(T, n, n)
    for (var1, var2, coeff) in zip(expr.qvars1, expr.qvars2, expr.qcoeffs)
        ind1, ind2 = varindices[var1], varindices[var2]
        grammian[ind1, ind2] = coeff
        grammian[ind2, ind1] = coeff
    end
    grammian, vars
end

isconvex(::Any) = false # fallback; TODO: more methods
isconvex(expr::GenericQuadExpr) = isposdef(first(grammian(expr)))

function relaxbilinear!(m::JuMP.Model; method=:Logarithmic1D)
    # replace each bilinear term in (nonconvex) quadratic constraints with outer approx
    product_dict = Dict() #m.ext[:Multilinear].product_dict
    if !isconvex(m.obj)
        aff = linearize_quadratic!(m, m.obj, product_dict, method)
        m.obj = JuMP.QuadExpr(aff)
    end
    for q in m.quadconstr
        if !isconvex(q)
            # TODO: merge terms (i.e. x*y + 2x*y = 3x*y)
            t = q.terms
            aff = linearize_quadratic!(m, t, product_dict, method)
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

function linearize_quadratic!(m::JuMP.Model, t::JuMP.QuadExpr, product_dict::Dict, method)
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
            disc_levelʸ = lʸ == uʸ ? 1 : (method != :Logarithmic2D ? 2 : default_disc_level)
            # disc = Discretization(linspace(lˣ,uˣ,disc_levelˣ), linspace(lʸ,uʸ,disc_levelʸ))
            disc = Discretization(linspace(lˣ,uˣ,disc_levelˣ), linspace(lʸ,uʸ,disc_levelʸ))
            z = outerapproximate(m, (t.qvars1[i],t.qvars2[i]), mlf, disc, method)
            product_dict[(x,y)] = z
            product_dict[(y,x)] = z
        end
        append!(aff, t.qcoeffs[i]*z)
    end
    aff
end

end # module
