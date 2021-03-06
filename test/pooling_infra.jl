using LightXML
using JuMP, MultilinearOpt, BARON, Gurobi, CPLEX

mutable struct Feed
    id::String
    fc::Float64
    uc::Float64
    lb::Float64
    ub::Float64
    qualities::Dict
end

mutable struct Pool
    id::String
    fc::Float64
    uc::Float64
    cap::Float64
    qualities::Dict
end

mutable struct Product
    id::String
    ur::Float64
    lb::Float64
    ub::Float64
    qualities::Dict
end

mutable struct Connection
    source::String
    destination::String
    fc::Float64
    uc::Float64
end

const Quality = String

c( x::Feed) = x.uc
AL(x::Feed) = x.lb
AU(x::Feed) = x.ub

SS(x::Pool) = x.cap

d(x::Product) = x.ur
DU(x::Product) = x.ub
DL(x::Product) = x.lb

CC(x::Feed, y::Quality) = x.qualities[y]
PL(x::Product, y::Quality) = x.qualities[y][1]
PU(x::Product, y::Quality) = x.qualities[y][2]

# I = feeds, J = products, L = pools, K = qualities
function pooling_problem(I, J, L, K, conns)
    pairs = [(x.source,x.destination) for x in conns]
    Tx(i,l) = (i.id,l.id) in pairs
    Ty(l,j) = (l.id,j.id) in pairs
    Tz(i,j) = (i.id,j.id) in pairs

    m = Model(solver=CplexSolver())
    @variable(m, 0 ≤ q[i in I, l in L] ≤ (Tx(i,l) ? 1 : Inf))
    @variable(m, 0 ≤ y[l in L, j in J] ≤ min(SS(l), DU(j), sum(AU(i) for i in I if Tx(i,l))))
    @variable(m, 0 ≤ z[i in I, j in J] ≤ min(AU(i), DU(j)))

    # Standard Pooling Problem

    @objective(m, Min, sum(c(i)*q[i,l]*y[l,j] for i in I, l in L, j in J if Tx(i,l) && Ty(l,j)) - sum(d(j)*y[l,j] for l in L, j in J if Ty(l,j)) - sum((d(j)-c(i))*z[i,j] for i in I, j in J if Tz(i,j)))

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
            @constraint(m, PL(j,k)*(sum(y[l,j] for l in L if Ty(l,j)) + sum(z[i,j] for i in I if Tz(i,j))) ≤ sum(CC(i,k)*z[i,j] for i in I if Tz(i,j)) + sum(CC(i,k)*q[i,l]*y[l,j] for i in I, l in L if Tx(i,l) && Ty(l,j)))
        end
        if PU(j,k) < Inf
            @constraint(m, PU(j,k)*(sum(y[l,j] for l in L if Ty(l,j)) + sum(z[i,j] for i in I if Tz(i,j))) ≥ sum(CC(i,k)*z[i,j] for i in I if Tz(i,j)) + sum(CC(i,k)*q[i,l]*y[l,j] for i in I, l in L if Tx(i,l) && Ty(l,j)))
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

    m
end

filebase = "/Users/huchette/Dropbox\ (MIT)/research/multilinear_relaxations/sample_problems/sample_problems/standard_lit_problems"

fp = open("/Users/huchette/Dropbox\ (MIT)/research/multilinear_relaxations/pooling_results.csv", "w")

for (roots, dirs, files) in walkdir(filebase), filename in files
    fi = joinpath(filebase,filename)
    xdoc = parse_file(fi)
    xroot = root(xdoc)

    feeds = Any[]
    pools = Any[]
    prods = Any[]
    conns = Any[]
    quals = Any[]
    for ce in child_elements(xroot)
        if name(ce) == "Feed"
            qualities = Dict(attribute(x,"quality_id") => parse(Float64, content(find_element(x,"Level"))) for x in get_elements_by_tagname(ce, "QualityLevel"))
            feed = Feed(attribute(ce, "id"),
                        parse(Float64, content(find_element(ce,"FixedCost"))),
                        parse(Float64, content(find_element(ce,"UnitCost"))),
                        parse(Float64, content(find_element(ce,"LowerBound"))),
                        parse(Float64, content(find_element(ce,"UpperBound"))),
                        qualities)
            push!(feeds, feed)
        elseif name(ce) == "Pool"
            qualities = Dict(attribute(x,"quality_id") => parse(Float64, content(find_element(x,"PercentReduction"))) for x in get_elements_by_tagname(ce, "QualityLevel"))
            pool = Pool(attribute(ce, "id"),
                        parse(Float64, content(find_element(ce,"FixedCost"))),
                        parse(Float64, content(find_element(ce,"UnitCost"))),
                        parse(Float64, content(find_element(ce,"Capacity"))),
                        qualities)
            push!(pools, pool)
        elseif name(ce) == "Product"
            qualities = Dict(attribute(x,"quality_id") => (parse(Float64, content(find_element(x,"LowerBound"))),parse(Float64, content(find_element(x,"UpperBound")))) for x in get_elements_by_tagname(ce, "QualityLevel"))
            product = Product(attribute(ce, "id"),
                           parse(Float64, content(find_element(ce,"UnitRevenue"))),
                           parse(Float64, content(find_element(ce,"LowerBound"))),
                           parse(Float64, content(find_element(ce,"UpperBound"))),
                           qualities)
            push!(prods, product)
        elseif name(ce) == "Connection"
            connection = Connection(attribute(ce, "source"),
                                    attribute(ce, "destination"),
                                    parse(Float64, content(find_element(ce,"FixedCost"))),
                                    parse(Float64, content(find_element(ce,"UnitCost"))))
            push!(conns, connection)
        elseif name(ce) == "Quality"
            push!(quals, attribute(ce, "id"))
        end
    end

    # MisenerLinear
    m1 = pooling_problem(feeds, prods, pools, quals, conns)
    relaxbilinear!(m1, method=:MisenerLinear, disc_level=33)
    t1 = @elapsed optimize!(m1)
    obj1 = objective_value(m1)
    println(fp, "$filename, MisenerLinear, $t1, $obj1")

    # MisenerLog1
    m2 = pooling_problem(feeds, prods, pools, quals, conns)
    relaxbilinear!(m2, method=:MisenerLog1, disc_level=33)
    t2 = @elapsed optimize!(m2)
    obj2 = objective_value(m2)
    println(fp, "$filename, MisenerLog1, $t2, $obj2")

    # MisenerLog2
    m3 = pooling_problem(feeds, prods, pools, quals, conns)
    relaxbilinear!(m3, method=:MisenerLog2, disc_level=33)
    t3 = @elapsed optimize!(m3)
    obj3 = objective_value(m3)
    println(fp, "$filename, MisenerLog2, $t3, $obj3")

    # Logarithmic 1D
    m4 = pooling_problem(feeds, prods, pools, quals, conns)
    relaxbilinear!(m4, method=:Logarithmic1D, disc_level=33)
    t4 = @elapsed optimize!(m4)
    obj4 = objective_value(m4)
    println(fp, "$filename, Logarithmic1D, $t4, $obj4")

    # Logarithmic 2D
    m5 = pooling_problem(feeds, prods, pools, quals, conns)
    relaxbilinear!(m5, method=:Logarithmic2D, disc_level=33)
    t5 = @elapsed optimize!(m5)
    obj5 = objective_value(m5)
    println(fp, "$filename, Logarithmic2D, $t5, $obj5")

    # ZigZag 1D
    m6 = pooling_problem(feeds, prods, pools, quals, conns)
    relaxbilinear!(m6, method=:ZigZag1D, disc_level=33)
    t6 = @elapsed optimize!(m6)
    obj6 = objective_value(m6)
    println(fp, "$filename, ZigZag1D, $t6, $obj6")

    # ZigZag 2D
    m7 = pooling_problem(feeds, prods, pools, quals, conns)
    relaxbilinear!(m7, method=:ZigZag2D, disc_level=33)
    t7 = @elapsed optimize!(m7)
    obj7 = objective_value(m7)
    println(fp, "$filename, ZigZag2D, $t7, $obj7")

    flush(fp)
end

close(fp)
