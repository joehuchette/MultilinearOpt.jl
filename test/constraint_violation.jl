using MultilinearOpt
using JuMP
using Gurobi
using Test
using Random
using LinearAlgebra
using MathOptInterface

const MOI = MathOptInterface

function test_constraint_violation(f, xbounds::NTuple{2}, ybounds::NTuple{2};
        method::Symbol, disc_level::Integer, rng::AbstractRNG, atol::Real, num_tests::Integer,
        sense::MOI.OptimizationSense)
    m = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0))
    @variable m x lower_bound=xbounds[1] upper_bound=xbounds[2]
    @variable m y lower_bound=ybounds[1] upper_bound=ybounds[2]
    @variable m z
    @constraint m z == f(x, y)
    relaxbilinear!(m, method=method, disc_level=disc_level)
    for i = 1 : num_tests
        if sense != MOI.FEASIBILITY_SENSE
            x̄ = xbounds[1] + rand(rng) * (xbounds[2] - xbounds[1])
            ȳ = ybounds[1] + rand(rng) * (ybounds[2] - ybounds[1])
            z̄ = f(x̄, ȳ) + rand(rng) - 0.5
            min_objective = sum(x -> x^2, [x, y, z] - [x̄, ȳ, z̄])
            if sense == MOI.MIN_SENSE
                @objective m Min min_objective
            else
                @objective m Max -min_objective
            end
        end
        optimize!(m)
        bounds_tol = 1e-8
        @test xbounds[1] - bounds_tol <= value(x) <= xbounds[2] + bounds_tol
        @test ybounds[1] - bounds_tol <= value(y) <= ybounds[2] + bounds_tol
        violation = f(value(x), value(y)) - value(z)
        # @show violation
        # @show value(x) value(y) value(z)
        # println()
        @test violation ≈ 0 atol=atol
    end
end

@testset "constraint violation" begin
    rng = MersenneTwister(1)
    atol = 1.1e-2
    num_tests = 10
    xbounds = (-0.5, 0.5)
    ybounds = (-0.4, 0.6)
    for (method, disc_level) in  [
            (:Logarithmic1D, 31),
            (:Logarithmic2D, 6),
            (:ZigZag1D, 31),
            (:ZigZag2D, 6),
            # (:Unary, 31) # broken
            (:MisenerLinear, 31),
            (:MisenerLog1, 31),
            (:MisenerLog2, 31)]

        for sense in [MOI.MIN_SENSE, MOI.MAX_SENSE, MOI.FEASIBILITY_SENSE]
            test_constraint_violation(*, xbounds, ybounds;
                method=method, disc_level=disc_level,
                atol=atol, rng=rng, num_tests=num_tests, sense=sense)

            test_constraint_violation((x, y) -> x * y + 3, xbounds, ybounds;
                method=method, disc_level=disc_level,
                atol=atol, rng=rng, num_tests=num_tests, sense=sense)

            test_constraint_violation((x, y) -> x^2, xbounds, ybounds;
                method=method, disc_level=disc_level,
                atol=atol, rng=rng, num_tests=num_tests, sense=sense)

            test_constraint_violation((x, y) -> x^2 + x * y - y, xbounds, ybounds;
                method=method, disc_level=disc_level,
                atol=2 * atol, rng=rng, num_tests=num_tests, sense=sense)
        end
    end
end
