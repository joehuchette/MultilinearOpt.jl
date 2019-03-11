@testset "isconvex" begin
    m = JuMP.Model()

    JuMP.@variables m begin
       x
       y
       z
   end

   # AffExpr
   @test MultilinearOpt.isconvex(x + y)
   @test MultilinearOpt.isconvex(x - z - 3)

   # QuadExpr
   @test MultilinearOpt.isconvex(x^2)
   @test MultilinearOpt.isconvex(x^2 + 0 * z^2) # test positive semidefinite gramian
   @test MultilinearOpt.isconvex(x^2 + 3 * y - z + 5)
   @test !MultilinearOpt.isconvex(-x^2)
   @test !MultilinearOpt.isconvex(x + y - z ^2 + x^2 + y^2)
   G = rand(3, 3)
   G = G * G'
   vars = [x, y, z]
   expr = dot(vars, G * vars)
   G_back, vars_back = MultilinearOpt.gramian(expr)
   @test vars_back == vars
   @test G_back ≈ G atol=1e-10
   @test MultilinearOpt.isconvex(expr)

   # LinearConstraint
   @test MultilinearOpt.isconvex(JuMP.constraint_object(JuMP.@constraint(m, x + 3 * y == 0)))
   @test MultilinearOpt.isconvex(JuMP.constraint_object(JuMP.@constraint(m, 2 * x - y >= z)))
   @test MultilinearOpt.isconvex(JuMP.constraint_object(JuMP.@constraint(m, 2 * x - y <= z)))

   # QuadConstr
   @test !MultilinearOpt.isconvex(JuMP.constraint_object(JuMP.@constraint(m, x == y * z)))

   @test MultilinearOpt.isconvex(JuMP.constraint_object(JuMP.@constraint(m, x^2 + y^2 <= z)))

   @test !MultilinearOpt.isconvex(JuMP.constraint_object(JuMP.@constraint(m, x^2 + y^2 <= z^2)))

   @test MultilinearOpt.isconvex(JuMP.constraint_object(JuMP.@constraint(m, -x^2 - y^2 >= -z)))
end
