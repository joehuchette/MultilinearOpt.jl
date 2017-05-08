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
   @test MultilinearOpt.isconvex(x^2 + 3 * y - z + 5)
   @test !MultilinearOpt.isconvex(-x^2)
   @test !MultilinearOpt.isconvex(x + y - z ^2 + x^2 + y^2)
   G = rand(3, 3)
   G = G * G'
   vars = [x; y; z]
   @test MultilinearOpt.isconvex(dot(vars, G * vars))

   # LinearConstraint
   @test MultilinearOpt.isconvex(JuMP.LinearConstraint(JuMP.@constraint(m, x + 3 * y == 0)))
   @test MultilinearOpt.isconvex(JuMP.LinearConstraint(JuMP.@constraint(m, 2 * x - y >= z)))
   @test MultilinearOpt.isconvex(JuMP.LinearConstraint(JuMP.@constraint(m, 2 * x - y <= z)))

   # QuadConstr
   JuMP.@constraint(m, x == y * z)
   @test !MultilinearOpt.isconvex(m.quadconstr[end])

   JuMP.@constraint(m, x^2 + y^2 <= z)
   @test MultilinearOpt.isconvex(m.quadconstr[end])

   JuMP.@constraint(m, x^2 + y^2 <= z^2)
   @test !MultilinearOpt.isconvex(m.quadconstr[end])

   JuMP.@constraint(m, -x^2 - y^2 >= -z)
   @test MultilinearOpt.isconvex(m.quadconstr[end])
end
