using LinearAlgebra
using SparseArrays

@testset "Linearization" begin


    
    param = iLQGameSolver.MultiAgentDynamics(4, 2, 2, 0.1)

    x = zeros(param.Nplayers*param.nx)
    u = zeros(param.Nplayers*param.nu)

    dynamics = iLQGameSolver.point_mass
    A, B = iLQGameSolver.lin_dyn_discreteRH(dynamics,param,x,u)
    @show size(A)
    c = 0.1
    m₁ = 1.0
    m₂ = 1.0
    A1 = sparse([0 0 1 0; 0 0 0 1; 0 0 (-c/m₁) 0; 0 0 0 (-c/m₁)])
    A2 = sparse([0 0 1 0; 0 0 0 1; 0 0 (-c/m₂) 0; 0 0 0 (-c/m₂)])
    ATest = blockdiag(A1, A2)
    Ad = param.dt .* ATest + I    #discretize (zero order hold)
    @test sparse(A) == Ad

end