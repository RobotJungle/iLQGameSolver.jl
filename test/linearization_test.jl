using LinearAlgebra
using SparseArrays

include("2PlayerFunctions.jl")

@testset "Linearization" begin
    # Setup the problem
    dt = 0.1                    # Step size [s]
    tf = 10.0                    # Horizon [s]
    N = trunc(Int, tf/dt)         # Number of steps (knot points)

    NHor = N
    # Define cost matrices 
    nx = 4 
    nu = 2
    Nplayer = 2

    dmax = 2.0                  # Distance that both agents should keep between each other [m]
    ρ = 0.0    

    Nu = nu * Nplayer
    Nx = nx * Nplayer
    Q = sparse(zeros(Float32, Nx*Nplayer, Nx))
    Qn = sparse(zeros(Float32, Nx*Nplayer, Nx))
    R = sparse(zeros(Float32, Nu, Nu))

    tol = 1e-4
    game = iLQGameSolver.GameSetup(nx, nu, Nplayer, Q, R, Qn, dt, tf, NHor, dmax, ρ, tol)

    x = zeros(Nx)
    u = zeros(Nu)

    dynamics = iLQGameSolver.pointMass
    A, B = iLQGameSolver.linearDiscreteDynamics(game, dynamics,x,u)

    c = 0.1
    m₁ = 1.0
    m₂ = 1.0
    A1 = sparse([0 0 1 0; 0 0 0 1; 0 0 (-c/m₁) 0; 0 0 0 (-c/m₁)])
    A2 = sparse([0 0 1 0; 0 0 0 1; 0 0 (-c/m₂) 0; 0 0 0 (-c/m₂)])
    ATest = blockdiag(A1, A2)
    B1 = sparse([0 0; 0 0; (1/m₁) 0; 0 (1/m₁); 0 0; 0 0; 0 0; 0 0])  #Control Jacobian for point mass 1
    B2 = sparse([0 0; 0 0; 0 0; 0 0; 0 0; 0 0; (1/m₂) 0; 0 (1/m₂)])    #Control Jacobian for point mass 2
    BTest = cat(B1,B2,dims=2)
    Ad = game.dt .* ATest + I       #discretize (zero order hold)
    Bd = game.dt .*BTest;           #discrete (zero order hold)

    @test sparse(A) == Ad
    @test sparse(B) == Bd
end