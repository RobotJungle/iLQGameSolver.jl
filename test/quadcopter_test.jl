using LinearAlgebra
using SparseArrays
using iLQGameSolver


@testset "Quadcopter Test" begin
    # Setup the problem

    # Setup the problem

    dt = 0.1                    # Step size [s]
    tf = 20.0                    # Horizon [s]
    N = Int(tf/dt)         # Number of steps (knot points)

    # Define cost matrices 
    nx = 9 
    nu = 4
    Nplayer = 2

    Nu = nu * Nplayer
    Nx = nx * Nplayer

    Q1 = sparse(zeros(Nx,Nx))     # State cost for agent 1
    Q1[1:nx,1:nx] .= 3.0*I(nx)
    Qn1 = Q1                    # Terminal cost for agent 1

    Q2 = sparse(zeros(Nx,Nx))     # State cost for agent 2
    Q2[nx+1:2*nx,nx+1:2*nx] .= 3.0*I(nx)
    Qn2 = Q2                    # Terminal cost for agent 2

    R11 = sparse(1.0*I(nu))              # Control cost for player 1
    R12 = sparse(0.0*I(nu))     # Control cost for player 1 associated with player 2's controls

    R21 = sparse(0.0*I(nu))     # Control cost for player 2 associated with player 1's controls
    R22 = sparse(1.0*I(nu))              # Contorl cost for player 2


    dmax = 2.0                  # Distance that both agents should keep between each other [m]
    ρ = 500.0                   # Penalty factor for violating the distance constraint

    # Q's are stacked vertically
    Q = sparse(zeros(Float32, Nx*Nplayer, Nx))
    # @show size([Q1; Q2; Q3]), size(Q)
    #Q .= [Q1; Q2]
    Q .= [Q1; Q2]

    # Qn's are stacked vertically
    Qn = sparse(zeros(Float32, Nx*Nplayer, Nx))
    #Qn .= [Qn1; Qn2]
    Qn .= [Qn1; Qn2]

    # R's are stacked as a matrix
    R = sparse(zeros(Float32, Nu, Nu))
    #R .= [R11 R12; R21 R22]
    R .= [R11 R12; R21 R22]

    NHor = 25
    tol = 1e-2

    game = iLQGameSolver.GameSetup(nx, nu, Nplayer, Q, R, Qn, dt, tf, NHor, dmax, ρ, tol)

    solver = iLQGameSolver.iLQSetup(Nx, Nu, Nplayer, NHor)

    solver.P = ones(NHor, Nu, Nx)*0.001
    solver.α = ones(NHor, Nu)*0.001;

    # Initial and final states

    x₀= [5.0; 5.0; 5.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 
    -5.0; -5.0; 5.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]        # Initial state

    xgoal = [  -5.0; -5.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 
            5.0; 5.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0];  # Final state

    # Input constraints
    umax = [10.0, pi, pi, pi, 
        10.0, pi, pi, pi]   

    umin = [-10.0, -pi, -pi, -pi, 
        -10.0, -pi, -pi, -pi] 

    ugoal = [0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0]     

    game.x0 .= x₀
    game.xf .= xgoal
    game.umin .= umin
    game.umax .= umax
    game.uf .= ugoal;

    X, U = iLQGameSolver.recedingHorizon(game, solver, iLQGameSolver.quadcopter, iLQGameSolver.costQuadcopter);

    xend = X[end,:,:]

    @test xend[1] ≈ xgoal[1] atol=1e-1 
    @test xend[2] ≈ xgoal[2] atol=1e0 
    @test xend[3] ≈ xgoal[3] atol=1e-1 
    @test xend[4] ≈ xgoal[4] atol=1e-1 
    @test xend[5] ≈ xgoal[5] atol=1e-1 
    @test xend[6] ≈ xgoal[6] atol=1e-1 
    @test xend[7] ≈ xgoal[7] atol=1e-1 
    @test xend[8] ≈ xgoal[8] atol=1e-1 
    @test xend[9] ≈ xgoal[9] atol=1e0 
    @test xend[10] ≈ xgoal[10] atol=1e-1 
    @test xend[11] ≈ xgoal[11] atol=1e0 
    @test xend[12] ≈ xgoal[12] atol=1e-1 

end