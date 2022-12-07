using LinearAlgebra
using SparseArrays
using InvertedIndices


@testset "solveilqGame Test" begin
   
       # Setup the problem

        dt = 0.1                    # Step size [s]
        tf = 10.0                    # Horizon [s]
        N = trunc(Int, tf/dt)         # Number of steps (knot points)

        # Define cost matrices 
        nx = 4 
        nu = 2
        Nplayer = 3

        Nu = nu * Nplayer
        Nx = nx * Nplayer

        Q1 = sparse(zeros(Nx,Nx))     # State cost for agent 1
        Q1[1:nx,1:nx] .= 3.0*I(nx)
        Qn1 = Q1                    # Terminal cost for agent 1

        Q2 = sparse(zeros(Nx,Nx))     # State cost for agent 2
        Q2[nx+1:2*nx,nx+1:2*nx] .= 3.0*I(nx)
        Qn2 = Q2                    # Terminal cost for agent 2

        Q3 = sparse(zeros(Nx,Nx))     # State cost for agent 2
        Q3[2*nx+1:3*nx,2*nx+1:3*nx] .= 3.0*I(nx)
        Qn3 = Q3                    # Terminal cost for agent 2

        R11 = sparse(1.0*I(2))              # Control cost for player 1
        R12 = sparse(0.0*I(2))     # Control cost for player 1 associated with player 2's controls
        R13 = sparse(0.0*I(2))     # Control cost for player 2 associated with player 1's controls
        R21 = sparse(0.0*I(2))     # Control cost for player 2 associated with player 1's controls
        R22 = sparse(1.0*I(2))              # Contorl cost for player 2
        R23 = sparse(0.0*I(2))     # Control cost for player 2 associated with player 1's controls
        R31 = sparse(0.0*I(2))     # Control cost for player 2 associated with player 1's controls
        R32 = sparse(0.0*I(2))     # Control cost for player 2 associated with player 1's controls
        R33 = sparse(1.0*I(2))     # Control cost for player 2 associated with player 1's controls

        dmax = 2.0                  # Distance that both agents should keep between each other [m]
        ρ = 5000.0                   # Penalty factor for violating the distance constraint

        # Q's are stacked vertically
        Q = sparse(zeros(Float32, Nx*Nplayer, Nx))
        # @show size([Q1; Q2; Q3]), size(Q)
        #Q .= [Q1; Q2]
        Q .= [Q1; Q2; Q3]

        # Qn's are stacked vertically
        Qn = sparse(zeros(Float32, Nx*Nplayer, Nx))
        #Qn .= [Qn1; Qn2]
        Qn .= [Qn1; Qn2; Qn3]

        # R's are stacked as a matrix
        R = sparse(zeros(Float32, Nu, Nu))
        #R .= [R11 R12; R21 R22]
        R .= [R11 R12 R13; R21 R22 R23; R31 R32 R33]

        NHor = N
        tol = 1e-2

        game = iLQGameSolver.GameSetup(nx, nu, Nplayer, Q, R, Qn, dt, tf, NHor, dmax, ρ, tol)

        solver = iLQGameSolver.iLQSetup(Nx, Nu, Nplayer, NHor)

        solver.P = ones(NHor, Nu, Nx)*0.03
        solver.α = ones(NHor, Nu)*0.03

        # Initial and final states
        # x₁, y₁, ̇x₁, ̇y₁, x₂, y₂, ̇x₂, ̇y₂       

        x₀= [   5.0; 0.0; 0.0; 0.0; 
        0.0; 5.0; 0.0; 0.0; 
        0.0; 0.0; 0.0; 0.0] 
        # Initial state

        xgoal = [   5.0; 10.0; 0.0; 0.0; 
        10.0; 5.0; 0.0; 0.0; 
        10.0; 10.0; 0.0; 0.0]   
        # Final state

        # Input constraints
        umax = [2.0, 2.0, 
        2.0, 2.0, 
        2.0, 2.0]   

        umin = [-2.0, -2.0, 
        -2.0, -2.0, 
        -2.0, -2.0]

        ugoal = [   0.0, 0.0, 
        0.0, 0.0,  
        0.0, 0.0]     

        game.x0 .= x₀
        game.xf .= xgoal
        game.umin .= umin
        game.umax .= umax
        game.uf .= ugoal;
        
        X,U = iLQGameSolver.solveILQGame(game, solver, iLQGameSolver.pointMass, iLQGameSolver.costPointMass, x₀, true);

        xend = X[end,:,:]

        @test xend[1] ≈ xgoal[1] atol=1e-2 
        @test xend[2] ≈ xgoal[2] atol=1e-2 
        @test xend[3] ≈ xgoal[3] atol=1e-2 
        @test xend[4] ≈ xgoal[4] atol=1e-2 
        @test xend[5] ≈ xgoal[5] atol=1e-2 
        @test xend[6] ≈ xgoal[6] atol=1e-2 
        @test xend[7] ≈ xgoal[7] atol=1e-2 
        @test xend[8] ≈ xgoal[8] atol=1e-2 
        @test xend[9] ≈ xgoal[9] atol=1e-2 
        @test xend[10] ≈ xgoal[10] atol=1e-2 
        @test xend[11] ≈ xgoal[11] atol=1e-2 
        @test xend[12] ≈ xgoal[12] atol=1e-2 
end