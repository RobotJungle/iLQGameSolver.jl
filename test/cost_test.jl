using LinearAlgebra
using SparseArrays
using InvertedIndices

include("2PlayerFunctions.jl")

@testset "Cost" begin
    # Setup the problem
    dt = 0.1                    # Step size [s]
    tf = 10.0                    # Horizon [s]
    N = trunc(Int, tf/dt)         # Number of steps (knot points)

    NHor = N

    # Define cost matrices 
    nx = 4 
    nu = 2
    Nplayer = 2

    dmax = 3.0                  # Distance that both agents should keep between each other [m]
    ρ = 500.0    

    Nu = nu * Nplayer
    Nx = nx * Nplayer

    Q1 = sparse(zeros(Nx,Nx))     # State cost for agent 1
    Q1[1:nx,1:nx] .= 3.0*I(nx)
    Qn1 = Q1                    # Terminal cost for agent 1

    Q2 = sparse(zeros(Nx,Nx))     # State cost for agent 2
    Q2[nx+1:2*nx,nx+1:2*nx] .= 1.0*I(nx)
    Qn2 = Q2                    # Terminal cost for agent 2

    Q = sparse(zeros(Float32, Nx*Nplayer, Nx))
    Q .= [Q1; Q2]

    Qn = sparse(zeros(Float32, Nx*Nplayer, Nx))
    Qn .= [Qn1; Qn2]

    R11 = sparse(1.0*I(2))              # Control cost for player 1
    R12 = sparse(0.0*I(2))     # Control cost for player 1 associated with player 2's controls
    R21 = sparse(0.0*I(2))     # Control cost for player 2 associated with player 1's controls
    R22 = sparse(1.0*I(2))              # Contorl cost for player 2

    R = sparse(zeros(Float32, Nu, Nu))
    R .= [R11 R12; R21 R22]

    Qₜ = zeros(Float32, (Nx*Nplayer, Nx))
    lₜ = zeros(Float32, (Nx, Nplayer))
    Rₜ = zeros(Float32, (Nu, Nu)) 
    rₜ = zeros(Float32, (Nu, Nplayer))

    game = iLQGameSolver.GameSetup(nx, nu, Nplayer, Q, R, Qn, dt, tf, NHor, dmax, ρ)
    solver = iLQGameSolver.iLQSetup(Nx, Nu, Nplayer, NHor)
    # Initial state 
    x₀ = [5.0; 0.0; 0.0; 0.0; 
         0.0; 5.0; 0.0; 0.0]  
       
    # Final state
    xgoal = [5.0; 10.0; 0.0; 0.0; 
            10.0; 5.0; 0.0; 0.0] 

    # Input constraints
    umax = [2.0, 2.0, 
            2.0, 2.0] 

    umin = [-2.0, -2.0, 
            -2.0, -2.0] 

    ugoal = [0.0, 0.0, 
             0.0, 0.0]      

    game.x0 .= x₀
    game.xf .= xgoal
    game.umin .= umin
    game.umax .= umax
    game.uf .= ugoal;

    x = x₀
    u = [1.0, 2.0, 4.0, 5.0]
    u1 = u[1:2]
    u2 = u[3:4]
    u1goal = ugoal[1:2]
    u2goal = ugoal[3:4]  

    costRH = zeros(Nplayer)
    costRHFinal = zeros(Nplayer)
    costQuadGen = zeros(Nplayer)
    costQuadGenFinal = zeros(Nplayer)
    for i in 1:Nplayer
        Nxi = 1+(i-1)*Nx     # Player i's state start index
        Nxf = i*Nx           # Player i's state final index
        nui = 1+(i-1)*nu     # Player i's control start index
        nuf = i*nu           # Player i's control final index

        costRH[i] = iLQGameSolver.costPointMass(game, i, Q[Nxi:Nxf,:], R[nui:nuf,nui:nuf], 
        R[nui:nuf,Not(nui:nuf)], Qn[Nxi:Nxf,:], x, u[nui:nuf], u[Not(nui:nuf)], false)

        costRHFinal[i] = iLQGameSolver.costPointMass(game, i, Q[Nxi:Nxf,:], R[nui:nuf,nui:nuf], 
        R[nui:nuf,Not(nui:nuf)], Qn[Nxi:Nxf,:], x, u[nui:nuf], u[Not(nui:nuf)], true)

        costQuadGen[i], Qₜ[Nxi:Nxf,:], lₜ[:,i], Rₜ[nui:nuf,nui:nuf], rₜ[nui:nuf,i], Rₜ[nui:nuf,Not(nui:nuf)], rₜ[Not(nui:nuf),i] = 
        iLQGameSolver.quadraticizeCost(game, iLQGameSolver.costPointMass, i, Q[Nxi:Nxf,:], R[nui:nuf,nui:nuf], R[nui:nuf,Not(nui:nuf)], 
        Qn[Nxi:Nxf,:], x, u[nui:nuf], u[Not(nui:nuf)], false)
    end

    ### Calculate the costs individually
    cost = zeros(Nplayer)
    costFinal = zeros(Nplayer)
    costQuad = zeros(Nplayer)
    costQuadFinal = zeros(Nplayer)

    # Player 1
    cost[1] = costPointMass2P(Q1, R11, R12, Qn1, x, u1, u2, xgoal, u1goal, u2goal, game.dmax, game.ρ, false)
    costFinal[1] = costPointMass2P(Q1, R11, R12, Qn1, x, u1, u2, xgoal, u1goal, u2goal, game.dmax, game.ρ, true)
    costQuad[1], Q̂1, l̂1, R̂11, r̂11, R̂12, r̂12 = quadraticizeCost2P(costPointMass2P, Q1, R11, R12, Qn1, x, u1, u2, xgoal, u1goal, u2goal, game.dmax, game.ρ, false)

    # Player 2
    cost[2] = costPointMass2P(Q2, R22, R21, Qn2, x, u2, u1, xgoal, u2goal, u1goal, game.dmax, game.ρ, false)
    costFinal[2] = costPointMass2P(Q2, R22, R21, Qn2, x, u2, u1, xgoal, u2goal, u1goal, game.dmax, game.ρ, true)
    costQuad[2], Q̂2, l̂2, R̂22, r̂22, R̂21, r̂21 = quadraticizeCost2P(costPointMass2P, Q2, R22, R21, Qn2, x, u2, u1, xgoal, u2goal, u1goal, game.dmax, game.ρ, false)

    @test costRH == cost
    @test costRHFinal == costFinal
    @test costQuadGen == costQuad
    @test Qₜ[1:Nx,:] == Q̂1
    @test Qₜ[Nx+1:end,:] == Q̂2
    @test Rₜ[1:nu,1:nu] == R̂11
    @test Rₜ[nu+1:end,nu+1:end] == R̂22
    # Most likely that the rest are also correct if the 7 tests pass ...
end