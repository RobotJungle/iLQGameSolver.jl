using LinearAlgebra
using SparseArrays
using InvertedIndices

include("2PlayerFunctions.jl")

@testset "LQGame" begin
    # Setup the problem
    dt = 0.1                    # Step size [s]
    tf = 10.0                    # Horizon [s]
    N = trunc(Int, tf/dt)
    NHor = N

    # Define cost matrices 
    nx = 4 
    nu = 2
    Nplayer = 2

    dmax = 2.0                  # Distance that both agents should keep between each other [m]
    ρ = 0.0    

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

    lₜ = zeros(Float32, (NHor, Nx, Nplayer))

    l1ₜ = lₜ[:,:,1]
    l2ₜ = lₜ[:,:,1]

    rₜ = zeros(Float32, (NHor, Nu, Nplayer))

    r11ₜ = rₜ[:,1:nu,1]
    r12ₜ = rₜ[:,Not(1:nu),1]
    r22ₜ = rₜ[:,1:nu,2]
    r21ₜ = rₜ[:,Not(1:nu),2]

    R11 = sparse(1.0*I(2))              # Control cost for player 1
    R12 = sparse(0.0*I(2))     # Control cost for player 1 associated with player 2's controls
    R21 = sparse(0.0*I(2))     # Control cost for player 2 associated with player 1's controls
    R22 = sparse(1.0*I(2))              # Contorl cost for player 2

    R = sparse(zeros(Float32, Nu, Nu))
    R .= [R11 R12; R21 R22]

    game = iLQGameSolver.GameSetup(nx, nu, Nplayer, Q, R, Qn, dt, tf, NHor, dmax, ρ)
    solver = iLQGameSolver.iLQSetup(Nx, Nu, Nplayer, NHor)

    x = zeros(Nx)
    u = zeros(Nu)

    dynamics = iLQGameSolver.pointMass
    A, B = iLQGameSolver.linearDiscreteDynamics(game, dynamics,x,u)

    Aₜ = zeros(Float32, (NHor, Nx, Nx))
    Bₜ = zeros(Float32, (NHor, Nx, Nu))
    Qₜ = zeros(Float32, (NHor, Nx*Nplayer, Nx))
    Rₜ = zeros(Float32, (NHor, Nu, Nu))
    Q1ₜ = zeros(Float32, (NHor, Nx, Nx))
    Q2ₜ = zeros(Float32, (NHor, Nx, Nx))
    R11ₜ = zeros(Float32, (NHor, nu, nu))
    R12ₜ = zeros(Float32, (NHor, nu, nu))
    R22ₜ = zeros(Float32, (NHor, nu, nu))
    R21ₜ = zeros(Float32, (NHor, nu, nu))
    for t in 1:NHor
        Aₜ[t,:,:] = A
        Bₜ[t,:,:] = B
        Qₜ[t,:,:] = Q
        Rₜ[t,:,:] = R
        Q1ₜ[t,:,:] = Q1
        Q2ₜ[t,:,:] = Q2
        R11ₜ[t,:,:] = R11
        R12ₜ[t,:,:] = R12
        R21ₜ[t,:,:] = R21
        R22ₜ[t,:,:] = R22
        solver.Aₜ[t,:,:] = A
        solver.Bₜ[t,:,:] = B
        solver.Qₜ[t,:,:] = Q
        solver.Rₜ[t,:,:] = R
    end

    solver.lₜ = zeros(Float32, (NHor, Nx, Nplayer))
    solver.rₜ = zeros(Float32, (NHor, Nu, Nplayer))


    B1ₜ = Bₜ[:,:,1:nu]
    B2ₜ = Bₜ[:,:,nu+1:end]

    iLQGameSolver.lqGame!(game, solver)

    P₁, P₂, α₁, α₂ = lqGame2P!(Aₜ, B1ₜ, B2ₜ, Q1ₜ, Q2ₜ, l1ₜ, l2ₜ, R11ₜ, R12ₜ, R21ₜ, R22ₜ, r11ₜ, r22ₜ, r12ₜ, r21ₜ, NHor)

    @test solver.P[:,1:nu,:] == P₁
    @test solver.α[:,1:nu] == α₁
    @test solver.P[:,nu+1:end,:] == P₂
    @test solver.α[:,nu+1:end] == α₂

end



# @testset "LQGame" begin
#     # Setup the problem
#     dt = 0.1                    # Step size [s]
#     tf = 10.0                    # Horizon [s]
#     N = trunc(Int, tf/dt)         # Number of steps (knot points)

#     NHor = N
#     # Define cost matrices 
#     nx = 4 
#     nu = 2
#     Nplayer = 2

#     dmax = 2.0                  # Distance that both agents should keep between each other [m]
#     ρ = 500.0    

#     Nu = nu * Nplayer
#     Nx = nx * Nplayer

#     Q1 = sparse(zeros(Nx,Nx))     # State cost for agent 1
#     Q1[1:nx,1:nx] .= 3.0*I(nx)
#     Qn1 = Q1                    # Terminal cost for agent 1

#     Q2 = sparse(zeros(Nx,Nx))     # State cost for agent 2
#     Q2[nx+1:2*nx,nx+1:2*nx] .= 3.0*I(nx)
#     Qn2 = Q2                    # Terminal cost for agent 2

#     Q = sparse(zeros(Float32, Nx*Nplayer, Nx))
#     Q .= [Q1; Q2]

#     Qn = sparse(zeros(Float32, Nx*Nplayer, Nx))
#     Qn .= [Qn1; Qn2]

#     lₜ = zeros(Float32, (NHor, Nx, Nplayer))
#     l1ₜ = lₜ[:,:,1]
#     l2ₜ = lₜ[:,:,1]

#     rₜ = zeros(Float32, (NHor, Nu, Nplayer))
#     r11ₜ = rₜ[:,1:nu,1]
#     r12ₜ = rₜ[:,Not(1:nu),1]
#     r22ₜ = rₜ[:,1:nu,2]
#     r21ₜ = rₜ[:,Not(1:nu),2]

#     R11 = sparse(1.0*I(2))              # Control cost for player 1
#     R12 = sparse(0.0*I(2))     # Control cost for player 1 associated with player 2's controls
#     R21 = sparse(0.0*I(2))     # Control cost for player 2 associated with player 1's controls
#     R22 = sparse(1.0*I(2))              # Contorl cost for player 2

#     R = sparse(zeros(Float32, Nu, Nu))
#     R .= [R11 R12; R21 R22]

#     game = iLQGameSolver.GameSetup(nx, nu, Nplayer, Q, R, Qn, dt, tf, NHor, dmax, ρ)
#     solver = iLQGameSolver.iLQSetup(Nx, Nu, Nplayer, NHor)

#     x = zeros(Nx)
#     u = zeros(Nu)

#     dynamics = iLQGameSolver.pointMass
#     A, B = iLQGameSolver.linearDiscreteDynamics(game, dynamics,x,u)

#     Aₜ = zeros(Float32, (NHor, Nx, Nx))
#     Bₜ = zeros(Float32, (NHor, Nx, Nu))
#     Qₜ = zeros(Float32, (NHor, Nx*Nplayer, Nx))
#     Rₜ = zeros(Float32, (NHor, Nu, Nu))
#     Q1ₜ = zeros(Float32, (NHor, Nx, Nx))
#     Q2ₜ = zeros(Float32, (NHor, Nx, Nx))
#     R11ₜ = zeros(Float32, (NHor, nu, nu))
#     R12ₜ = zeros(Float32, (NHor, nu, nu))
#     R22ₜ = zeros(Float32, (NHor, nu, nu))
#     R21ₜ = zeros(Float32, (NHor, nu, nu))
#     for t in 1:NHor
#         Aₜ[t,:,:] = A
#         Bₜ[t,:,:] = B
#         Qₜ[t,:,:] = Q
#         Rₜ[t,:,:] = R
#         solver.Aₜ[t,:,:] = A
#         solver.Bₜ[t,:,:] = B
#         solver.Qₜ[t,:,:] = Q
#         solver.Rₜ[t,:,:] = R
#         Q1ₜ[t,:,:] = Q1
#         Q2ₜ[t,:,:] = Q2
#         R11ₜ[t,:,:] = R11
#         R12ₜ[t,:,:] = R12
#         R21ₜ[t,:,:] = R21
#         R22ₜ[t,:,:] = R22
#     end

#     B1ₜ = Bₜ[:,:,1:nu]
#     B2ₜ = Bₜ[:,:,nu+1:end]

#     P, α = iLQGameSolver.lqGame!(game, Aₜ, Bₜ, Qₜ, lₜ, Rₜ, rₜ)

#     P₁, P₂, α₁, α₂ = lqGame2P!(Aₜ, B1ₜ, B2ₜ, Q1ₜ, Q2ₜ, l1ₜ, l2ₜ, R11ₜ, R12ₜ, R21ₜ, R22ₜ, r11ₜ, r22ₜ, r12ₜ, r21ₜ, NHor)

#     @show P
#     @show P₁

#     # @test solver.P[:,1:nu,:] == P₁
#     # @test solver.α[:,1:nu] == α₁
#     # @test solver.P[:,nu+1:end,:] == P₂
#     # @test solver.α[:,nu+1:end] == α₂

# end