using LinearAlgebra
using SparseArrays
using StaticArrays
using InvertedIndices

"""
    GameSolver 

Struct for holding the game parameters. 

    nx: Number of states for each player
    nu: Number of control inputs for each player 
    Nplayer: Total number of players 
    x0: Intial state vector 
    xf: Final state goal vector
    umin: Minimum control input vector
    umax: Maximum control input vector
    uf: Final control input goal vector
    Q: State error cost matrix
    R: Control input matrix
    Qn: Terminal state error cost matrix
    dt: Timestep in seconds
    H: Time horizon in seconds
    dmax: Minimum distance between all agents in meters
    ρ: Penalty factor for violating distance constraint
"""
struct GameSolver
    nx::Int64
    nu::Int64
    Nplayer::Int64
    x0::Vector{Float64}
    xf::Vector{Float64}
    umin::Vector{Float64}
    umax::Vector{Float64}
    uf::Vector{Float64}
    Q::SparseMatrixCSC{Float32,Int}
    R::SparseMatrixCSC{Float32,Int}
    Qn::SparseMatrixCSC{Float32,Int}
    dt::Float64
    H::Float64
    dmax::Float64
    ρ::Float64
end


"""
    GameSetup(n,m,Nplayer,Q,R,Qn,dt,H,dmax,ρ)

Generates a `LQGameSolver` that uses solveiLQGames to solve the problem.
"""
function GameSetup(nx::Int64, nu::Int64, Nplayer::Int64, Q::SparseMatrixCSC{Float32,Int}, 
                    R::SparseMatrixCSC{Float32,Int}, Qn::SparseMatrixCSC{Float32,Int}, 
                    dt::Float64, H::Float64, dmax::Float64, ρ::Float64)
    Nx = Nplayer*nx    
    Nu = Nplayer*nu
    x0 = zeros(Nx)
    xf = zeros(Nx)
    umin = zeros(Nu)
    umax = zeros(Nu)
    uf = zeros(Nu)
    GameSolver(nx, nu, Nplayer, x0, xf, umin, umax, uf, Q, R, Qn, dt, H, dmax, ρ)
end

"""
    solveILQGame(game, dynamics, costf)

Solves the LQ game iteratively.

Inputs:
    game: GameSolver struct (see solveilqGame.jl)
    dynamics: Dynamics function for entire game
    costf: Cost function for the game

Outputs:
    xₜ: States at each timestep for the converged solution (k_steps, Nx)
    uₜ: Control inputs at each timestep for the converged solution (k_steps, Nu)
"""

function solveILQGame(game::GameSolver, dynamics, costf)
    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer
    dt = game.dt
    H = game.H
    ρ = game.ρ

    Nx = Nplayer*nx
    Nu = Nplayer*nu

    k_steps = trunc(Int, H/dt) 

    x̂ = zeros(k_steps, Nx) 
    û = zeros(k_steps, Nu)

    # Initialize Random gains
    P = rand(k_steps, Nu, Nx)*0.01
    α = rand(k_steps, Nu)*0.01

    # Rollout players, to obtain Initial feasible trajectory
    xₜ, uₜ = rolloutRK4(game, dynamics, x̂, û, P, α, 0.0)

    Aₜ = zeros(Float32, (k_steps, Nx, Nx))
    Bₜ = zeros(Float32, (k_steps, Nx, Nu)) # Added

    Qₜ = zeros(Float32, (k_steps, Nx*Nplayer, Nx))

    lₜ = zeros(Float32, (k_steps, Nx, Nplayer))

    Rₜ = zeros(Float32, (k_steps, Nu, Nu)) 

    rₜ = zeros(Float32, (k_steps, Nu, Nplayer))

    Q = game.Q
    R = game.R
    Qn = game.Qn
    
    converged = false

    βreg = 1.0 # Regularization parameter
    αscale = 0.5 # Linesearch parameter
    while !converged
        converged = isConverged(xₜ, x̂, tol = 1e-2)
        total_cost = zeros(Nplayer) # Added

        for t = 1:(k_steps-1)
            # Obtain linearized discrete dynamics
            Aₜ[t,:,:], Bₜ[t,:,:] = linearDiscreteDynamics(game, dynamics, xₜ[t,:], uₜ[t,:])

            # Obtain quadraticized cost function 
            for i = 1:Nplayer

                Nxi, Nxf, nui, nuf = getPlayerIdx(game, i) # get player i's indices

                costval, Qₜ[t,Nxi:Nxf,:], lₜ[t,:,i], Rₜ[t,nui:nuf,nui:nuf], 
                rₜ[t,nui:nuf,i], Rₜ[t,nui:nuf,Not(nui:nuf)], rₜ[t,Not(nui:nuf),i] = 
                quadraticizeCost(game, costf, i, Q[Nxi:Nxf,:], R[nui:nuf,nui:nuf], R[nui:nuf,Not(nui:nuf)], 
                Qn[Nxi:Nxf,:], xₜ[t,:], uₜ[t,nui:nuf], uₜ[t, Not(nui:nuf)], false)

                while !isposdef(Qₜ[t,Nxi:Nxf,:])
                    Qₜ[t,Nxi:Nxf,:] += βreg*I
                end
                total_cost[i] += costval
            end
        
        end

        for i = 1:Nplayer

            Nxi, Nxf, nui, nuf = getPlayerIdx(game, i) # get player i's indices

            costval, Qₜ[end,Nxi:Nxf,:], lₜ[end,:,i], Rₜ[end,nui:nuf,nui:nuf], 
            rₜ[end,nui:nuf,i], Rₜ[end,nui:nuf,Not(nui:nuf)], rₜ[end,Not(nui:nuf),i] = 
            quadraticizeCost(game, costf, i, Q[Nxi:Nxf,:], R[nui:nuf,nui:nuf], R[nui:nuf,Not(nui:nuf)], 
            Qn[Nxi:Nxf,:], xₜ[end,:], uₜ[end,nui:nuf], uₜ[end, Not(nui:nuf)], true)
            
            total_cost[i] += costval
        end

        P, α = lqGame!(game, Aₜ, Bₜ, Qₜ, lₜ, Rₜ, rₜ)

        x̂ = xₜ
        û = uₜ

        # Rollout players with new control law
        xₜ, uₜ = rolloutRK4(game, dynamics, x̂, û, P, α, αscale)

    end

    return xₜ, uₜ
end
