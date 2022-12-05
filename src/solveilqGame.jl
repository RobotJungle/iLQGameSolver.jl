using LinearAlgebra
using SparseArrays
using StaticArrays
using InvertedIndices


"""
    GameSolver
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



function solveILQGame(game::GameSolver, dynamics, costf)
    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer
    dt = game.dt
    H = game.H
    ρ = game.ρ
    x₀ = game.x0
    umin = game.umin
    umax = game.umax

    Nx = Nplayer*nx
    Nu = Nplayer*nu

    # m1 = nu
    # m2 = nu  

    k_steps = trunc(Int, H/dt) 

    x̂ = zeros(k_steps, Nx) 
    û = zeros(k_steps, Nu)

    P = rand(k_steps, Nu, Nx)*0.01
    α = rand(k_steps, Nu)*0.01

    # Rollout players
    ##!!!Pass game struct instead!!!
    xₜ, uₜ = rolloutRK4(game, dynamics, x̂, û, P, α, 0.0)

    Aₜ = zeros(Float32, (k_steps, Nx, Nx))
    Bₜ = zeros(Float32, (k_steps, Nx, Nu)) # Added
    # B1ₜ = zeros(Float32, (Nx, m1, k_steps))
    # B2ₜ = zeros(Float32, (Nx, m2, k_steps))

    Qₜ = zeros(Float32, (k_steps, Nx*Nplayer, Nx))
    # Q1ₜ = zeros(Float32, (Nx, Nx, k_steps))
    # Q2ₜ = zeros(Float32, (Nx, Nx, k_steps))

    lₜ = zeros(Float32, (k_steps, Nx, Nplayer))
    # l1ₜ = zeros(Float32, (Nx, k_steps))
    # l2ₜ = zeros(Float32, (Nx, k_steps))

    Rₜ = zeros(Float32, (k_steps, Nu, Nu)) 
    # R11ₜ = zeros(Float32, (m1, m1, k_steps))
    # R12ₜ = zeros(Float32, (m1, m2, k_steps))
    # R22ₜ = zeros(Float32, (m2, m2, k_steps))
    # R21ₜ = zeros(Float32, (m2, m1, k_steps))

    rₜ = zeros(Float32, (k_steps, Nu, Nplayer))
    # r11ₜ = zeros(Float32, (m1, k_steps))
    # r12ₜ = zeros(Float32, (m1, k_steps))
    # r22ₜ = zeros(Float32, (m2, k_steps))
    # r21ₜ = zeros(Float32, (m2, k_steps))

    Q = game.Q
    R = game.R
    Qn = game.Qn
    
    converged = false

    βreg = 1.0
    while !converged
        converged = isConverged(xₜ, x̂, tol = 1e-2)
        total_cost = zeros(Nplayer) # Added

        for t = 1:(k_steps-1)
            
            Aₜ[t,:,:], Bₜ[t,:,:] = linearDiscreteDynamics(game, dynamics, xₜ[t,:], uₜ[t,:])

            # Player cost
            for i = 1:Nplayer
                
                Nxi = 1+(i-1)*Nx     # Player i's state start index
                Nxf = i*Nx           # Player i's state final index
                nui = 1+(i-1)*nu     # Player i's control start index
                nuf = i*nu           # Player i's control final index

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
            Nxi = 1+(i-1)*Nx     # Player i's state start index
            Nxf = i*Nx           # Player i's state final index
            nui = 1+(i-1)*nu     # Player i's control start index
            nuf = i*nu           # Player i's control final index

            costval, Qₜ[end,Nxi:Nxf,:], lₜ[end,:,i], Rₜ[end,nui:nuf,nui:nuf], 
            rₜ[end,nui:nuf,i], Rₜ[end,nui:nuf,Not(nui:nuf)], rₜ[end,Not(nui:nuf),i] = 
            quadraticizeCost(game, costf, i, Q[Nxi:Nxf,:], R[nui:nuf,nui:nuf], R[nui:nuf,Not(nui:nuf)], 
            Qn[Nxi:Nxf,:], xₜ[end,:], uₜ[end,nui:nuf], uₜ[end, Not(nui:nuf)], true)
            
            total_cost[i] += costval
        end

        P, α = lqGame!(game, Aₜ, Bₜ, Qₜ, lₜ, Rₜ, rₜ, k_steps)

        x̂ = xₜ
        û = uₜ

        # Rollout players with new control law
        xₜ, uₜ = rolloutRK4(game, dynamics, x̂, û, P, α, 0.5)

    end

    return xₜ, uₜ
end
