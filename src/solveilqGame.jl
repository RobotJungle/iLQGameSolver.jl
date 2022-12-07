using LinearAlgebra
using SparseArrays
using StaticArrays
using InvertedIndices

mutable struct iLQStruct
    P::Array{Float64,3}
    α::Matrix{Float64}
    Aₜ::Array{Float32,3}
    Bₜ::Array{Float32,3}
    Qₜ::Array{Float32,3}
    lₜ::Array{Float32,3}
    Rₜ::Array{Float32,3}
    rₜ::Array{Float32,3}
    S::Matrix{Float64}
    Y::Matrix{Float64}
    Yα::Vector{Float64}
    x̂::Matrix{Float64}
    û::Matrix{Float64}
end

function iLQSetup(Nx::Int64, Nu::Int64, Nplayer::Int64, NHor::Int64)
    P = rand(NHor, Nu, Nx)*0.01
    α = rand(NHor, Nu)*0.01
    Aₜ = zeros(Float32, (NHor, Nx, Nx))
    Bₜ = zeros(Float32, (NHor, Nx, Nu)) # Added
    Qₜ = zeros(Float32, (NHor, Nx*Nplayer, Nx))
    lₜ = zeros(Float32, (NHor, Nx, Nplayer))
    Rₜ = zeros(Float32, (NHor, Nu, Nu)) 
    rₜ = zeros(Float32, (NHor, Nu, Nplayer))
    S = zeros(Nu, Nu)
    Y = zeros(Nu, Nx)
    Yα = zeros(Nu)
    x̂ = zeros(NHor, Nx) 
    û = zeros(NHor, Nu)
    iLQStruct(P, α, Aₜ, Bₜ, Qₜ, lₜ, Rₜ, rₜ, S, Y, Yα, x̂, û)
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

function solveILQGame(game, solver, dynamics, costf, kstart, x0, terminal)
    Nplayer = game.Nplayer
    NHor = game.NHor
    Q = game.Q
    R = game.R
    Qn = game.Qn
    tf = game.tf
    dt = game.dt
    N = trunc(Int, tf/dt)

    Aₜ = solver.Aₜ
    Bₜ = solver.Bₜ
    Qₜ = solver.Qₜ
    lₜ = solver.lₜ
    Rₜ = solver.Rₜ
    rₜ = solver.rₜ

    # Rollout players, to obtain Initial feasible trajectory
    xₜ, uₜ = rolloutRK4(game, solver, dynamics, x0, 0.0)

    converged = false

    βreg = 1.0 # Regularization parameter
    αscale = 1.0 # Linesearch parameter
    while !converged
        converged = isConverged(xₜ, solver.x̂, tol = 1e-2)
        total_cost = zeros(Nplayer) # Added

        for t = 1:(NHor-1)
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
        if terminal
            for i = 1:Nplayer

                Nxi, Nxf, nui, nuf = getPlayerIdx(game, i) # get player i's indices

                costval, Qₜ[end,Nxi:Nxf,:], lₜ[end,:,i], Rₜ[end,nui:nuf,nui:nuf], 
                rₜ[end,nui:nuf,i], Rₜ[end,nui:nuf,Not(nui:nuf)], rₜ[end,Not(nui:nuf),i] = 
                quadraticizeCost(game, costf, i, Q[Nxi:Nxf,:], R[nui:nuf,nui:nuf], R[nui:nuf,Not(nui:nuf)], 
                Qn[Nxi:Nxf,:], xₜ[end,:], uₜ[end,nui:nuf], uₜ[end, Not(nui:nuf)], true)
                
                total_cost[i] += costval
            end

            # indx = kstart+NHor
            # for i = 1:Nplayer

            #     Nxi, Nxf, nui, nuf = getPlayerIdx(game, i) # get player i's indices
    
            #     costval, Qₜ[indx,Nxi:Nxf,:], lₜ[indx,:,i], Rₜ[indx,nui:nuf,nui:nuf], 
            #     rₜ[indx,nui:nuf,i], Rₜ[indx,nui:nuf,Not(nui:nuf)], rₜ[indx,Not(nui:nuf),i] = 
            #     quadraticizeCost(game, costf, i, Q[Nxi:Nxf,:], R[nui:nuf,nui:nuf], R[nui:nuf,Not(nui:nuf)], 
            #     Qn[Nxi:Nxf,:], xₜ[indx,:], uₜ[indx,nui:nuf], uₜ[indx, Not(nui:nuf)], true)
                
            #     total_cost[i] += costval
            # end
        end

        lqGame!(game, solver)

        solver.x̂ = xₜ
        solver.û = uₜ

        # Rollout players with new control law
        xₜ, uₜ = rolloutRK4(game, solver, dynamics, x0, αscale)
        # xₜ[1,:,:] = xₜ[kstart,:,:]
        # uₜ[1,:,:] = uₜ[kstart,:,:]
    end

    return xₜ, uₜ#xₜ[2,:,:], uₜ[1,:,:]
end
