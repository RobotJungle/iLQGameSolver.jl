using LinearAlgebra
using SparseArrays
using StaticArrays


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
    Q::Vector{SparseMatrixCSC{Float64,Int}}
    R::Vector{SparseMatrixCSC{Float64,Int}}
    Qn::Vector{SparseMatrixCSC{Float64,Int}}
    dt::Float64
    H::Float64
    dmax::Float64
    ρ::Float64
end


"""
    GameSetup(n,m,Nplayer,Q,R,Qn,dt,H,dmax,ρ)

Generates a `LQGameSolver` that uses solveiLQGames to solve the problem.
"""
function GameSetup(nx::Int64, nu::Int64, Nplayer::Int64, Q::Vector{SparseMatrixCSC{Float64,Int}}, 
                    R::Vector{SparseMatrixCSC{Float64,Int}}, Qn::Vector{SparseMatrixCSC{Float64,Int}}, 
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



function solveILQGameRH(game::GameSolver, dynamics, costf)

    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer
    dt = game.dt
    H = game.H
    dmax = game.dmax
    ρ = game.ρ
    x₀ = game.x0
    umin = game.umin
    umax = game.umax

    Nx = Nplayer*nx
    Nu = Nplayer*nu

    m1 = nu
    m2 = nu  

    k_steps = trunc(Int, game.H/dt) 

    x̂ = zeros(k_steps, Nx) 
    û = zeros(k_steps, Nu)

    P = rand(Nu, Nx, k_steps)*0.01
    α = rand(Nu, k_steps)*0.01

    # Rollout players
    xₜ, uₜ = Rollout_RK4(dynamics, x₀, x̂, û, game.umin, game.umax, game.H, game.dt, P, α, 0.0)

    Aₜ = zeros(Float32, (Nx, Nx, k_steps))
    Bₜ = zeros(Float32, (Nx, Nu, k_steps)) # Added

    B1ₜ = zeros(Float32, (Nx, m1, k_steps))
    B2ₜ = zeros(Float32, (Nx, m2, k_steps))

    Q1ₜ = zeros(Float32, (Nx, Nx, k_steps))
    Q2ₜ = zeros(Float32, (Nx, Nx, k_steps))

    l1ₜ = zeros(Float32, (Nx, k_steps))
    l2ₜ = zeros(Float32, (Nx, k_steps))

    # R[timestep in k_steps][index of matrix in vector]
    Rₜ = [game.R for _ = 1:k_steps] # Added
    Qₜ = [game.Q for _ = 1:k_steps] # Added
    rₜ = zeros(Float32, (Nu, k_steps)) # Added 25x slower than zeros()

    R11ₜ = zeros(Float32, (m1, m1, k_steps))
    R12ₜ = zeros(Float32, (m1, m2, k_steps))
    R22ₜ = zeros(Float32, (m2, m2, k_steps))
    R21ₜ = zeros(Float32, (m2, m1, k_steps))

    r11ₜ = zeros(Float32, (m1, k_steps))
    r12ₜ = zeros(Float32, (m1, k_steps))
    r22ₜ = zeros(Float32, (m2, k_steps))
    r21ₜ = zeros(Float32, (m2, k_steps))

    ### Hardcode:
    Q1 = game.Q[1]
    Q2 = game.Q[2]

    Qn1 = game.Qn[1]
    Qn2 = game.Qn[2]

    R11 = game.R[1]
    R12 = game.R[2]
    R21 = game.R[3]
    R22 = game.R[4]

    u1goal = game.uf[1:nu]
    u2goal = game.uf[nu+1:nu*Nplayer]

    xgoal = game.xf

    converged = false
 
    βreg = 1.0
    while !converged
        converged = isConverged(xₜ, x̂, tol = 1e-2)
        total_cost = zeros(Nplayer) # Added
        total_cost1 = 0
        total_cost2 = 0
        u1ₜ = uₜ[:, 1:m1]
        u2ₜ = uₜ[:, m1+1:m1+m2]
        for t = 1:(k_steps-1)
            
            A, B = lin_dyn_discrete(dynamics, xₜ[t,:], uₜ[t,:], dt)

            Aₜ[:,:,t] = A
            B1ₜ[:,:,t] = B[:, 1:m1]
            B2ₜ[:,:,t] = B[:, m1+1:m1+m2] #end

            #Player 1 cost
            costval1, Q1ₜ[:,:,t], l1ₜ[:,t], R11ₜ[:,:,t], r11ₜ[:,t], R12ₜ[:,:,t], r12ₜ[:,t] = 
            quadratic_cost(costf, Q1, R11, R12, Qn1, xₜ[t,:], u1ₜ[t,:], u2ₜ[t,:], xgoal, u1goal, u2goal, dmax, ρ, false)
            
            #Player 2 cost
            costval2, Q2ₜ[:,:,t], l2ₜ[:,t], R22ₜ[:,:,t], r22ₜ[:,t], R21ₜ[:,:,t], r21ₜ[:,t] = 
            quadratic_cost(costf, Q2, R22, R21, Qn2, xₜ[t,:], u2ₜ[t,:], u1ₜ[t,:], xgoal, u2goal, u1goal, dmax, ρ, false)
            
            # Regularization
            while !isposdef(Q1ₜ[:,:,t])
                Q1ₜ[:,:,t] = Q1ₜ[:,:,t] + βreg*I
            end
            while !isposdef(Q2ₜ[:,:,t])
                Q2ₜ[:,:,t] = Q2ₜ[:,:,t] + βreg*I
            end

            total_cost1 += costval1
            total_cost2 += costval2
        end
        #Player 1 Terminal cost
        costval1, Q1ₜ[:,:,end], l1ₜ[:,end], R11ₜ[:,:,end], r11ₜ[:,end], R12ₜ[:,:,end], r12ₜ[:,end] = 
        quadratic_cost(costf, Q1, R11, R12, Qn1, xₜ[end,:], u1ₜ[end,:], u2ₜ[end,:], xgoal, u1goal, u2goal, dmax, ρ, true)
        
        #Player 2 Terminal cost
        costval2, Q2ₜ[:,:,end], l2ₜ[:,end], R22ₜ[:,:,end], r22ₜ[:,end], R21ₜ[:,:,end], r21ₜ[:,end] = 
        quadratic_cost(costf, Q2, R22, R21, Qn2, xₜ[end,:], u2ₜ[end,:], u2ₜ[end,:], xgoal, u2goal, u1goal, dmax, ρ, true)

        total_cost1 += costval1
        total_cost2 += costval2

        P₁, P₂, α₁, α₂ = lqGame!(Aₜ, B1ₜ, B2ₜ, Q1ₜ, Q2ₜ, l1ₜ, l2ₜ, R11ₜ, R12ₜ, R21ₜ, R22ₜ, r11ₜ, r22ₜ, r12ₜ, r21ₜ, k_steps)
        P = cat(P₁, P₂, dims=1)
        α = cat(α₁ , α₂, dims=1)
        
        x̂ = xₜ
        û₁ = u1ₜ
        û₂ = u2ₜ
        û = cat(û₁, û₂, dims=2)

        # Rollout players with new control law
        xₜ, uₜ = Rollout_RK4(dynamics, x₀, x̂, û, umin, umax, H, dt, P, α, 0.5)

    end

    return xₜ, uₜ
end
