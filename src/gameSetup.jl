using LinearAlgebra
using SparseArrays

"""
    GameSetup

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
struct GameStruct
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
    tf::Float64
    NHor::Int64
    dmax::Float64
    ρ::Float64
    tol::Float64
end

"""
    GameSetup(n,m,Nplayer,Q,R,Qn,dt,H,dmax,ρ)

Generates a `LQGameSolver` that uses solveiLQGames to solve the problem.
"""
function GameSetup(nx::Int64, nu::Int64, Nplayer::Int64, Q::SparseMatrixCSC{Float32,Int}, 
                    R::SparseMatrixCSC{Float32,Int}, Qn::SparseMatrixCSC{Float32,Int}, 
                    dt::Float64, tf::Float64, NHor::Int64, dmax::Float64, ρ::Float64, tol::Float64)
    Nx = Nplayer*nx    
    Nu = Nplayer*nu
    x0 = zeros(Nx)
    xf = zeros(Nx)
    umin = zeros(Nu)
    umax = zeros(Nu)
    uf = zeros(Nu)
    GameStruct(nx, nu, Nplayer, x0, xf, umin, umax, uf, Q, R, Qn, dt, tf, NHor, dmax, ρ, tol)
end