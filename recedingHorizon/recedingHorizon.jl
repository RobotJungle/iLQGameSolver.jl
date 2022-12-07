using LinearAlgebra
using SparseArrays

function recedingHorizon(game, solver, dynamics, costf)

    Nplayer = game.Nplayer
    Nx = game.nx * Nplayer
    Nu = game.nu * Nplayer
    dt = game.dt
    tf = game.tf
    NHor = game.NHor

    T = range(0, tf, step=dt)
    N = length(T)

    # From 1:Final Time
    X = [@SVector zeros(Nx) for _ = 1:N]
    U = [@SVector zeros(Nu) for _ = 1:N-1]

    X[1] = game.x0

    tstart = time_ns()
    # From 1 to the final time step
    for k = 1:N-1-NHor
        X[k+1], U[k] = iLQGameSolver.solveILQGame(game, solver, dynamics, costf, k, X[k], false)
        println(k)
    end
    for k = N-NHor:N-1
        X[k+1], U[k] = iLQGameSolver.solveILQGame(game, solver, dynamics, costf, k, X[k], true)
        println(k)
    end
    tend = time_ns()

    rate = N / (tend - tstart) * 1e9
    println("Controller ran at $rate Hz")
    return X,U,T

end