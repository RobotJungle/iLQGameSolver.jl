using LinearAlgebra
using ForwardDiff

"""
    linearDiscreteDynamics(game, dynamics, x, u)

Linearizes the continuous dynamics of the whole game wrt the current state, x,
and the control inputs, u. Then discretizes the linearized continuous dynamics.

Inputs:
    game: GameSolver Struct (see solveilqGame.jl)
    dynamics: Dynamics function
    x: state vector
    u: control input vector

Outputs:
    A: Linearized Discrete state matrix
    B: Linearized Discrete control input matrix
"""

function linearDiscreteDynamics(game, dynamics, x, u)
    A = ForwardDiff.jacobian(dx -> dynamics(game, dx, u, true), x)
    B = ForwardDiff.jacobian(du -> dynamics(game, x, du, false), u)
    A = game.dt .* A + I
    B = game.dt .* B
    return A, B
end

