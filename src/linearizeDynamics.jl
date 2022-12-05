using LinearAlgebra
using ForwardDiff

function linearDiscreteDynamics(game, dynamics, x, u)
    A = ForwardDiff.jacobian(dx -> dynamics(game, dx, u, true), x)
    B = ForwardDiff.jacobian(du -> dynamics(game, x, du, false), u)
    A = game.dt .* A + I
    B = game.dt .* B
    return A, B
end

