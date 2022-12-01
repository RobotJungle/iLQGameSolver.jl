using LinearAlgebra
using ForwardDiff

function lin_dyn(dynamics, x, u)
    A = ForwardDiff.jacobian(dx -> dynamics(dx, u), x)
    B = ForwardDiff.jacobian(du -> dynamics(x, du), u)
    return A, B
end

function lin_dyn_discrete(dynamics, x, u, dt)
    A = ForwardDiff.jacobian(dx -> dynamics(dx, u), x)
    B = ForwardDiff.jacobian(du -> dynamics(x, du), u)
    A = dt .* A + I
    B = dt .* B
    return A, B
end

function Ad(dynamics, x, u, dt)
    A = ForwardDiff.jacobian(dx -> dynamics(dx, u), x)
    A = dt .* A + I
    return A
end

function Bd(dynamics, x, u, dt)
    B = ForwardDiff.jacobian(du -> dynamics(x, du), u)
    B = dt .* B
    return B
end

