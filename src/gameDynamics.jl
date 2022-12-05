using LinearAlgebra
using SparseArrays
using StaticArrays

"""
n Agent dynamics
2D Point Mass:
State x: [x, y, ẋ, ẏ]
Input u: [Fx, Fy]
"""
function pointMass(game::GameSolver, x, u, B)
    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer
    Nx = Nplayer*nx
    Nu = Nplayer*nu
    @assert Nx == length(x) "State input doesn't match size of definition"
    @assert Nu == length(u) "Control input doesn't match size of definition"

    c = 0.1     # Damping coefficient [N-s/m]
    m = 1.0     # Mass [kg]
    if B
        xₖ = zeros(eltype(x), Nx)
    else
        xₖ = zeros(eltype(u), Nx)
    end
    for i in 1:Nplayer
        ẋ = x[nx*i - 1]
        ẍ = -(c/m)*ẋ + u[nu*i - 1]/m
        ẏ = x[nx*i]
        ÿ = -(c/m)*ẏ + u[nu*i]/m
        xₖ[1+(i-1)*nx:i*nx] = [ẋ; ẏ; ẍ; ÿ]
    end
    return xₖ
end;

# """
# n Agent
# Differential Drive:
# State x: [x, y, θ, v]
# Input u: [a, ω]
# """
# function diffDrive4D(dynamics::MultiAgentDynamics , x::Vector{Float64}, u::Vector{Float64})
#     Nx = dynamics.nx   
#     Nu = dynamics.nu
#     Nplayers = dynamics.Nplayers
#     @assert Nx*Nplayers == length(x) "State input doesn't match size of definition"
#     @assert Nu*Nplayers == length(u) "Control input doesn't match size of definition"
#     ẋ = zeros(Nplayers)     # Velocity in x [m/s]
#     ẏ = zeros(Nplayers)     # Acceleration in x [m^2/s]
#     θ̇ = zeros(Nplayers)     # Velocity in y [m/s]
#     v̇ = zeros(Nplayers)     # Acceleration in y [m^2/s]
#     u = zeros(Nplayers*Nu)  # Contol Inputs [N]
#     for i in 1:Nplayers
#         ẋ[i] = cos(x[Nx*i - 1])*x[Nx*i] 
#         ẏ[i] = sin(x[Nx*i - 1])*x[Nx*i] 
#         θ̇[i] = u[Nu*i]
#         v̇[i] = u[Nu*i -1]
#     end

#     return [ẋ; ẏ; θ̇; v̇]
# end;