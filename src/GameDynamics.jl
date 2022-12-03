using LinearAlgebra
using SparseArrays
using StaticArrays

"""
    MultiAgentDynamics
"""
struct MultiAgentDynamics
    nx::Int64           # Number of states for each player
    nu::Int64           # Number of inputs for each player
    Nplayers::Int64     # Total number of players
end

"""
n Agent dynamics
2D Point Mass:
State x: [x, y, ẋ, ẏ]
Input u: [Fx, Fy]
"""

function point_mass(dynamics::MultiAgentDynamics , x, u, B)
    Nx = dynamics.nx
    Nu = dynamics.nu
    Nplayers = dynamics.Nplayers
    @assert Nx*Nplayers == length(x) "State input doesn't match size of definition"
    @assert Nu*Nplayers == length(u) "Control input doesn't match size of definition"
    c = 0.2     # Damping coefficient [N-s/m]
    m = 1.0     # Mass [kg]
    if B
        xₖ = zeros(eltype(x), Nplayers*Nx)
    else
        xₖ = zeros(eltype(u), Nplayers*Nx)
    end
    for i in 1:Nplayers
        ẋ = x[Nx*i - 1]
        ẍ = -(c/m)*ẋ + u[Nu*i - 1]/m
        ẏ = x[Nx*i]
        ÿ = -(c/m)*ẏ + u[Nu*i]/m
        xₖ[1+(i-1)*Nx:i*Nx] = [ẋ; ẏ; ẍ; ÿ]
    end
    return xₖ
end;

"""
n Agent
Differential Drive:
State x: [x, y, θ, v]
Input u: [a, ω]
"""
function diff_drive_4D(dynamics::MultiAgentDynamics , x::Vector{Float64}, u::Vector{Float64})
    Nx = dynamics.nx   
    Nu = dynamics.nu
    Nplayers = dynamics.Nplayers
    @assert Nx*Nplayers == length(x) "State input doesn't match size of definition"
    @assert Nu*Nplayers == length(u) "Control input doesn't match size of definition"
    ẋ = zeros(Nplayers)     # Velocity in x [m/s]
    ẏ = zeros(Nplayers)     # Acceleration in x [m^2/s]
    θ̇ = zeros(Nplayers)     # Velocity in y [m/s]
    v̇ = zeros(Nplayers)     # Acceleration in y [m^2/s]
    u = zeros(Nplayers*Nu)  # Contol Inputs [N]
    for i in 1:Nplayers
        ẋ[i] = cos(x[Nx*i - 1])*x[Nx*i] 
        ẏ[i] = sin(x[Nx*i - 1])*x[Nx*i] 
        θ̇[i] = u[Nu*i]
        v̇[i] = u[Nu*i -1]
    end

    return [ẋ; ẏ; θ̇; v̇]
end;