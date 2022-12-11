using LinearAlgebra
using SparseArrays
using StaticArrays

"""
n Agent dynamics
2D Point Mass:
State x: [x, y, ẋ, ẏ]
Input u: [Fx, Fy]
"""
function pointMass(game, x, u, B)
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
    return xₖ #+ rand(Nx)*0.05
end


"""
2 Agent
Quadcopters:
State x: [x, y, z, ̇x, ̇y, ̇z, ϕ, θ, Ψ]
Input u: [̇zᵢ, ϕᵢ, θᵢ, ̇Ψᵢ]
"""
function quadcopter(game, x, u, B)

    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer
    Nx = Nplayer*nx
    Nu = Nplayer*nu

    τz = 1.0
    τθ = 1.0
    τϕ = 1.0
    g = 9.81

    # Extract player 1 inputs
    żcmd₁ = u[1]
    ϕcmd₁ = u[2]
    θcmd₁ = u[3]
    Ψ̇cmd₁ = u[4]

    # Extract player 1 states (1-3)
    ẋ₁ = x[4]
    ẏ₁ = x[5]
    ż₁ = x[6]

    ϕ₁ = x[7]
    θ₁ = x[8]
    Ψ₁ = x[9]

    # Solve for player 1 states (4-9)
    z̈₁ = (1/τz)*(żcmd₁ - ż₁)
    ẍ₁ = sin(ϕ₁)*sin(Ψ₁)+cos(ϕ₁)*sin(θ₁)*cos(Ψ₁)*(z̈₁ + g)/(cos(θ₁)*cos(ϕ₁))
    ÿ₁ = -sin(ϕ₁)*sin(Ψ₁)+cos(ϕ₁)*sin(θ₁)*sin(Ψ₁)*(z̈₁ + g)/(cos(θ₁)*cos(ϕ₁))
    ϕ̇₁ = (1/τϕ)*(ϕcmd₁ - ϕ₁)
    θ̇₁ = (1/τθ)*(θcmd₁ - θ₁)
    Ψ̇₁ = Ψ̇cmd₁

    # Extract player 2 inputs
    żcmd₂ = u[5] 
    ϕcmd₂ = u[6]
    θcmd₂ = u[7]
    Ψ̇cmd₂ = u[8]

    # Extract player 2 states (10-12)
    ẋ₂ = x[13]
    ẏ₂ = x[14]
    ż₂ = x[15]

    ϕ₂ = x[16]
    θ₂ = x[17]
    Ψ₂ = x[18]

    # Solve for player 2 states (13-18)
    z̈₂ = (1/τz)*(żcmd₂ - ż₂)
    ẍ₂ = sin(ϕ₂)*sin(Ψ₂)+cos(ϕ₂)*sin(θ₂)*cos(Ψ₂)*(z̈₂ + g)/(cos(θ₂)*cos(ϕ₂))
    ÿ₂ = -sin(ϕ₂)*sin(Ψ₂)+cos(ϕ₂)*sin(θ₂)*sin(Ψ₂)*(z̈₂ + g)/(cos(θ₂)*cos(ϕ₂))
    ϕ̇₂ = (1/τϕ)*(ϕcmd₂ - ϕ₂)
    θ̇₂ = (1/τθ)*(θcmd₂ - θ₂)
    Ψ̇₂ = Ψ̇cmd₂

    return [ẋ₁; ẏ₁; ż₁; ẍ₁; ÿ₁; z̈₁; ϕ̇₁; θ̇₁; Ψ̇₁; ẋ₂; ẏ₂; ż₂; ẍ₂; ÿ₂; z̈₂; ϕ̇₂; θ̇₂; Ψ̇₂]

end


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