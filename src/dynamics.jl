using LinearAlgebra

#Todo: make this more generalizable to multiple agents

"""
2 Agent
2D Point Mass:
State x: [x, y, ẋ, ẏ]
Input u: [FX, Fy]
"""

function point_mass(x, u)
    c = 0.1     # Damping coefficient [N-s/m]
    m = 1.0     # Mass [kg]
    ẋ₁ = x[3]
    ẍ₁ = -(c/m)*ẋ₁ + u[1]/(m)   
    ẏ₁ = x[4]
    ÿ₁ = -(c/m)*ẏ₁ + u[2]/(m)  
    ẋ₂ = x[7]
    ẍ₂ = -(c/m)*ẋ₂ + u[3]/(m)   
    ẏ₂ = x[8]
    ÿ₂ = -(c/m)*ẏ₂ + u[4]/(m)  
    return [ẋ₁; ẏ₁; ẍ₁; ÿ₁; ẋ₂; ẏ₂; ẍ₂; ÿ₂]
end;

"""
2 Agent
Differential Drive:
State x: [x, y, θ]
Input u: [v, ω]
"""
function diff_drive_3D(x, u)
    #x[3] = atan(sin(x[3]),cos(x[3]))
    ẋ₁ = cos(x[3])*u[1]
    ẏ₁ = sin(x[3])*u[1]
    θ̇₁ = u[2]
    #x[6] = atan(sin(x[6]),cos(x[6]))
    ẋ₂ = cos(x[6])*u[3]
    ẏ₂ = sin(x[6])*u[3]
    θ̇₂ = u[4]
    # l = 0.16
    # ẋ₁ = 0.5*(u[1] + u[2])*cos(x[3])
    # ẏ₁ = 0.5*(u[1] + u[2])*sin(x[3])
    # θ̇₁ = (u[1] - u[2])/l
    # v̇ᵣ₁ = u[1]
    # v̇ₗ₁ = u[2]

    # ẋ₂ = 0.5*(u[3] + u[4])*cos(x[8])
    # ẏ₂ = 0.5*(u[3] + u[4])*sin(x[8])
    # θ̇₂ = (u[3] - u[4])/l
    # v̇ᵣ₂ = u[3]
    # v̇ₗ₂ = u[4]

    return [ẋ₁; ẏ₁; θ̇₁; ẋ₂; ẏ₂; θ̇₂]#[ẋ₁; ẏ₁; θ̇₁; v̇ᵣ₁; v̇ₗ₁; ẋ₂; ẏ₂; θ̇₂; v̇ᵣ₂; v̇ₗ₂]
end

function diff_drive_4D(x, u)
    #x[3] = atan(sin(x[3]),cos(x[3]))
    ẋ₁ = cos(x[3])*x[4]
    ẏ₁ = sin(x[3])*x[4]
    θ̇₁ = u[2]
    v̇₁ = u[1]

    #x[6] = atan(sin(x[6]),cos(x[6]))
    ẋ₂ = cos(x[7])*x[8]
    ẏ₂ = sin(x[7])*x[8]
    θ̇₂ = u[4]
    v̇₂ = u[3]

    return [ẋ₁; ẏ₁; θ̇₁; v̇₁; ẋ₂; ẏ₂; θ̇₂; v̇₂]
end


"""
2 Agent
Quadcopters:
State x: [x, y, z, ̇x, ̇y, ̇z, ϕ, θ, Ψ]
Input u: [̇zᵢ, ϕᵢ, θᵢ, ̇Ψᵢ]
"""
function quadcopter(x, u)
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