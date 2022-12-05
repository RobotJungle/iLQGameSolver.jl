# Functions for 2 Player games
using LinearAlgebra
using ForwardDiff

function lqGame2P!(Aₜ, B1ₜ, B2ₜ, Q1ₜ, Q2ₜ, l1ₜ, l2ₜ, R11ₜ, R12ₜ, R21ₜ, R22ₜ, r11ₜ, r22ₜ, r12ₜ, r21ₜ, k_steps)
    """
    n = total number of states for both players
    m = number of inputs for each player
    Input
    Ad is an nxn matrix (8x8)
    B1d is an nxm matrix (8x2)
    B2d is an nxm matrix (8x2)
    u is mx1 (2x1)
    Q1 is an nxn matrix (8x8)
    Q2 is an nxn matrix (8x8)
    l1 is a 8x1
    R11 is an mxm matrix (2x2)
    R12 is an mxm matrix (2x2)
    R21 is an mxm matrix (2x2)
    R22 is an mxm matrix (2x2)
    Output
    P₁ is an mxn matrix (2x8)
    P₂ is an mxn matrix (2x8)
    α₁ (1x2)
    ζ₁ is an nx1 matrix (8x1)
    """
    ~, n, m = size(B1ₜ)

    V₁ = copy(Q1ₜ[end,:,:]) # At last time step
    V₂ = copy(Q2ₜ[end,:,:]) # At last time step

    ζ₁ = copy(l1ₜ[end,:]) # At last time step
    ζ₂ = copy(l2ₜ[end,:]) # At last time step

    P₁ = zeros(Float32, (k_steps, m, n))
    P₂ = zeros(Float32, (k_steps, m, n))
    
    α₁ = zeros(Float32, (k_steps, m))
    α₂ = zeros(Float32, (k_steps, m))

    for t in (k_steps-1):-1:1
        # solving for Ps, check equation 19 in document
        S11 = R11ₜ[t,:,:] + (B1ₜ[t,:,:]' * V₁ * B1ₜ[t,:,:]) #2x2
        S12 = B1ₜ[t,:,:]' * V₁ * B2ₜ[t,:,:] # 2x2
        S22 = R22ₜ[t,:,:] + (B2ₜ[t,:,:]' * V₂ * B2ₜ[t,:,:])
        S21 = B2ₜ[t,:,:]' * V₂ * B1ₜ[t,:,:]
        S = [S11 S12; S21 S22] # 4x4
        Y1 = B1ₜ[t,:,:]' * V₁ * Aₜ[t,:,:] # 2x8
        Y2 = B2ₜ[t,:,:]' * V₂ * Aₜ[t,:,:] 
        Y = [Y1; Y2] # 4 x 8
        P = S\Y # 4x8
        P₁[t,:,:] = P[1:m, :] #2x8
        P₂[t,:,:] = P[m+1:2*m, :]
        
        # solve for αs (right hand side of the eqn)
        Yα1 = (B1ₜ[t,:,:]' * ζ₁) + r11ₜ[t,:] # 2x2
        Yα2 = (B2ₜ[t,:,:]' * ζ₂) + r22ₜ[t,:] # 2x2
        Yα = [Yα1; Yα2]  # 4x2
        α = S\Yα # 4x2
        α₁[t,:] = α[1:m, :]
        α₂[t,:] = α[m+1:2*m, :]
        
        #Update value function(s)
        Fₜ = Aₜ[t,:,:] - (B1ₜ[t,:,:]*P₁[t,:,:] + B2ₜ[t,:,:]*P₂[t,:,:])

        βₜ = - (B1ₜ[t,:,:] * α₁[t,:] + B2ₜ[t,:,:] * α₂[t,:])
        
        ζ₁ = l1ₜ[t,:] + ((P₁[t,:,:]' * R11ₜ[t,:,:] * α₁[t,:]) - (P₁[t,:,:]' * r11ₜ[t,:])) + 
            ((P₂[t,:,:]' * R12ₜ[t,:,:] * α₂[t,:]) - (P₂[t,:,:]' * r12ₜ[t,:])) + Fₜ'*(ζ₁ + (V₁ * βₜ))
        
        ζ₂ = l2ₜ[t,:] + ((P₁[t,:,:]' * R21ₜ[t,:,:] * α₁[t,:]) - (P₁[t,:,:]' * r21ₜ[t,:])) + 
            ((P₂[t,:,:]' * R22ₜ[t,:,:] * α₂[t,:]) - (P₂[t,:,:]' * r22ₜ[t,:])) + Fₜ'*(ζ₂ + (V₂ * βₜ))
        
        V₁ = Q1ₜ[t,:,:] + (P₁[t,:,:]' * R11ₜ[t,:,:] * P₁[t,:,:]) + 
            (P₂[t,:,:]' * R12ₜ[t,:,:] * P₂[t,:,:]) + (Fₜ' * V₁ * Fₜ)
        
        V₂ = Q2ₜ[t,:,:] + (P₁[t,:,:]' * R21ₜ[t,:,:] * P₁[t,:,:]) + 
            (P₂[t,:,:]' * R22ₜ[t,:,:] * P₂[t,:,:]) + (Fₜ' * V₂ * Fₜ)
    end
    return P₁, P₂, α₁, α₂
end


function costPointMass2P(Qi, Rii, Rij, Qni, x, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B)

    goal = x - xgoal
    rel_dist = (x[1:2] - x[5:6])'*I*(x[1:2] - x[5:6])
    if B 
        return 0.5*goal'*Qni*goal
    else            
        dx = x - xgoal
        dui = ui - uigoal
        duj = uj - ujgoal
        return 0.5*(dx'*Qi*dx + dui'*Rii*dui + duj'*Rij*duj) + ρ*(min(sqrt(rel_dist) - dmax, 0))^2 
        
    end
end

function quadraticizeCost2P(cost_fun, Qi, Rii, Rij, Qni, x, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B)
    """
    2nd order Taylor expansion of cost at t
    I neglected the mixed partials in the hessian
    """
    Q̂i = ForwardDiff.hessian(dx -> cost_fun(Qi, Rii, Rij, Qni, dx, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B), x)
    l̂i = ForwardDiff.gradient(dx -> cost_fun(Qi, Rii, Rij, Qni, dx, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B), x)
    R̂ii = ForwardDiff.hessian(du -> cost_fun(Qi, Rii, Rij, Qni, x, du, uj, xgoal, uigoal, ujgoal, dmax, ρ, B), ui)
    r̂ii = ForwardDiff.gradient(du -> cost_fun(Qi, Rii, Rij, Qni, x, du, uj, xgoal, uigoal, ujgoal, dmax, ρ, B), ui)
    R̂ij = ForwardDiff.hessian(du -> cost_fun(Qi, Rii, Rij, Qni, x, ui, du, xgoal, uigoal, ujgoal, dmax, ρ, B), uj)
    r̂ij = ForwardDiff.gradient(du -> cost_fun(Qi, Rii, Rij, Qni, x, ui, du, xgoal, uigoal, ujgoal, dmax, ρ, B), uj)

    if B
        dx = x - xgoal
        dui = zeros(size(ui))
        duj = zeros(size(uj))
    else
        dx = x - xgoal
        dui = ui - uigoal
        duj = uj - ujgoal
    end
    cost = 0.5 * dx' * (Q̂i*dx + 2*l̂i) + 0.5 * dui' * (R̂ii*dui + 2*r̂ii) + 0.5 * duj' * (R̂ij*duj + 2*r̂ij)
    return cost, Q̂i, l̂i, R̂ii, r̂ii, R̂ij, r̂ij
end


