using LinearAlgebra


function lqGame!(Aₜ, B1ₜ, B2ₜ, Q1ₜ, Q2ₜ, l1ₜ, l2ₜ, R11ₜ, R12ₜ, R21ₜ, R22ₜ, r11ₜ, r22ₜ, r12ₜ, r21ₜ, k_steps)
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


function Rollout_RK4(fun, x₀, x̂, û, umin, umax, H, dt, P, α, α_scale)
    """
    Rollout dynamics with initial state x₀ 
    and control law u = -Px - α
    P is an n x b gain matrix
    α is m x 1
    """
    
    m = size(û)[2] #2 controls
    k_steps = trunc(Int, H/dt) 
    xₜ = zeros(k_steps, length(x₀)) # 1500 x n
    uₜ = zeros(k_steps, m)
    xₜ[1,:] .= x₀
    for t=1:(k_steps-1)
        # WHAT IS x̂ in xₜ[t,:] - x̂
        #uₜ[t,:] .= clamp.([0,0] - P[t,:,:]*(xₜ[t,:] - [20,20,0,0]) - α[t,:], umin, umax)
        uₜ[t,:] .= clamp.(û[t,:] - P[t,:,:]*(xₜ[t,:] - x̂[t,:]) - α_scale*α[t,:], umin, umax)
        k1 = fun(xₜ[t,:], uₜ[t,:])
        k2 = fun(xₜ[t,:] + 0.5*dt*k1, uₜ[t,:])
        k3 = fun(xₜ[t,:] + 0.5*dt*k2, uₜ[t,:])
        k4 = fun(xₜ[t,:] + dt*k3, uₜ[t,:])
        xₜ[t+1,:] .= xₜ[t,:] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    end
    
    return xₜ, uₜ
end

function rollout_PM(x₀, x̂, û, umin, umax, H, dt, P, α, α_scale)

    m₁ = 1
    m₂ = 1
    c = 0.1
    A1 = sparse([0 0 1 0; 0 0 0 1; 0 0 (-c/m₁) 0; 0 0 0 (-c/m₁)])
    A2 = sparse([0 0 1 0; 0 0 0 1; 0 0 (-c/m₂) 0; 0 0 0 (-c/m₂)])
    A = blockdiag(A1, A2)
    B1 = sparse([0 0; 0 0; (1/m₁) 0; 0 (1/m₁); 0 0; 0 0; 0 0; 0 0])  #Control Jacobian for point mass 1
    B2 = sparse([0 0; 0 0; 0 0; 0 0; 0 0; 0 0; (1/m₂) 0; 0 (1/m₂)])    #Control Jacobian for point mass 2

    Ad = dt .* A + I    #discretize (zero order hold)
    B1d = dt .*B1   #discrete (zero order hold)
    B2d = dt .*B2;   #discrete (zero order hold)

    m = 2 #2 controls
    k_steps = trunc(Int, H/dt) 
    xₜ = zeros(k_steps, length(x₀)) # 1500 x n
    u1ₜ = zeros(k_steps, m) 
    u2ₜ = zeros(k_steps, m) 
    xₜ[1,:] .= x₀
    for t=1:(k_steps-1)
        u1ₜ[t,:] .= clamp.(û[1][t,:] - P[1][t,:,:]*(xₜ[t,:] - x̂[t,:]) - α_scale*α[1][t,:], umin, umax)
        u2ₜ[t,:] .= clamp.(û[2][t,:] - P[2][t,:,:]*(xₜ[t,:] - x̂[t,:]) - α_scale*α[2][t,:], umin, umax)
        ## Hardcode
        xₜ[t+1,:] .= Ad*xₜ[t,:] + B1d*u1ₜ[t,:] + B2d*u2ₜ[t,:]
#         u1ₜ[t,:] .= clamp.([0,0] - P[1][t,:,:]*(x̂[t,:] - xgoal) - α_scale*α[1][t,:], umin, umax)
#         u2ₜ[t,:] .= clamp.([0,0] - P[2][t,:,:]*(x̂[t,:] - xgoal) - α_scale*α[2][t,:], umin, umax)
#         xₜ[t+1,:] .= xgoal + Ad*(x̂[t,:] - xgoal) + B1d*(u1ₜ[t,:] - [0,0]) + B2d*(u1ₜ[t,:] - [0,0])
    end
    return xₜ, u1ₜ, u2ₜ
end


function isConverged(current, last; tol = 1e-4)
    if norm(current - last) > tol
        return false
    else 
        return true
    end
end