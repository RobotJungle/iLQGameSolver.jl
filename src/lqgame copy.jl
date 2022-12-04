using LinearAlgebra
using InvertedIndices


function lqGame!(game::GameSolver, Aₜ, Bₜ, Qₜ, lₜ, Rₜ, rₜ, k_steps, Nplayers)
    """
    n = total number of states for each players
    m = total number of inputs for each player
    Nx = total number of states for all players (Nplayers*n)
    Nu = total number of inputs for all players (Nplayers*m)
    k_steps = number of knot points in time horizon
    u is mx1
    Input
    Aₜ is an k_steps by Nx by Nx matrix (k_steps, Nx, Nx)
    Bₜ is an k_steps by Nx by Nu matrix (k_steps, Nx, Nu)
    Qₜ is an k_steps by Nx by Nx matrix (k_steps, Nx, Nx)
    lₜ is an k_steps by Nx by 1 matrix (k_steps, Nx, )
    Rₜ is an k_steps by Nplayers by Nplayers matrix (2x2)
    i.e. Rₜ[1]:
    [
        [R11] [R12] [R13] ... [R1 Nplayers]
        [R21] [R22] [R23] ... [R2 Nplayers]
          .     .     .
          .     .          .
          .     .               .
        [R Nplayers 1] [R Nplayers 2] [R Nplayers 3] ... [R Nplayers Nplayers]
    ] where Rij are matrices
    rₜ is an k_steps by Nplayers by Nplayers (k_steps, Nplayers, Nplayers)
    Similar to Rₜ but rij are vectors 
    Output
    P₁ is an mxn matrix (2x8)
    P₂ is an mxn matrix (2x8)
    α₁ (1x2)
    ζ₁ is an nx1 matrix (8x1)
    """
    nx = game.nx
    nu = game.nu
    Nx = game.Nx
    Nu = game.Nu
    #Nx, Nu = size(Bₜ)

    V = copy(Qₜ[end,:,:])[1] # At last time step

    ζ = copy(lₜ[end,:])[1] # At last time step

    P = zeros(Float32, (k_steps, Nu, Nx))
    
    α = zeros(Float32, (k_steps, Nu))

    #S = deepcopy(Rₜ[1]) #!Very Bad
    S = zeros(Nu, Nu)
    Y = zeros(Nu, Nx)
    Yα = zeros(Nu)

    for t in (k_steps-1):-1:1
        # solving for Ps,αs check equation 19 in document
        for i in 1:Nplayers
            Ni = 1+(i-1)*nu     # Player i's start index
            Nf = i*nu           # Player i's final index
            # left hand side of the matrix
            S[Ni:Nf,Ni:Nf] = Rₜ[t][i,i] + (Bₜ[t,:,Ni:Nf]' * V[i] * Bₜ[t,:,Ni:Nf])     #[Sii, .., ..]
            S[Ni:Nf,Not(Ni:Nf)] = Bₜ[t,:,Ni:Nf]' * V[i] * Bₜ[t,:,Not(Ni:Nf)]         #[.., Sij]
            
            # right hand side of the matrix
            Y = Bₜ[t,:,:]' * V[i] * Aₜ[t,:,:]            # right side for Ps       
            Yα = (Bₜ[t,:,:]' * ζ[i]) + rₜ[t][i,i]        # right side for αs Nu by Nx by 1
        end

        P[t,:,:] = S\Y     # Nu by Nx
        α[t,:] = S\Yα    # Nu by 1
            
        
        # Update value function(s)
        Fₜ = Aₜ[t,:,:] - (Bₜ[t,:,:]*P[t,:,:])    # Nx by Nu 
        
        βₜ = - (Bₜ[t,:,:] * α[t,:])
        
        for i in 1:Nplayers
            Ni = 1+(i-1)*nu     # Player i's start index
            Nf = i*nu           # Player i's final index
            Rij = Diagonal(repeat(Rₜ[t][Ni:Nf,:],3))

            # update zeta
            ζ[i] = lₜ[t][i] + ((P[t,:,:]' * Rij * α[t,:]) - (P[t,:,:]' * rₜ[t][i,i])) + Fₜ'*(ζ[i] + (V[i] * βₜ))

            # update value
            V[i] = Qₜ[t][i] + (P[t,:,:]' * Rij * P[t,:,:]) + (Fₜ' * V[i] * Fₜ)
        end

    end
    return P, α
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
        #uₜ[t,:] .= clamp.([0,0] - P[:,:,t]*(xₜ[t,:] - [20,20,0,0]) - α[:,t], umin, umax)
        uₜ[t,:] .= clamp.(û[t,:] - P[:,:,t]*(xₜ[t,:] - x̂[t,:]) - α_scale*α[:,t], umin, umax)
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
        u1ₜ[t,:] .= clamp.(û[1][t,:] - P[1][:,:,t]*(xₜ[t,:] - x̂[t,:]) - α_scale*α[1][:,t], umin, umax)
        u2ₜ[t,:] .= clamp.(û[2][t,:] - P[2][:,:,t]*(xₜ[t,:] - x̂[t,:]) - α_scale*α[2][:,t], umin, umax)
        ## Hardcode
        xₜ[t+1,:] .= Ad*xₜ[t,:] + B1d*u1ₜ[t,:] + B2d*u2ₜ[t,:]
#         u1ₜ[t,:] .= clamp.([0,0] - P[1][:,:,t]*(x̂[t,:] - xgoal) - α_scale*α[1][:,t], umin, umax)
#         u2ₜ[t,:] .= clamp.([0,0] - P[2][:,:,t]*(x̂[t,:] - xgoal) - α_scale*α[2][:,t], umin, umax)
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