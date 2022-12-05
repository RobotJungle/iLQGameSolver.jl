using LinearAlgebra
using InvertedIndices


function lqGame!(game::GameSolver, Aₜ, Bₜ, Qₜ, lₜ, Rₜ, rₜ, k_steps)
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
    Nplayer = game.Nplayer
    Nx = nx * Nplayer
    Nu = nu * Nplayer
    #Nx, Nu = size(Bₜ)

    V = copy(Qₜ[end,:,:]) # At last time step

    ζ = copy(lₜ[end,:,:]) # At last time step

    P = zeros(Float32, (k_steps, Nu, Nx))
    
    α = zeros(Float32, (k_steps, Nu))

    #S = deepcopy(Rₜ[1]) #!Very Bad
    S = zeros(Nu, Nu)
    Y = zeros(Nu, Nx)
    Yα = zeros(Nu)

    for t in (k_steps-1):-1:1
        # solving for Ps,αs check equation 19 in document
        for i in 1:Nplayer
            nui = 1+(i-1)*nu     # Player i's start index
            nuf = i*nu           # Player i's final index

            Nxi = 1+(i-1)*Nx     # Player i's state start index
            Nxf = i*Nx           # Player i's state final index

            # left hand side of the matrix
            S[nui:nuf,nui:nuf] = Rₜ[t,nui:nuf,nui:nuf] + (Bₜ[t,:,nui:nuf]' * V[Nxi:Nxf,:] * Bₜ[t,:,nui:nuf])     #[Sii, .., ..]
            S[nui:nuf,Not(nui:nuf)] = Bₜ[t,:,nui:nuf]' * V[Nxi:Nxf,:] * Bₜ[t,:,Not(nui:nuf)]         #[.., Sij]
            
            # @show size(Bₜ[t,:,nui:nuf])
            # @show size(ζ[:,i])
            # @show size(Yα[nui:nuf])
            # @show size(rₜ[t,nui:nuf,i] )
            # right hand side of the matrix
            Y[nui:nuf,:] = Bₜ[t,:,nui:nuf]' * V[Nxi:Nxf,:] * Aₜ[t,:,:]            # right side for Ps       
            Yα[nui:nuf] = (Bₜ[t,:,nui:nuf]' * ζ[:,i]) + rₜ[t,nui:nuf,i]        # right side for αs Nu by Nx by 1
        end
        P[t,:,:] = S\Y     # Nu by Nx
        α[t,:] = S\Yα    # Nu by 1
            
        
        # Update value function(s)
        Fₜ = Aₜ[t,:,:] - (Bₜ[t,:,:]*P[t,:,:])    # Nx by Nu 
        
        βₜ = - (Bₜ[t,:,:] * α[t,:])
        
        for i in 1:Nplayer
            nui = 1+(i-1)*nu     # Player i's start index
            nuf = i*nu           # Player i's final index
            Nxi = 1+(i-1)*Nx     # Player i's state start index
            Nxf = i*Nx           # Player i's state final index

            Rij = Diagonal(repeat(Rₜ[t,nui:nuf,:], Nplayer))

            # update zeta
            ζ[:,i] = lₜ[t,:,i] + ((P[t,:,:]' * Rij * α[t,:]) - (P[t,:,:]' * rₜ[t,:,i])) + Fₜ'*(ζ[:,i] + (V[Nxi:Nxf,:] * βₜ))

            # update value
            V[Nxi:Nxf,:] = Qₜ[t,Nxi:Nxf,:] + (P[t,:,:]' * Rij * P[t,:,:]) + (Fₜ' * V[Nxi:Nxf,:] * Fₜ)
        end

    end
    return P, α
end


function rolloutRK4(game::GameSolver, dynamics, x̂, û, P, α, α_scale)
    """
    Rollout dynamics with initial state x₀ 
    and control law u = -Px - α
    P is an n x b gain matrix
    α is m x 1
    """
    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer
    dt = game.dt
    H = game.H
    x₀ = game.x0
    umin = game.umin
    umax = game.umax

    Nx = Nplayer*nx
    Nu = Nplayer*nu
    k_steps = trunc(Int, H/dt) 

    k_steps = trunc(Int, H/dt) 
    xₜ = zeros(k_steps, Nx)
    uₜ = zeros(k_steps, Nu)
    xₜ[1,:] .= x₀
    for t=1:(k_steps-1)
        uₜ[t,:] .= clamp.(û[t,:] - P[t,:,:]*(xₜ[t,:] - x̂[t,:]) - α_scale*α[t,:], umin, umax)
        k1 = dynamics(game, xₜ[t,:], uₜ[t,:], true)
        k2 = dynamics(game, xₜ[t,:] + 0.5*dt*k1, uₜ[t,:], true)
        k3 = dynamics(game, xₜ[t,:] + 0.5*dt*k2, uₜ[t,:], true)
        k4 = dynamics(game, xₜ[t,:] + dt*k3, uₜ[t,:], true)
        xₜ[t+1,:] .= xₜ[t,:] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    end
    
    return xₜ, uₜ
end

function isConverged(current, last; tol = 1e-4)
    if norm(current - last) > tol
        return false
    else 
        return true
    end
end
