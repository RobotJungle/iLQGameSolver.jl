using LinearAlgebra
using InvertedIndices

"""
    lgame!(game, Aₜ, Bₜ, Qₜ, lₜ, Rₜ, rₜ)

Solve the LQ game as described by Basar and Olsder in chapter 6 of
'Dynamic Noncooperative Game Theory'. 

Inputs:
    game: GameSolver struct (see solveilqGame.jl)
    Aₜ: Linearized discrete state matrix (k_steps, Nx, Nx)
    Bₜ: Linearized discrete control input matrix (k_steps, Nx, Nu)
    Qₜ: State cost matrices, stacked vertically (k_steps, Nx*Nplayer, Nx)
    i.e. Qₜ[1,:,:] (at t = 1):
    [
        [Q1]
        [Q2] 
          .   
          .     
          .     
        [QNplayer] 
    ] 
    lₜ: State cost vectors, stacked vertically (k_steps, Nx, Nplayer)
    Rₜ: Control input cost matrices (k_steps, Nu, Nu)
    i.e. Rₜ[1,:,:] (at t = 1):
    [
        [R11] [R12] [R13] ... [R1 Nplayer]
        [R21] [R22] [R23] ... [R2 Nplayer]
            .     .     .
            .     .          .
            .     .               .
        [R Nplayers 1] [R Nplayers 2] [R Nplayers 3] ... [R Nplayers Nplayers]
    ] where Rij are matrices
    rₜ: Control input cost vectors (k_steps, Nu, Nplayers)

Outputs:
    u⋆ = -Px - α
    P: Optimal control gain matrix (k_steps, Nu, Nx)
    α: Optimal control offset vector (k_steps, Nu)
"""
function lqGame!(game, solver)
    nx = game.nx
    nu = game.nu

    Nplayer = game.Nplayer
    Nx = nx*Nplayer
    Nu = nu*Nplayer

    NHor = game.NHor

    Aₜ = solver.Aₜ
    Bₜ = solver.Bₜ
    Qₜ = solver.Qₜ
    lₜ = solver.lₜ
    Rₜ = solver.Rₜ
    rₜ = solver.rₜ

    # Should use for warm-starting successive solutions
    # Careful! lqgame_test.jl will fail!
    # P = solver.P
    # α = solver.α

    P = zeros(Float32, (NHor, Nu, Nx))
    α = zeros(Float32, (NHor, Nu))

    V = copy(Qₜ[end,:,:]) # At last time step
    ζ = copy(lₜ[end,:,:]) # At last time step

    S = solver.S
    Y = solver.Y
    Yα = solver.Yα

    for t = (NHor-1):-1:1
        # solving for Ps,αs check equation 19 in document
        for i = 1:Nplayer
            Nxi, Nxf, nui, nuf = getPlayerIdx(game, i) # get player i's indices
            
            # left hand side of the matrix
            S[nui:nuf,nui:nuf] = Rₜ[t,nui:nuf,nui:nuf] + (Bₜ[t,:,nui:nuf]' * V[Nxi:Nxf,:] * Bₜ[t,:,nui:nuf])     #[Sii, .., ..]
            S[nui:nuf,Not(nui:nuf)] = Bₜ[t,:,nui:nuf]' * V[Nxi:Nxf,:] * Bₜ[t,:,Not(nui:nuf)]         #[.., Sij]
            
            Y[nui:nuf,:] = Bₜ[t,:,nui:nuf]' * V[Nxi:Nxf,:] * Aₜ[t,:,:]            # right side for Ps       
            Yα[nui:nuf] = (Bₜ[t,:,nui:nuf]' * ζ[:,i]) + rₜ[t,nui:nuf,i]        # right side for αs Nu by Nx by 1
        end
        P[t,:,:] = S\Y     # Nu by Nx
        α[t,:] = S\Yα    # Nu by 1
            
        # Update value function(s)
        Fₜ = Aₜ[t,:,:] - (Bₜ[t,:,:]*P[t,:,:])    # Nx by Nu 
        βₜ = - (Bₜ[t,:,:] * α[t,:])
        
        for i = 1:Nplayer

            Nxi, Nxf, nui, nuf = getPlayerIdx(game, i)  # get player i's indices

            Rij = Diagonal(repeat(Rₜ[t,nui:nuf,:], Nplayer))

            # update zeta
            ζ[:,i] = lₜ[t,:,i] + ((P[t,:,:]' * Rij * α[t,:]) - (P[t,:,:]' * rₜ[t,:,i])) + Fₜ'*(ζ[:,i] + (V[Nxi:Nxf,:] * βₜ))

            # update value
            V[Nxi:Nxf,:] = Qₜ[t,Nxi:Nxf,:] + (P[t,:,:]' * Rij * P[t,:,:]) + (Fₜ' * V[Nxi:Nxf,:] * Fₜ)
        end
        
    end
        solver.P = P
        solver.α = α
    return nothing
end

# Rollout dynamics with initial state x₀ and control law u = -Px - α
# P is an n x b gain matrix and α is m x 1
function rolloutRK4(game, solver, dynamics, x0, α_scale)
    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer
    dt = game.dt
    umin = game.umin
    umax = game.umax

    Nx = Nplayer*nx
    Nu = Nplayer*nu

    NHor = game.NHor

    P = solver.P
    α = solver.α
    x̂ = solver.x̂ 
    û = solver.û

    xₜ = zeros(NHor, Nx)
    uₜ = zeros(NHor, Nu)
    xₜ[1,:] .= x0
    for t=1:(NHor-1)
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
