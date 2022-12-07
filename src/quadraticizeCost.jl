using LinearAlgebra
using ForwardDiff

"""
    costPointMass(game, i, Qi, Rii, Rij, Qni, x, ui, uj, B)

A custom function for an Nplayer point mass interaction. The cost
is composed of state tracking error, control input tracking error, 
terminal state error, and collision avoidance. 

Collision avoidance: If a player is less than a distance game.dmax 
from another player, then they both incur a penalty. That penalty is 
quadratic as it is proportional to how close together they are. 
If a player is not violating the constraint, then the penalty is zero.

Inputs:
    game: GameSolver struct (see solveilqGame.jl)
    i: Player number 
    Qi: Player i's state tracking cost matrix (Nx, Nx)
    Rii: Player i's control input cost matrix wrt to it's control input 
        (nu, nu)
    Rij: Player i's control input cost matrix wrt to all other players' 
        control inputs ((Nplayer-1)*nu, (Nplayer-1*nu))
    Qni: Player i's terminal state cost matrix (Nx, Nx)
    x: State vector (Nx)
    ui: Player i's control inputs vector (nu)
    uj: All other players' control inputs vector ((Nplayer-1)*nu)
    B: Terminal state? (Bool)

Outputs:
    Cost: Player i's scalar cost
"""
function costPointMass(game, i, Qi, Rii, Rij, Qni, x, ui, uj, B)

    # No need to pass Qi, Rii, Rij, Qni into so many functions.
    # They are only used here so should just use the game struct
    # instead of passing as args. 
    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer

    dx = x - game.xf
    rel_dist = 0
    xi = 1+(i-1)*nx             # Player i's x state
    yi = 2+(i-1)*nx             # Player i's y state
    for j = 1:Nplayer
        xj = 1+(j-1)*nx         # Player j's x state
        yj = 2+(j-1)*nx         # Player j's y state

        rel_dist += (x[xi:yi] - x[xj:yj])'*I*(x[xi:yi] - x[xj:yj])
    end
    
    if B 
        return 0.5*dx'*Qni*dx
    else            
        Rij = Diagonal(repeat(Rij, Nplayer-1))
        nui = 1+(i-1)*nu     # Player i's control start index
        nuf = i*nu           # Player i's control final index
        dui = ui - game.uf[nui:nuf]
        duj = uj - game.uf[Not(nui:nuf)]

        ## !!!Stack Rii and Rij for single matrix multiplication!!!
        # du'*R*du
        return 0.5*(dx'*Qi*dx + dui'*Rii*dui + duj'*Rij*duj) + game.ρ*(min(sqrt(rel_dist) - game.dmax, 0))^2     
    end
end


"""
    quadraticizeCost(game, cost_fun, i, Qi, Rii, Rij, Qni, x, ui, uj, B)

A custom function for an Nplayer point mass interaction. The cost
is composed of state tracking error, control input tracking error, 
terminal state error, and collision avoidance. 

Collision avoidance: If a player is less than a distance game.dmax 
from another player, then they both incur a penalty. That penalty is 
quadratic as it is proportional to how close together they are. 
If a player is not violating the constraint, then the penalty is zero.

Inputs:
    game: GameSolver struct (see solveilqGame.jl)
    cost_fun: Cost function for the game 
    i: Player number 
    Qi: Player i's state tracking cost matrix (Nx, Nx)
    Rii: Player i's control input cost matrix wrt to it's control input 
        (nu, nu)
    Rij: Player i's control input cost matrix wrt to all other players' 
        control inputs ((Nplayer-1)*nu, (Nplayer-1*nu))
    Qni: Player i's terminal state cost matrix (Nx, Nx)
    x: State vector (Nx)
    ui: Player i's control inputs vector (nu)
    uj: All other players' control inputs vector ((Nplayer-1)*nu)
    B: Terminal state? (Bool)

Outputs:
    cost: Player i's scalar cost
    Q̂i: Player i's state tracking cost matrix (Nx, Nx)
    l̂i: Player i's state cost vector (Nx, Nplayer) 
    R̂ii: Player i's control input cost matrix wrt to it's control input 
        (nu, nu)
    r̂ii: Player i's control input cost vector wrt to it's control input 
        (nu)
    R̂ij:Player i's control input cost matrix wrt to all other players' 
        control inputs ((Nplayer-1)*nu, (Nplayer-1*nu))
    r̂ij: Player i's control input cost vector wrt to all other players' 
    control inputs ((Nplayer-1)*nu)
"""
function quadraticizeCost(game, cost_fun, i, Qi, Rii, Rij, Qni, x, ui, uj, B)
    """
    2nd order Taylor expansion of cost at t
    I neglected the mixed partials in the hessian
    """
    Nplayer = game.Nplayer
    nu = game.nu
    Q̂i = ForwardDiff.hessian(dx -> cost_fun(game, i, Qi, Rii, Rij, Qni, dx, ui, uj, B), x)
    l̂i = ForwardDiff.gradient(dx -> cost_fun(game, i, Qi, Rii, Rij, Qni, dx, ui, uj, B), x)
    R̂ii = ForwardDiff.hessian(du -> cost_fun(game, i, Qi, Rii, Rij, Qni, x, du, uj, B), ui)
    r̂ii = ForwardDiff.gradient(du -> cost_fun(game, i, Qi, Rii, Rij, Qni, x, du, uj, B), ui)
    R̂ijH = ForwardDiff.hessian(du -> cost_fun(game, i, Qi, Rii, Rij, Qni, x, ui, du, B), uj)
    r̂ij = ForwardDiff.gradient(du -> cost_fun(game, i, Qi, Rii, Rij, Qni, x, ui, du, B), uj)

    dx = x - game.xf
    if B 
        dui = zeros(size(ui))
        duj = zeros(size(uj))
    else
        # !!!! Doing the same thing twice; here and in the cost function!!!!!
        nu = game.nu
        nui = 1+(i-1)*nu     # Player i's control start index
        nuf = i*nu           # Player i's control final index
        dui = ui - game.uf[nui:nuf]
        duj = uj - game.uf[Not(nui:nuf)]
    end
    cost = 0.5 * dx' * (Q̂i*dx + 2*l̂i) + 0.5 * dui' * (R̂ii*dui + 2*r̂ii) + 0.5 * duj' * (R̂ijH*duj + 2*r̂ij)

    R̂ij = zeros(nu, (Nplayer-1)*nu)
    for i =1:Nplayer-1
        R̂ij .+= R̂ijH[1+(i-1)*nu:i*nu,:] 
    end

    return cost, Q̂i, l̂i, R̂ii, r̂ii, R̂ij, r̂ij
end