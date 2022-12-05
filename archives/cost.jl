using LinearAlgebra
using ForwardDiff

function costPointMass(Qi, Rii, Rij, Qni, x, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B)

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

function costDiffDrive(Qi, Rii, Rij, Qni, x, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B)

    goal = x - xgoal
    if size(Qi)[1] == 6
        rel_dist = (x[1:2] - x[4:5])'*I*(x[1:2] - x[4:5])
    elseif size(Qi)[1] == 8
        rel_dist = (x[1:2] - x[5:6])'*I*(x[1:2] - x[5:6])
    else
        rel_dist = (x[1:2] - x[6:7])'*I*(x[1:2] - x[6:7])
    end

    if B 
        return 0.5*goal'*Qni*goal
    else            
        dx = x - xgoal
        dui = ui - uigoal
        duj = uj - ujgoal
        return 0.5*(dx'*Qi*dx + dui'*Rii*dui + duj'*Rij*duj) + ρ*(min(sqrt(rel_dist) - dmax, 0))^2 
        
    end
end

function quadratic_cost(cost_fun, Qi, Rii, Rij, Qni, x, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B)
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


### Quad Stuff
function costQuadcopter(Qi, Rii, Rij, Qni, x, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B)

    goal = x - xgoal
    rel_dist = (x[1:3] - x[10:12])'*I*(x[1:3] - x[10:12])
    if B 
        return 0.5*goal'*Qni*goal
    else            
        dx = x - xgoal
        dui = ui - uigoal
        duj = uj - ujgoal
        return 0.5*(dx'*Qi*dx + dui'*Rii*dui + duj'*Rij*duj) + ρ*(min(sqrt(rel_dist) - dmax, 0))^2 
        
    end
end

###################################################################################################

function costPointMassRH(game, i, Qi, Rii, Rij, Qni, x, ui, uj, B)

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


function quadratic_costRH(game, cost_fun, i, Qi, Rii, Rij, Qni, x, ui, uj, B)
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
        # Is this correct?
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
    # @show Q̂i
    # @show size(Q̂i), size(l̂i), size(R̂ii), size(r̂ii), size(R̂ij), size(r̂ij)
    R̂ij = zeros(nu, (Nplayer-1)*nu)
    for i =1:Nplayer-1
        R̂ij .+= R̂ijH[1+(i-1)*nu:i*nu,:] 
    end

    return cost, Q̂i, l̂i, R̂ii, r̂ii, R̂ij, r̂ij
end