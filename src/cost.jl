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
    I neglected the mixed paritals in the hessian
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

function costPointMassRH(Qi, Rii, Rij, Qni, x, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B)

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


function quadratic_costRH(cost_fun, Qi, Rii, Rij, Qni, x, ui, uj, xgoal, uigoal, ujgoal, dmax, ρ, B)
    """
    2nd order Taylor expansion of cost at t
    I neglected the mixed paritals in the hessian
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