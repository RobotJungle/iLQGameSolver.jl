using LinearAlgebra

function solveILQGame(dynamics, costf, x₀, xgoal, u1goal, u2goal, Q1, Q2, Qn1, Qn2, R11, R12, R21, R22, umin, umax, dmax, ρ, dt, H)

    n = length(x₀)
    m1 = size(R11)[1]
    m2 = size(R22)[1]
    k_steps = trunc(Int, H/dt) 

    x̂ = zeros(k_steps, n) # 1500 x need
    û₁ = zeros(k_steps, m1)
    û₂ = zeros(k_steps, m2) 
    û = cat(û₁, û₂, dims=2)
    P = rand(m1 + m2, n, k_steps)*0.01
    α = rand(m1 + m2, k_steps)*0.01

    # Rollout players
    xₜ, uₜ = Rollout_RK4(dynamics, x₀, x̂, û, umin, umax, H, dt, P, α, 0.0)

    Aₜ = zeros(Float32, (n, n, k_steps))
    B1ₜ = zeros(Float32, (n, m1, k_steps))
    B2ₜ = zeros(Float32, (n, m2, k_steps))

    Q1ₜ = zeros(Float32, (n, n, k_steps))
    Q2ₜ = zeros(Float32, (n, n, k_steps))

    l1ₜ = zeros(Float32, (n, k_steps))
    l2ₜ = zeros(Float32, (n, k_steps))

    ## CHECK THIS: When changing to agents with different input size
    R11ₜ = zeros(Float32, (m1, m1, k_steps))
    R12ₜ = zeros(Float32, (m1, m2, k_steps))
    R22ₜ = zeros(Float32, (m2, m2, k_steps))
    R21ₜ = zeros(Float32, (m2, m1, k_steps))

    r11ₜ = zeros(Float32, (m1, k_steps))
    r12ₜ = zeros(Float32, (m1, k_steps))
    r22ₜ = zeros(Float32, (m2, k_steps))
    r21ₜ = zeros(Float32, (m2, k_steps))

    converged = false
    # u1goal = [0; 0]; u2goal = [0; 0]; 
    # ugoal = cat(u1goal, u2goal, dims=1)
    βreg = 1.0
    while !converged
        converged = isConverged(xₜ, x̂, tol = 1e-2)
        total_cost1 = 0
        total_cost2 = 0
        u1ₜ = uₜ[:, 1:m1]
        u2ₜ = uₜ[:, m1+1:m1+m2]
        for t = 1:(k_steps-1)
            
            A, B = lin_dyn_discrete(dynamics, xₜ[t,:], uₜ[t,:], dt)
            #A, B = lin_dyn_discrete(dynamics, xₜ[t,:] - x̂[t,:], uₜ[t,:] - û[t,:], dt)

            Aₜ[:,:,t] = A
            #B1Shape = size(B1)
            #B2Shape = size(B2)
            B1ₜ[:,:,t] = B[:, 1:m1]
            B2ₜ[:,:,t] = B[:, m1+1:m1+m2] #end


            #Player 1 cost
            costval1, Q1ₜ[:,:,t], l1ₜ[:,t], R11ₜ[:,:,t], r11ₜ[:,t], R12ₜ[:,:,t], r12ₜ[:,t] = 
            quadratic_cost(costf, Q1, R11, R12, Qn1, xₜ[t,:], u1ₜ[t,:], u2ₜ[t,:], xgoal, u1goal, u2goal, dmax, ρ, false)
            
            #Player 2 cost
            costval2, Q2ₜ[:,:,t], l2ₜ[:,t], R22ₜ[:,:,t], r22ₜ[:,t], R21ₜ[:,:,t], r21ₜ[:,t] = 
            quadratic_cost(costf, Q2, R22, R21, Qn2, xₜ[t,:], u2ₜ[t,:], u1ₜ[t,:], xgoal, u2goal, u1goal, dmax, ρ, false)
            
            # Regularization
            while !isposdef(Q1ₜ[:,:,t])
                Q1ₜ[:,:,t] = Q1ₜ[:,:,t] + βreg*I
            end
            while !isposdef(Q2ₜ[:,:,t])
                Q2ₜ[:,:,t] = Q2ₜ[:,:,t] + βreg*I
            end

            total_cost1 += costval1
            total_cost2 += costval2
        end
        #Player 1 Terminal cost
        costval1, Q1ₜ[:,:,end], l1ₜ[:,end], R11ₜ[:,:,end], r11ₜ[:,end], R12ₜ[:,:,end], r12ₜ[:,end] = 
        quadratic_cost(costf, Q1, R11, R12, Qn1, xₜ[end,:], u1ₜ[end,:], u2ₜ[end,:], xgoal, u1goal, u2goal, dmax, ρ, true)
        
        #Player 2 Terminal cost
        costval2, Q2ₜ[:,:,end], l2ₜ[:,end], R22ₜ[:,:,end], r22ₜ[:,end], R21ₜ[:,:,end], r21ₜ[:,end] = 
        quadratic_cost(costf, Q2, R22, R21, Qn2, xₜ[end,:], u2ₜ[end,:], u2ₜ[end,:], xgoal, u2goal, u1goal, dmax, ρ, true)

        total_cost1 += costval1
        total_cost2 += costval2

        P₁, P₂, α₁, α₂ = lqGame!(Aₜ, B1ₜ, B2ₜ, Q1ₜ, Q2ₜ, l1ₜ, l2ₜ, R11ₜ, R12ₜ, R21ₜ, R22ₜ, r11ₜ, r22ₜ, r12ₜ, r21ₜ, k_steps)
        P = cat(P₁, P₂, dims=1)
        α = cat(α₁ , α₂, dims=1)
        
        x̂ = xₜ
        û₁ = u1ₜ
        û₂ = u2ₜ
        û = cat(û₁, û₂, dims=2)

        # Rollout players with new control law
        xₜ, uₜ = Rollout_RK4(dynamics, x₀, x̂, û, umin, umax, H, dt, P, α, 0.5)
    end

    return xₜ, uₜ
end
