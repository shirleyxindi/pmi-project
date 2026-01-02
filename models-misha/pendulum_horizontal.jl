using Gen
using Plots

# The same as the original pendulum model
function pendulum_simulator(L, b, theta_init, dtheta_init, T, dt)
    g = 9.81
    trajectory = Float64[]
    
    # The angle and angular velocity
    theta = theta_init
    dtheta = dtheta_init
    
    for t in 1:T
        # Acceleration: -(g/L)sin(theta) - (b * dtheta)
        ddtheta = -(g/L) * sin(theta) - (b * dtheta)
        dtheta += ddtheta * dt
        theta += dtheta * dt
        
        push!(trajectory, theta)
    end
    
    return trajectory
end

@gen function pendulum_horizontal_model(T::Int)
    # Priors
    L ~ uniform(0.5, 5.0) 
    b ~ uniform(0.0, 0.5)

    # Constants
    dt = 0.05
    theta_init = 1.5
    dtheta_init = 0.0
    
    # We simulate the full angular trajectory first
    true_angles = pendulum_simulator(L, b, theta_init, dtheta_init, T, dt)
    
    # We observe horizontal position x instead of the angle theta. Now L and theta are coupled together in the observation
    noise_level = 0.1
    for i in 1:T
        true_x = L * sin(true_angles[i])
        {(:x, i)} ~ normal(true_x, noise_level)
    end
    
    return true_angles
end

function main()
    # The amount of time steps to simulate
    T = 100
    
    # Simulate the trace
    trace = Gen.simulate(pendulum_horizontal_model, (T,))
    
    choices = get_choices(trace)
    sampled_L = choices[:L]
    
    # Get the true angles
    true_angles = get_retval(trace)
    # Now we derive the true horizontal position
    true_x = [sampled_L * sin(theta) for theta in true_angles]
    
    # Get the noisy observations
    observed_x = [choices[(:x, i)] for i in 1:T]
    
    println("Sampled Length: ", round(sampled_L, digits=3))
    
    p = plot(true_x, 
             label="True Trajectory (L*sin(θ))", 
             lw=2, 
             xlabel="Time Steps", 
             ylabel="Horizontal Position (meters)",
             title="Pendulum Simulation")
             
    scatter!(observed_x, label="Observations", alpha=0.6)
    display(p)
end

main()