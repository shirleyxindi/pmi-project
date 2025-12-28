using Gen
using Plots

# An Euler integrator for a damped pendulum with parameters length (L), damping (b), time_step (dt), total_steps (T)
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

@gen function pendulum_model(T::Int)
    # The priors (parameters we want to infer): length and damping
    L ~ uniform(0.5, 5.0) 
    b ~ uniform(0.0, 0.5)

    # Fixed constants for this experiment
    dt = 0.05
    theta_init = 1.5
    dtheta_init = 0.0
    
    true_trajectory = pendulum_simulator(L, b, theta_init, dtheta_init, T, dt)
    
    # We observe the trajectory with some measurement noise
    noise_level = 0.1
    for i in 1:T
        {(:y, i)} ~ normal(true_trajectory[i], noise_level)
    end
    
    return true_trajectory
end

function main()
    # The amount of time steps to simulate
    T = 100

    # Simulate a trace from the prior with a random L and b
    trace = Gen.simulate(pendulum_model, (T,))

    choices = get_choices(trace)
    sampled_L = choices[:L]
    sampled_b = choices[:b]
    
    # Get the return value (the true trajectory without noise)
    true_trajectory = get_retval(trace)

    # Get the observed data (the trajectory with noise)
    observed_data = [choices[(:y, i)] for i in 1:T]

    println("Simulation Results:")
    println("Sampled Length (L): ", round(sampled_L, digits=3))
    println("Sampled Damping (b): ", round(sampled_b, digits=3))
    println("First 5 observed data points: ", observed_data[1:5])

    p = plot(true_trajectory, 
             label="True Trajectory", 
             lw=2, 
             title="Pendulum Simulation",
             xlabel="Time Steps",
             ylabel="Angle (radians)")
             
    scatter!(observed_data, 
             label="Observations", 
             alpha=0.6,
             markerstrokewidth=0)
             
    display(p)
end

main()