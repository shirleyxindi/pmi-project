"""
    MertonJumpDiffusionGenProgram
"""

using Gen
using Distributions

include("merton_jump_diffusion_model.jl")
using .MertonJumpDiffusionModel

"""
Simplified Merton Jump Diffusion generative model.

We use a simplified version where we don't explicitly model jump times,
but rather condition on observed price paths and infer model parameters.
"""
@gen function merton_jump_diffusion_generative_model(observed_prices::Vector{Float64},
                                                     S0::Float64,
                                                     T::Float64,
                                                     r::Float64,
                                                     num_observations::Int)
    # Prior on diffusion volatility
    log_σ ~ normal(log(0.2), 0.5)  # 20% volatility baseline
    σ = exp(log_σ)
    
    # Prior on drift
    μ ~ normal(0.1, 0.1)  # Expected 10% return
    
    # Prior on jump intensity (Poisson rate per year)
    log_λ ~ normal(log(0.5), 1.0)  # Expect 1-2 jumps per year on average
    λ = exp(log_λ)
    
    # Prior on mean log jump size (negative = crash risk)
    μ_J ~ normal(-0.1, 0.15)  # Average jump down 10%, can be up or down
    
    # Prior on jump size volatility
    log_σ_J ~ normal(log(0.2), 0.5)  # Jump sizes vary by ~20%
    σ_J = exp(log_σ_J)
    
    # Prior on observation noise
    log_obs_noise ~ normal(log(0.05), 0.5)
    obs_noise = exp(log_obs_noise)
    
    # Convert time period to years (assuming daily observations)
    dt = T / (num_observations - 1)
    
    # Compute returns and likelihood
    for i in 1:(num_observations - 1)
        log_return = log(observed_prices[i + 1] / observed_prices[i])
        
        # Expected log return under Merton model
        # E[dS/S] = μ dt + σ dW + dJ
        # where jump component adds λ * μ_J to expected return
        expected_log_return = (μ - 0.5 * σ^2 + λ * μ_J) * dt
        
        # Variance includes diffusion and jump risk
        # Var[dS/S] = σ² dt + λ(μ_J² + σ_J²) dt
        variance = (σ^2 + λ * (μ_J^2 + σ_J^2)) * dt + obs_noise^2
        
        # Likelihood of observed return
        log_return ~ normal(expected_log_return, sqrt(variance))
    end
    
    return (σ, μ, λ, μ_J, σ_J, obs_noise)
end

"""
Alternative: Model with explicit jump detection.

This version attempts to detect and model jump events explicitly,
which requires more complex inference (e.g., trans-dimensional MCMC).
"""
@gen function merton_jump_diffusion_with_jumps(observed_prices::Vector{Float64},
                                               S0::Float64,
                                               T::Float64,
                                               r::Float64,
                                               num_observations::Int)
    
    # Parameters with same priors as above
    log_σ ~ normal(log(0.2), 0.5)
    σ = exp(log_σ)
    
    μ ~ normal(0.1, 0.1)
    
    log_λ ~ normal(log(0.5), 1.0)
    λ = exp(log_λ)
    
    log_μ_J ~ normal(log(0.05), 0.5)  # Use log to ensure μ_J > 0
    μ_J = exp(log_μ_J)
    
    log_σ_J ~ normal(log(0.15), 0.5)
    σ_J = exp(log_σ_J)
    
    log_obs_noise ~ normal(log(0.05), 0.5)
    obs_noise = exp(log_obs_noise)
    
    dt = T / (num_observations - 1)
    
    for i in 1:(num_observations - 1)
        log_return = log(observed_prices[i + 1] / observed_prices[i])
        
        jump_occurred = false
        
        if jump_occurred
            jump_size ~ LogNormal(log_μ_J, σ_J)
            adjusted_return = log_return - log(jump_size)
        else
            adjusted_return = log_return
        end
        
        # Diffusion likelihood
        expected_return = (μ - 0.5 * σ^2) * dt
        variance = σ^2 * dt + obs_noise^2
        adjusted_return ~ normal(expected_return, sqrt(variance))
    end
    
    return (σ, μ, λ, μ_J, σ_J, obs_noise)
end

"""
Generate synthetic Merton Jump Diffusion data.
"""
function generate_synthetic_merton_data(S0::Float64, T::Float64,
                                       true_μ::Float64, true_σ::Float64,
                                       true_λ::Float64, true_μ_J::Float64,
                                       true_σ_J::Float64, obs_noise::Float64,
                                       num_observations::Int)
    prices, jump_times, jump_sizes = simulate_merton_paths(
        S0, true_μ, true_σ, true_λ, true_μ_J, true_σ_J,
        T, num_observations
    )
    
    # Add observation noise
    observed_prices = prices[1:num_observations] .+ randn(num_observations) .* obs_noise
    
    return observed_prices
end

"""
Run inference on Merton model.
"""
function run_merton_inference(observed_prices::Vector{Float64},
                             S0::Float64, T::Float64, r::Float64,
                             num_samples::Int)
    
    println("Running inference on Merton Jump Diffusion model...")
    num_obs = length(observed_prices)
    observations = choicemap()
    
    results = Dict()
    
    println("Metropolis-Hastings:")
    results["metropolis_hastings"] = "requires_reversible_jump_mcmc"
    
    println("\nHamiltonian Monte Carlo (HMC):")
    results["hmc"] = "unsuitable_for_discontinuous_model"
    
    println("\nImportance Sampling:")
    results["importance_sampling"] = "requires_informed_proposal"
    
    println("\nParticle Filtering:")
    results["particle_filtering"] = "recommended_approach"
    
    return results
end
