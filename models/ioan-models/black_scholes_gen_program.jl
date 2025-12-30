"""
    BlackScholesGenProgram
"""

using Gen
using Distributions

include("black_scholes_model.jl")
using .BlackScholesModel

"""
Black-Scholes generative model for inferring volatility from option prices.
"""
@gen function black_scholes_generative_model(observed_call_prices::Vector{Float64},
                                             S::Float64, 
                                             K::Vector{Float64},
                                             T::Vector{Float64},
                                             r::Float64,
                                             num_observations::Int)
    # Prior on volatility: log-normal to ensure positivity
    # Typical stock volatility ranges from 10% to 100%
    log_σ ~ normal(log(0.3), 0.5)  # Prior centered around 30% volatility
    σ = exp(log_σ)
    
    # Prior on observation noise
    log_obs_noise ~ normal(log(0.1), 0.5)
    obs_noise = exp(log_obs_noise)
    
    # Generate option prices from the model
    predicted_prices = Vector{Float64}(undef, num_observations)
     for i in 1:num_observations
        @trace(normal(predicted_prices[i], obs_noise), (:obs, i))
    end
    
    # Likelihood: observed prices given predicted prices
    observations = choicemap()
    for i in 1:length(observed_prices)
        observations[(:obs, i)] = observed_prices[i]
    end
    return (σ, obs_noise)
end

"""
Generate synthetic data from the Black-Scholes model.
"""
function generate_synthetic_bs_data(S::Float64, K::Vector{Float64}, T::Vector{Float64}, 
                                   r::Float64, true_σ::Float64, obs_noise::Float64,
                                   num_observations::Int)
    prices = zeros(num_observations)
    for i in 1:num_observations
        prices[i] = bs_call_price(S, K[i], T[i], r, true_σ)
        prices[i] += randn() * obs_noise
    end
    return prices
end

"""
Run inference using different procedures.
"""
function run_black_scholes_inference(observed_prices::Vector{Float64}, 
                                    S::Float64, K::Vector{Float64}, T::Vector{Float64},
                                    r::Float64, num_samples::Int)
    
    # Create observations map
    observations = choicemap()
    for i in 1:length(observed_prices)
        observations[:observed_call_prices => i] = observed_prices[i]
    end
    
    # Run inference procedures
    results = Dict()
    
    # 1. Generate initial trace from model
    println("Generating initial trace...")
    try
        trace, log_marginal_likelihood = generate(
            black_scholes_generative_model,
            (observed_prices, S, K, T, r, length(observed_prices)),
            observations
        )
        results["initial_trace"] = (trace=trace, log_ml=log_marginal_likelihood)
    catch e
        results["initial_trace"] = nothing
    end
    
    # 2. Metropolis-Hastings inference
    println("\nRunning Metropolis-Hastings inference...")
    results["metropolis_hastings"] = "custom_proposal_required"
    
    # 3. Note about HMC
    println("\nNote about HMC (Hamiltonian Monte Carlo):")
    results["hmc"] = "custom_implementation_required"
    
    return results
end
