"""
    inference_merton.jl
"""

module MertonInference

using Random
using Distributions
using Statistics

export run_inference_merton, InferenceResults

struct InferenceResults
    method::String
    samples::Matrix{Float64}
    log_weights::Vector{Float64}
    acceptance_rate::Float64
    n_iterations::Int
    posterior_mean::Vector{Float64}
    posterior_std::Vector{Float64}
    jump_count_estimates::Vector{Int}
end

"""
    metropolis_hastings_merton(obs_prices, S, K, T, r, obs_noise, n_samples, merton_likelihood_fn)
"""
function metropolis_hastings_merton(obs_prices::Vector{Float64}, S::Float64,
                                   K::Vector{Float64}, T::Vector{Float64}, r::Float64,
                                   obs_noise::Float64, n_samples::Int,
                                   merton_likelihood_fn::Function)
    
    current_state = [log(0.25), log(0.5), log(0.05), log(0.15), log(obs_noise)]
    current_lik = merton_likelihood_fn(obs_prices, S, K, T, r,
                                      exp(current_state[1]), exp(current_state[2]),
                                      exp(current_state[3]), exp(current_state[4]),
                                      exp(current_state[5]))
    
    samples = zeros(n_samples, 5)
    proposal_cov = Diagonal([0.3^2, 0.4^2, 0.4^2, 0.4^2, 0.3^2])
    accepted = 0
    
    for i in 1:n_samples
        proposal = current_state + cholesky(Hermitian(Matrix(proposal_cov))).L * randn(5)
        
        if all(proposal .> [-2, -2, -3, -3, -3]) && all(proposal .< [2, 2, 2, 2, 1])
            proposed_lik = merton_likelihood_fn(obs_prices, S, K, T, r,
                                               exp(proposal[1]), exp(proposal[2]),
                                               exp(proposal[3]), exp(proposal[4]),
                                               exp(proposal[5]))
            
            log_alpha = proposed_lik - current_lik
            
            if log(rand()) < log_alpha
                current_state = proposal
                current_lik = proposed_lik
                accepted += 1
            end
        end
        
        samples[i, :] = current_state
        
        if i % 100 == 0
            adapt_factor = 1 + 0.01 * (accepted / i - 0.234)
            proposal_cov = proposal_cov .* adapt_factor
        end
    end
    
    burn_in = div(n_samples, 5)
    samples = samples[burn_in+1:end, :]
    
    σ_samples = exp.(samples[:, 1])
    λ_samples = exp.(samples[:, 2])
    μj_samples = exp.(samples[:, 3])
    σj_samples = exp.(samples[:, 4])
    noise_samples = exp.(samples[:, 5])
    all_samples = hcat(σ_samples, λ_samples, μj_samples, σj_samples, noise_samples)
    
    return InferenceResults(
        "Metropolis-Hastings",
        all_samples,
        Float64[],
        accepted / n_samples,
        n_samples - burn_in,
        vec(mean(all_samples, dims=1)),
        vec(std(all_samples, dims=1)),
        Int[]
    )
end

"""
    reversible_jump_mcmc_merton(obs_prices, S, K, T, r, obs_noise, n_samples, merton_likelihood_fn)
"""
function reversible_jump_mcmc_merton(obs_prices::Vector{Float64}, S::Float64,
                                    K::Vector{Float64}, T::Vector{Float64}, r::Float64,
                                    obs_noise::Float64, n_samples::Int,
                                    merton_likelihood_fn::Function)
    
    current_state = [log(0.25), log(0.5), log(0.05), log(0.15), log(obs_noise)]
    current_lik = merton_likelihood_fn(obs_prices, S, K, T, r,
                                      exp(current_state[1]), exp(current_state[2]),
                                      exp(current_state[3]), exp(current_state[4]),
                                      exp(current_state[5]))
    
    samples = zeros(n_samples, 5)
    proposal_cov = Diagonal([0.25^2, 0.35^2, 0.35^2, 0.35^2, 0.25^2])
    accepted_param = 0
    accepted_jump = 0
    
    for i in 1:n_samples
        if rand() < 0.8
            proposal = current_state + cholesky(Hermitian(Matrix(proposal_cov))).L * randn(5)
            
            if all(proposal .> [-2, -2, -3, -3, -3]) && all(proposal .< [2, 2, 2, 2, 1])
                proposed_lik = merton_likelihood_fn(obs_prices, S, K, T, r,
                                                   exp(proposal[1]), exp(proposal[2]),
                                                   exp(proposal[3]), exp(proposal[4]),
                                                   exp(proposal[5]))
                
                log_alpha = proposed_lik - current_lik
                
                if log(rand()) < log_alpha
                    current_state = proposal
                    current_lik = proposed_lik
                    accepted_param += 1
                end
            end
        else
            λ_factor = exp(0.3 * randn())
            proposal = copy(current_state)
            proposal[2] += log(λ_factor)
            
            if proposal[2] > -2 && proposal[2] < 2
                proposed_lik = merton_likelihood_fn(obs_prices, S, K, T, r,
                                                   exp(proposal[1]), exp(proposal[2]),
                                                   exp(proposal[3]), exp(proposal[4]),
                                                   exp(proposal[5]))
                
                log_alpha = proposed_lik - current_lik + log(abs(λ_factor))
                
                if log(rand()) < log_alpha
                    current_state = proposal
                    current_lik = proposed_lik
                    accepted_jump += 1
                end
            end
        end
        
        samples[i, :] = current_state
    end
    
    burn_in = div(n_samples, 5)
    samples = samples[burn_in+1:end, :]
    
    σ_samples = exp.(samples[:, 1])
    λ_samples = exp.(samples[:, 2])
    μj_samples = exp.(samples[:, 3])
    σj_samples = exp.(samples[:, 4])
    noise_samples = exp.(samples[:, 5])
    all_samples = hcat(σ_samples, λ_samples, μj_samples, σj_samples, noise_samples)
    
    jump_estimates = round.(Int, λ_samples .* T[1])
    
    return InferenceResults(
        "Reversible Jump MCMC",
        all_samples,
        Float64[],
        (accepted_param + accepted_jump) / n_samples,
        n_samples - burn_in,
        vec(mean(all_samples, dims=1)),
        vec(std(all_samples, dims=1)),
        jump_estimates
    )
end

"""
    run_inference_merton(obs_prices, S, K, T, r, obs_noise, methods, n_samples, merton_likelihood_fn)
"""
function run_inference_merton(obs_prices::Vector{Float64}, S::Float64,
                             K::Vector{Float64}, T::Vector{Float64}, r::Float64,
                             obs_noise::Float64, methods::Vector{String}, n_samples::Int,
                             merton_likelihood_fn::Function)
    
    results = Dict{String, InferenceResults}()
    
    for method in methods
        println("\nRunning $method inference on Merton model...")
        
        if method == "MH"
            results["MH"] = metropolis_hastings_merton(obs_prices, S, K, T, r, obs_noise, n_samples, merton_likelihood_fn)
        elseif method == "RJMCMC"
            results["RJMCMC"] = reversible_jump_mcmc_merton(obs_prices, S, K, T, r, obs_noise, n_samples, merton_likelihood_fn)
        else
            println("Unknown method: $method")
        end
    end
    
    return results
end

end # module MertonInference
