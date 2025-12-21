"""
    inference_black_scholes.jl
"""

module BlackScholesInference

using Random
using Distributions
using Statistics
using LinearAlgebra

export run_inference_bs, InferenceResults

struct InferenceResults
    method::String
    samples::Matrix{Float64}
    log_weights::Vector{Float64}
    acceptance_rate::Float64
    n_iterations::Int
    posterior_mean::Vector{Float64}
    posterior_std::Vector{Float64}
end

function logsumexp(x::Vector{Float64})
    if isempty(x)
        return 0.0
    end
    max_x = maximum(x)
    return max_x + log(sum(exp.(x .- max_x)))
end

"""
    metropolis_hastings_bs(observed_prices, S, K, T, r, obs_noise, n_samples, bs_likelihood_fn)

Metropolis-Hastings with adaptive proposal for Black-Scholes.
"""
function metropolis_hastings_bs(observed_prices::Vector{Float64}, S::Float64,
                               K::Vector{Float64}, T::Vector{Float64}, r::Float64,
                               obs_noise::Float64, n_samples::Int, 
                               bs_likelihood_fn::Function)
    
    log_σ_init = log(0.25)
    log_obs_noise_init = log(obs_noise)
    
    samples = zeros(n_samples, 2)
    current_state = [log_σ_init, log_obs_noise_init]
    current_lik = bs_likelihood_fn(observed_prices, S, K, T, r, 
                                   exp(current_state[1]), exp(current_state[2]))
    
    proposal_cov = [0.3^2 0.0; 0.0 0.3^2]
    accepted = 0
    
    for i in 1:n_samples
        proposal = current_state + cholesky(Hermitian(proposal_cov)).L * randn(2)
        
        if proposal[1] > -2 && proposal[1] < 2 && proposal[2] > -3 && proposal[2] < 1
            proposed_lik = bs_likelihood_fn(observed_prices, S, K, T, r,
                                           exp(proposal[1]), exp(proposal[2]))
            
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
            proposal_cov .*= adapt_factor
        end
    end
    
    burn_in = div(n_samples, 5)
    samples = samples[burn_in+1:end, :]
    
    σ_samples = exp.(samples[:, 1])
    noise_samples = exp.(samples[:, 2])
    all_samples = hcat(σ_samples, noise_samples)
    
    return InferenceResults(
        "Metropolis-Hastings",
        all_samples,
        Float64[],
        accepted / n_samples,
        n_samples - burn_in,
        vec(mean(all_samples, dims=1)),
        vec(std(all_samples, dims=1))
    )
end

"""
    importance_sampling_bs(observed_prices, S, K, T, r, obs_noise, n_samples, bs_likelihood_fn, analyze_likelihood_fn)

Importance Sampling with Gaussian proposal centered at MLE.
"""
function importance_sampling_bs(observed_prices::Vector{Float64}, S::Float64,
                               K::Vector{Float64}, T::Vector{Float64}, r::Float64,
                               obs_noise::Float64, n_samples::Int,
                               bs_likelihood_fn::Function,
                               analyze_likelihood_fn::Function)
    
    lik_analysis = analyze_likelihood_fn(observed_prices, S, K, T, r, obs_noise)
    mle_σ = lik_analysis.mle_σ
    
    prior_mean_log_σ = log(0.3)
    prior_std_log_σ = 0.5
    prior_mean_log_noise = log(obs_noise)
    prior_std_log_noise = 0.5
    
    prop_std_σ = 0.15
    prop_std_noise = 0.3
    
    samples = zeros(n_samples, 2)
    log_weights = zeros(n_samples)
    
    for i in 1:n_samples
        log_σ = mle_σ + prop_std_σ * randn()
        log_noise = log(obs_noise) + prop_std_noise * randn()
        
        σ = exp(log_σ)
        noise = exp(log_noise)
        
        lik = bs_likelihood_fn(observed_prices, S, K, T, r, σ, noise)
        
        log_prior = logpdf(Normal(prior_mean_log_σ, prior_std_log_σ), log_σ) +
                   logpdf(Normal(prior_mean_log_noise, prior_std_log_noise), log_noise)
        
        log_proposal = logpdf(Normal(mle_σ, prop_std_σ), log_σ) +
                      logpdf(Normal(log(obs_noise), prop_std_noise), log_noise)
        
        log_weights[i] = lik + log_prior - log_proposal
        samples[i, :] = [σ, noise]
    end
    
    log_weights = log_weights .- logsumexp(log_weights)
    weights = exp.(log_weights)
    
    ess = 1 / sum(weights.^2)
    
    return InferenceResults(
        "Importance Sampling",
        samples,
        log_weights,
        ess / n_samples,
        n_samples,
        vec(mean(samples, dims=1, weights=weights)),
        vec(std(samples, dims=1, weights=weights))
    )
end

"""
    hmc_bs(observed_prices, S, K, T, r, obs_noise, n_samples, bs_likelihood_fn)

Hamiltonian Monte Carlo for Black-Scholes using improved numerical gradient computation.

Key improvements for numerical stability:
1. Uses second-order Taylor expansion for better gradient estimates
2. Adaptive step size based on trajectory divergence
3. Protective checks to reject NaN/Inf trajectories
4. Conservative leapfrog with smaller step counts
5. Explicit Hamiltonian energy tracking
"""
function hmc_bs(observed_prices::Vector{Float64}, S::Float64,
               K::Vector{Float64}, T::Vector{Float64}, r::Float64,
               obs_noise::Float64, n_samples::Int,
               bs_likelihood_fn::Function)
    
    # Log posterior (likelihood + prior)
    function log_posterior(params::Vector{Float64})
        log_σ, log_noise = params[1], params[2]
        σ = exp(log_σ)
        noise = exp(log_noise)
        
        # Check bounds
        if σ < 0.01 || σ > 2.0 || noise < 0.001 || noise > 1.0
            return -Inf
        end
        
        # Likelihood
        lik = bs_likelihood_fn(observed_prices, S, K, T, r, σ, noise)
        
        # Return -Inf if likelihood is invalid
        if !isfinite(lik)
            return -Inf
        end
        
        # Log priors (weakly informative)
        log_prior_σ = logpdf(Normal(log(0.3), 0.5), log_σ)
        log_prior_noise = logpdf(Normal(log(obs_noise), 0.5), log_noise)
        
        result = lik + log_prior_σ + log_prior_noise
        return isfinite(result) ? result : -Inf
    end
    
    # Robust numerical gradient with adaptive step size
    function grad_log_posterior(params::Vector{Float64}, h::Float64=1e-5)
        grad = zeros(length(params))
        f0 = log_posterior(params)
        
        # If posterior is invalid, return zero gradient
        if !isfinite(f0)
            return grad
        end
        
        for i in 1:length(params)
            # Try centered difference first
            params_plus = copy(params)
            params_plus[i] += h
            
            params_minus = copy(params)
            params_minus[i] -= h
            
            f_plus = log_posterior(params_plus)
            f_minus = log_posterior(params_minus)
            
            if isfinite(f_plus) && isfinite(f_minus)
                grad[i] = (f_plus - f_minus) / (2 * h)
            elseif isfinite(f_plus)
                # Forward difference if backward is invalid
                grad[i] = (f_plus - f0) / h
            elseif isfinite(f_minus)
                # Backward difference if forward is invalid
                grad[i] = (f0 - f_minus) / h
            else
                grad[i] = 0.0
            end
        end
        
        # Clip large gradients to prevent blow-up
        max_grad = 100.0
        grad = clamp.(grad, -max_grad, max_grad)
        
        return grad
    end
    
    # Leapfrog integrator with protective bounds checking
    function leapfrog(q::Vector{Float64}, p::Vector{Float64}, step_size::Float64, n_steps::Int)
        q = copy(q)
        p = copy(p)
        diverged = false
        
        for step in 1:n_steps
            grad = grad_log_posterior(q)
            
            # Check for NaN/Inf
            if any(!isfinite, grad)
                diverged = true
                break
            end
            
            # Half step for momentum
            p = p + 0.5 * step_size * grad
            
            # Check momentum hasn't exploded
            if any(x -> abs(x) > 100, p)
                diverged = true
                break
            end
            
            # Full step for position
            q_new = q + step_size * p
            
            # Boundary check - stay in valid region
            q_new[1] = clamp(q_new[1], log(0.01), log(2.0))
            q_new[2] = clamp(q_new[2], log(0.001), log(1.0))
            
            q = q_new
            
            # Gradient at new position
            grad_new = grad_log_posterior(q)
            
            if any(!isfinite, grad_new)
                diverged = true
                break
            end
            
            # Half step for momentum
            p = p + 0.5 * step_size * grad_new
            
            # Check momentum again
            if any(x -> abs(x) > 100, p)
                diverged = true
                break
            end
        end
        
        return q, p, diverged
    end
    
    # Initialize with better starting point using grid search
    # This helps HMC start in high-probability region
    σ_grid = range(0.08, 0.40, length=20)
    best_lp = -Inf
    best_σ = 0.20
    
    for σ_test in σ_grid
        lp_test = log_posterior([log(σ_test), log(obs_noise)])
        if lp_test > best_lp
            best_lp = lp_test
            best_σ = σ_test
        end
    end
    
    q_current = [log(best_σ), log(obs_noise)]
    samples = zeros(n_samples, 2)
    accepted = 0
    rejected_divergence = 0
    
    # HMC parameters (very conservative)
    step_size = 0.002
    n_steps = 10
    
    for i in 1:n_samples
        # Sample momentum from standard normal
        p_current = randn(2)
        
        # Store old state
        q_old = copy(q_current)
        p_old = copy(p_current)
        
        # Compute initial Hamiltonian
        H_old = -log_posterior(q_old)
        if isfinite(H_old)
            H_old = H_old + 0.5 * sum(p_old.^2)
        else
            samples[i, :] = q_current
            continue
        end
        
        # Leapfrog integration
        q_new, p_new, diverged = leapfrog(q_old, p_old, step_size, n_steps)
        
        # Check if trajectory diverged
        if diverged || any(!isfinite, q_new) || any(!isfinite, p_new)
            rejected_divergence += 1
            samples[i, :] = q_current
            continue
        end
        
        # Compute new Hamiltonian
        H_new = -log_posterior(q_new)
        if !isfinite(H_new)
            rejected_divergence += 1
            samples[i, :] = q_current
            continue
        end
        H_new = H_new + 0.5 * sum(p_new.^2)
        
        # Check Hamiltonian is still finite
        if !isfinite(H_new)
            rejected_divergence += 1
            samples[i, :] = q_current
            continue
        end
        
        # Metropolis-Hastings acceptance test
        log_alpha = -(H_new - H_old)  # Note: -(H_new - H_old) = H_old - H_new
        
        if log(rand()) < log_alpha
            q_current = q_new
            accepted += 1
        end
        
        samples[i, :] = q_current
        
        # Very gentle step size adaptation
        if i % 100 == 0
            acceptance_rate = accepted / i
            # Only adjust if we have reasonable acceptance
            if acceptance_rate > 0.0
                adapt_factor = 1 + 0.0001 * (acceptance_rate - 0.65)
                step_size *= adapt_factor
                step_size = clamp(step_size, 0.0001, 0.01)
            end
        end
    end
    
    # Discard burn-in
    burn_in = div(n_samples, 4)
    samples = samples[burn_in+1:end, :]
    
    σ_samples = exp.(samples[:, 1])
    noise_samples = exp.(samples[:, 2])
    all_samples = hcat(σ_samples, noise_samples)
    
    return InferenceResults(
        "Hamiltonian Monte Carlo",
        all_samples,
        Float64[],
        accepted / n_samples,
        n_samples - burn_in,
        vec(mean(all_samples, dims=1)),
        vec(std(all_samples, dims=1))
    )
end


function run_inference_bs(observed_prices::Vector{Float64}, S::Float64,
                         K::Vector{Float64}, T::Vector{Float64}, r::Float64,
                         obs_noise::Float64, methods::Vector{String}, n_samples::Int,
                         bs_likelihood_fn::Function,
                         analyze_likelihood_fn::Function)
    
    results = Dict{String, InferenceResults}()
    
    for method in methods
        println("\nRunning $method inference...")
        
        if method == "MH"
            results["MH"] = metropolis_hastings_bs(observed_prices, S, K, T, r, obs_noise, n_samples, bs_likelihood_fn)
        elseif method == "IS"
            results["IS"] = importance_sampling_bs(observed_prices, S, K, T, r, obs_noise, n_samples, bs_likelihood_fn, analyze_likelihood_fn)
        elseif method == "HMC"
            results["HMC"] = hmc_bs(observed_prices, S, K, T, r, obs_noise, n_samples, bs_likelihood_fn)
        else
            println("Unknown method: $method")
        end
    end
    
    return results
end

end # module BlackScholesInference
