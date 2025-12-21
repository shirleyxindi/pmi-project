"""
Part II: Comprehensive Inference Method Comparison
Compares three inference methods (MH, IS, HMC) on Black-Scholes model
with detailed convergence analysis and efficiency metrics.
"""

using Distributions
using Random
using Statistics
using Printf
using LinearAlgebra

Random.seed!(123)

include("black_scholes_model.jl")
using .BlackScholesModel

println("="^90)
println("PART II: INFERENCE METHOD COMPARISON")
println("="^90)

# ============================================================================
# METRICS: Autocorrelation and ESS
# ============================================================================

function compute_autocorr_time(samples::Vector{Float64}, max_lag::Int=100)
    """Integrated autocorrelation time"""
    n = length(samples)
    x = samples .- mean(samples)
    c0 = sum(x.^2) / n
    
    if c0 < 1e-10
        return 1.0
    end
    
    tau = 0.5
    for k in 1:min(max_lag, n÷2)
        rho_k = sum(x[1:n-k] .* x[k+1:n]) / (n * c0)
        if abs(rho_k) < 0.02
            break
        end
        tau += rho_k
    end
    
    return max(1.0, tau)
end

function compute_ess(samples::Vector{Float64})
    """Effective Sample Size"""
    tau = compute_autocorr_time(samples)
    return length(samples) / tau
end

# ============================================================================
# TEST SETUP: Black-Scholes Volatility Estimation
# ============================================================================

println("\n" * "="^90)
println("INFERENCE SCENARIO: Black-Scholes Volatility & Noise Estimation")
println("="^90)

# Setup: realistic test case
S0 = 100.0
K = [95.0, 100.0, 105.0, 110.0]
T = [0.25, 0.5, 0.75, 1.0]
r = 0.05
true_σ = 0.20
true_obs_noise = 0.01  # 1% measurement noise

println("\nScenario Parameters:")
println("  Underlying price (S₀):     \$$(S0)")
println("  Risk-free rate (r):        $(r)")
println("  True volatility (σ):       $(true_σ)")
println("  True obs_noise:            $(true_obs_noise)")
println("  Option prices:             $(length(K)) calls at various strikes/maturities")

# Generate synthetic observed prices
observed_prices = Float64[]
for i in 1:length(K)
    price = bs_call_price(S0, K[i], T[i], r, true_σ)
    noise = true_obs_noise * price * randn()
    obs = price + noise
    obs = max(obs, 0.1)
    push!(observed_prices, obs)
end

println("  Data noise:                Proportional to price (~1%)")

# ============================================================================
# METHOD 1: METROPOLIS-HASTINGS
# ============================================================================

println("\n" * "="^90)
println("METHOD 1: Metropolis-Hastings (Adaptive Proposal)")
println("="^90)
println("\nAlgorithm: Random walk in log-parameter space")
println("  • Proposal: 2D Gaussian random walk")
println("  • Adaptation: Target acceptance rate = 23.4%")
println("  • Burn-in: First 25% of samples")

function run_mh_inference(n_samples::Int)
    """Run adaptive Metropolis-Hastings for Black-Scholes"""
    
    log_σ_init = log(true_σ + 0.02)
    log_noise_init = log(true_obs_noise + 0.001)
    samples = zeros(n_samples, 2)
    current_params = [log_σ_init, log_noise_init]
    
    function log_likelihood(log_σ, log_noise)
        σ = exp(log_σ)
        noise = exp(log_noise)
        
        if σ < 0.05 || σ > 0.8 || noise < 0.001 || noise > 0.1
            return -1e10
        end
        
        ll = 0.0
        for i in 1:length(observed_prices)
            pred = bs_call_price(S0, K[i], T[i], r, σ)
            ll += logpdf(Normal(pred, noise * pred), observed_prices[i])
        end
        return ll
    end
    
    current_ll = log_likelihood(current_params[1], current_params[2])
    proposal_cov = [0.15^2 0.0; 0.0 0.25^2]
    n_accepted = 0
    
    for iter in 1:n_samples
        prop = current_params + cholesky(Hermitian(proposal_cov)).L * randn(2)
        prop_ll = log_likelihood(prop[1], prop[2])
        
        if prop_ll > -1e10 && log(rand()) < (prop_ll - current_ll)
            current_params = prop
            current_ll = prop_ll
            n_accepted += 1
        end
        
        samples[iter, :] = current_params
        
        if iter % 100 == 0 && iter > 500
            accept_rate = n_accepted / iter
            adapt = 1.0 + 0.01 * (accept_rate - 0.234)
            proposal_cov .*= adapt
        end
    end
    
    return samples, n_accepted / n_samples
end

println("\nResults: Adaptive MH for increasing sample counts")
println(@sprintf "%-10s %10s %10s %10s %10s %10s" "N_Samples" "σ_hat" "σ_std" "ESS" "ESS/Iter" "Accept%")
println("-"^90)

mh_results = Dict()

for n_samp in [500, 1000, 2000, 5000]
    samples, accept_rate = run_mh_inference(n_samp)
    burn_in = div(n_samp, 4)
    σ_chain = exp.(samples[burn_in+1:end, 1])
    
    σ_mean = mean(σ_chain)
    σ_std = std(σ_chain)
    ess_σ = compute_ess(σ_chain)
    ess_per_iter = ess_σ / n_samp
    
    mh_results[n_samp] = (mean=σ_mean, std=σ_std, ess=ess_σ, accept=accept_rate)
    
    @printf "%-10d %10.5f %10.5f %10.1f %10.4f %10.1f%%\n" n_samp σ_mean σ_std ess_σ ess_per_iter (accept_rate*100)
end

# ============================================================================
# METHOD 2: IMPORTANCE SAMPLING
# ============================================================================

println("\n" * "="^90)
println("METHOD 2: Importance Sampling (Gaussian Proposal)")
println("="^90)
println("\nAlgorithm: Independent proposal-based sampling")
println("  • Proposal: Gaussian centered at MLE estimate")
println("  • Weighting: Importance weights from posterior ratio")
println("  • ESS: Computed from weight distribution")

function run_is_inference(n_samples::Int)
    """Run importance sampling with Gaussian proposal"""
    
    # MLE via grid search
    σ_grid = 0.08:0.02:0.40
    best_σ = 0.15
    best_ll = -Inf
    
    for σ_try in σ_grid
        ll = 0.0
        for i in 1:length(observed_prices)
            pred = bs_call_price(S0, K[i], T[i], r, σ_try)
            ll += logpdf(Normal(pred, true_obs_noise * pred), observed_prices[i])
        end
        if ll > best_ll
            best_ll = ll
            best_σ = σ_try
        end
    end
    
    samples = zeros(n_samples)
    log_weights = zeros(n_samples)
    proposal_σ = 0.08
    
    for i in 1:n_samples
        log_σ = log(best_σ) + proposal_σ * randn()
        σ = exp(log_σ)
        
        if σ > 0.05 && σ < 0.8
            ll = 0.0
            for j in 1:length(observed_prices)
                pred = bs_call_price(S0, K[j], T[j], r, σ)
                ll += logpdf(Normal(pred, true_obs_noise * pred), observed_prices[j])
            end
            
            log_prior = logpdf(Normal(log(0.25), 0.4), log_σ)
            log_proposal = logpdf(Normal(log(best_σ), proposal_σ), log_σ)
            
            log_weights[i] = ll + log_prior - log_proposal
            samples[i] = σ
        else
            log_weights[i] = -Inf
            samples[i] = best_σ
        end
    end
    
    max_w = maximum(log_weights[isfinite.(log_weights)])
    weights = exp.(log_weights .- max_w)
    weights = weights ./ sum(weights)
    
    return samples, weights
end

println("\nResults: Importance Sampling for increasing sample counts")
println(@sprintf "%-10s %10s %10s %10s %10s" "N_Samples" "σ_hat" "σ_std" "ESS" "ESS/Iter")
println("-"^90)

is_results = Dict()

for n_samp in [500, 1000, 2000, 5000]
    samples, weights = run_is_inference(n_samp)
    
    σ_mean = sum(samples .* weights)
    σ_var = sum((samples .- σ_mean).^2 .* weights)
    σ_std = sqrt(σ_var)
    
    ess_σ = 1.0 / sum(weights.^2)
    ess_per_iter = ess_σ / n_samp
    
    is_results[n_samp] = (mean=σ_mean, std=σ_std, ess=ess_σ)
    
    @printf "%-10d %10.5f %10.5f %10.1f %10.4f\n" n_samp σ_mean σ_std ess_σ ess_per_iter
end

# ============================================================================
# METHOD 3: HAMILTONIAN MONTE CARLO
# ============================================================================

println("\n" * "="^90)
println("METHOD 3: Hamiltonian Monte Carlo (Numerical Gradient)")
println("="^90)
println("\nAlgorithm: Gradient-based sampling with leapfrog integrator")
println("  • Gradient: Numerical finite differences (h=1e-4)")
println("  • Integration: Leapfrog with 15 steps")
println("  • Adaptation: Target acceptance rate = 65%")
println("  • Burn-in: First 25% of samples")

function run_hmc_inference(n_samples::Int)
    """Run HMC with improved numerical gradient and initialization"""
    
    function log_posterior(log_σ::Float64, log_noise::Float64)
        σ = exp(log_σ)
        noise = exp(log_noise)
        
        if σ < 0.05 || σ > 0.8 || noise < 0.001 || noise > 0.1
            return -1e10
        end
        
        ll = 0.0
        for i in 1:length(observed_prices)
            pred = bs_call_price(S0, K[i], T[i], r, σ)
            ll += logpdf(Normal(pred, noise * pred), observed_prices[i])
        end
        
        log_prior_σ = logpdf(Normal(log(0.25), 0.4), log_σ)
        log_prior_noise = logpdf(Normal(log(true_obs_noise), 0.5), log_noise)
        
        return ll + log_prior_σ + log_prior_noise
    end
    
    function grad_log_posterior(log_σ::Float64, log_noise::Float64, h::Float64=1e-5)
        # Centered difference for better accuracy
        f_σ_plus = log_posterior(log_σ + h, log_noise)
        f_σ_minus = log_posterior(log_σ - h, log_noise)
        f_n_plus = log_posterior(log_σ, log_noise + h)
        f_n_minus = log_posterior(log_σ, log_noise - h)
        
        grad_σ = (f_σ_plus - f_σ_minus) / (2 * h)
        grad_noise = (f_n_plus - f_n_minus) / (2 * h)
        
        # Clip large gradients
        grad_σ = clamp(grad_σ, -50.0, 50.0)
        grad_noise = clamp(grad_noise, -50.0, 50.0)
        
        return [grad_σ, grad_noise]
    end
    
    function leapfrog(log_σ, log_noise, p_σ, p_noise, step_size, n_steps)
        for _ in 1:n_steps
            grad = grad_log_posterior(log_σ, log_noise)
            p_σ += 0.5 * step_size * grad[1]
            p_noise += 0.5 * step_size * grad[2]
            
            log_σ += step_size * p_σ
            log_noise += step_size * p_noise
            
            # Enforce bounds
            log_σ = clamp(log_σ, log(0.05), log(0.8))
            log_noise = clamp(log_noise, log(0.001), log(0.1))
            
            grad = grad_log_posterior(log_σ, log_noise)
            p_σ += 0.5 * step_size * grad[1]
            p_noise += 0.5 * step_size * grad[2]
        end
        
        return log_σ, log_noise, p_σ, p_noise
    end
    
    # Smart initialization: find highest posterior on grid
    best_lp = -1e10
    best_log_σ = log(0.20)
    for σ_test in range(0.10, 0.35, length=15)
        lp_test = log_posterior(log(σ_test), log(true_obs_noise))
        if lp_test > best_lp
            best_lp = lp_test
            best_log_σ = log(σ_test)
        end
    end
    
    log_σ = best_log_σ
    log_noise = log(true_obs_noise)
    
    samples = zeros(n_samples, 2)
    n_accepted = 0
    
    step_size = 0.005  # Conservative initial step size
    n_steps = 10       # Moderate integration steps
    
    for iter in 1:n_samples
        p_σ = randn()
        p_noise = randn()
        
        log_σ_old = log_σ
        log_noise_old = log_noise
        p_σ_old = p_σ
        p_noise_old = p_noise
        
        log_σ, log_noise, p_σ, p_noise = leapfrog(
            log_σ, log_noise, p_σ, p_noise, step_size, n_steps
        )
        
        H_old = -log_posterior(log_σ_old, log_noise_old) + 0.5 * (p_σ_old^2 + p_noise_old^2)
        H_new = -log_posterior(log_σ, log_noise) + 0.5 * (p_σ^2 + p_noise^2)
        
        if log(rand()) < -(H_new - H_old)
            n_accepted += 1
        else
            log_σ = log_σ_old
            log_noise = log_noise_old
        end
        
        samples[iter, :] = [log_σ, log_noise]
        
        # Gentle step size adaptation
        if iter % 50 == 0 && iter > 200
            rate = n_accepted / iter
            step_size *= (1.0 + 0.0005 * (rate - 0.65))
            step_size = clamp(step_size, 0.001, 0.02)
        end
    end
    
    return samples, n_accepted / n_samples
end

println("\nResults: HMC for increasing sample counts")
println(@sprintf "%-10s %10s %10s %10s %10s %10s" "N_Samples" "σ_hat" "σ_std" "ESS" "ESS/Iter" "Accept%")
println("-"^90)

hmc_results = Dict()

for n_samp in [500, 1000, 2000, 5000]
    samples, accept_rate = run_hmc_inference(n_samp)
    
    burn_in = div(n_samp, 4)
    σ_chain = exp.(samples[burn_in+1:end, 1])
    
    if length(σ_chain) > 10
        σ_mean = mean(σ_chain)
        σ_std = std(σ_chain)
        ess_σ = compute_ess(σ_chain)
        ess_per_iter = ess_σ / n_samp
        
        hmc_results[n_samp] = (mean=σ_mean, std=σ_std, ess=ess_σ, accept=accept_rate)
        
        @printf "%-10d %10.5f %10.5f %10.1f %10.4f %10.1f%%\n" n_samp σ_mean σ_std ess_σ ess_per_iter (accept_rate*100)
    end
end

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

println("\n" * "="^90)
println("EFFICIENCY COMPARISON SUMMARY")
println("="^90)
println("\nMetric: ESS per iteration (samples generating independent information)")
println("Higher values = more efficient sampling\n")

for n_samp in [1000, 2000, 5000]
    println("Sample Size: $n_samp")
    println(@sprintf "  %-8s %10s %10s %10s" "Method" "σ_Estimate" "ESS/Iter" "Efficiency")
    println("  " * "-"^45)
    
    if haskey(mh_results, n_samp)
        r = mh_results[n_samp]
        eff = (r.ess / n_samp) * r.accept
        @printf "  %-8s %10.5f %10.4f %10.4f\n" "MH" r.mean (r.ess/n_samp) eff
    end
    
    if haskey(is_results, n_samp)
        r = is_results[n_samp]
        @printf "  %-8s %10.5f %10.4f %10.4f\n" "IS" r.mean (r.ess/n_samp) (r.ess/n_samp)
    end
    
    if haskey(hmc_results, n_samp)
        r = hmc_results[n_samp]
        eff = (r.ess / n_samp) * r.accept
        @printf "  %-8s %10.5f %10.4f %10.4f\n" "HMC" r.mean (r.ess/n_samp) eff
    end
    println()
end

# ============================================================================
# BIAS AND CONVERGENCE ANALYSIS
# ============================================================================

println("="^90)
println("PARAMETER ESTIMATION ACCURACY")
println("="^90)
println("\nBias = Estimated σ - True σ ($(true_σ))")
println("Lower bias indicates better estimation\n")

for n_samp in [1000, 5000]
    println("Sample Size: $n_samp")
    println(@sprintf "  %-8s %10s %10s %10s" "Method" "Bias" "Std Error" "RMSE")
    println("  " * "-"^50)
    
    if haskey(mh_results, n_samp)
        r = mh_results[n_samp]
        bias = r.mean - true_σ
        rmse = sqrt(bias^2 + r.std^2)
        @printf "  %-8s %10.6f %10.6f %10.6f\n" "MH" bias r.std rmse
    end
    
    if haskey(is_results, n_samp)
        r = is_results[n_samp]
        bias = r.mean - true_σ
        rmse = sqrt(bias^2 + r.std^2)
        @printf "  %-8s %10.6f %10.6f %10.6f\n" "IS" bias r.std rmse
    end
    
    if haskey(hmc_results, n_samp)
        r = hmc_results[n_samp]
        bias = r.mean - true_σ
        rmse = sqrt(bias^2 + r.std^2)
        @printf "  %-8s %10.6f %10.6f %10.6f\n" "HMC" bias r.std rmse
    end
    println()
end

println("\n" * "="^90)
println("end part 2")