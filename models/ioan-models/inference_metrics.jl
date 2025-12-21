"""
    inference_metrics.jl
"""

module InferenceMetrics

using Statistics
using Random
using Printf

export compute_ess, compute_autocorr_time, gelman_rubin_diagnostic, 
       posterior_interval, inference_summary, compute_hpd_interval

"""
    compute_ess(samples, log_weights=nothing)

Compute Effective Sample Size (ESS).
- For weighted samples: ESS = (Σ weights)² / Σ weights²
- For unweighted: uses autocorrelation time
"""
function compute_ess(samples::Vector{Float64}, log_weights::Union{Vector{Float64}, Nothing}=nothing)
    n = length(samples)
    
    if log_weights !== nothing && !isempty(log_weights)
        # Weighted ESS (importance sampling)
        max_lw = maximum(log_weights)
        weights = exp.(log_weights .- max_lw)
        weights = weights / sum(weights)
        ess = 1.0 / sum(weights.^2)
    else
        # Unweighted ESS using autocorrelation
        tau = compute_autocorr_time(samples)
        ess = n / tau
    end
    
    return ess
end

"""
    compute_autocorr_time(samples)

Compute integrated autocorrelation time using FFT.
"""
function compute_autocorr_time(samples::Vector{Float64})
    n = length(samples)
    
    # Demean
    x = samples .- mean(samples)
    
    # Compute autocorrelation
    c0 = sum(x.^2) / n
    
    if c0 < 1e-10
        return 1.0
    end
    
    # Compute autocorrelations up to max lag
    max_lag = min(n ÷ 2, 500)
    acf = zeros(max_lag)
    
    for lag in 1:max_lag
        acf[lag] = sum(x[1:n-lag] .* x[lag+1:n]) / n / c0
    end
    
    # Integrated autocorrelation time (with auto-windowing)
    tau_int = 0.5
    for lag in 1:max_lag
        if acf[lag] < 0.05 * exp(-lag / 30)  # Automatic windowing
            break
        end
        tau_int += acf[lag]
    end
    
    return max(1.0, tau_int)
end

"""
    gelman_rubin_diagnostic(chains::Vector{Matrix{Float64}})

Compute Gelman-Rubin R̂ convergence diagnostic.
Requires multiple independent chains.
R̂ < 1.1 indicates convergence.
"""
function gelman_rubin_diagnostic(chains::Vector{Matrix{Float64}})
    m = length(chains)  # number of chains
    n = size(chains[1], 1)  # iterations per chain
    
    if m < 2
        return fill(NaN, size(chains[1], 2))
    end
    
    n_params = size(chains[1], 2)
    r_hat = zeros(n_params)
    
    for p in 1:n_params
        # Extract parameter samples from all chains
        param_chains = [chains[i][:, p] for i in 1:m]
        
        # Within-chain variance
        W = mean([var(param_chains[i]) for i in 1:m])
        
        # Between-chain variance
        chain_means = [mean(param_chains[i]) for i in 1:m]
        global_mean = mean(chain_means)
        B = n / (m - 1) * sum((chain_means .- global_mean).^2)
        
        # Estimated variance
        var_est = (1 - 1/n) * W + (1/n) * B
        
        # Gelman-Rubin R̂
        r_hat[p] = sqrt(var_est / W)
    end
    
    return r_hat
end

"""
    compute_hpd_interval(samples, credible_level=0.95)

Compute Highest Posterior Density (HPD) credible interval.
"""
function compute_hpd_interval(samples::Vector{Float64}, credible_level::Float64=0.95)
    n = length(samples)
    sorted = sort(samples)
    
    # Find interval width
    n_in_interval = round(Int, n * credible_level)
    interval_width = sorted[n_in_interval] - sorted[1]
    
    # Find densest interval
    min_idx = 1
    min_width = interval_width
    
    for i in 1:(n - n_in_interval)
        width = sorted[i + n_in_interval] - sorted[i]
        if width < min_width
            min_width = width
            min_idx = i
        end
    end
    
    return (sorted[min_idx], sorted[min_idx + n_in_interval])
end

"""
    posterior_interval(samples, credible_level=0.95, interval_type="hpd")

Compute posterior credible interval.
"""
function posterior_interval(samples::Vector{Float64}, credible_level::Float64=0.95, 
                           interval_type::String="hpd")
    
    if interval_type == "hpd"
        return compute_hpd_interval(samples, credible_level)
    else  # equal-tailed
        α = (1 - credible_level) / 2
        return (quantile(samples, α), quantile(samples, 1 - α))
    end
end

"""
    inference_summary(results::Dict)

Print comprehensive summary of inference results.
"""
function inference_summary(results::Dict)
    println("\n" * "="^80)
    println("INFERENCE RESULTS SUMMARY")
    println("="^80)
    
    for (method, result) in results
        println("\n" * "-"^80)
        println("Method: $(result.method)")
        println("-"^80)
        
        # Sample statistics
        println("\nPosterior Statistics:")
        for i in 1:length(result.posterior_mean)
            mean_val = result.posterior_mean[i]
            std_val = result.posterior_std[i]
            println("  Parameter $i: Mean = $(round(mean_val, digits=6)), Std = $(round(std_val, digits=6))")
        end
        
        # Effective sample size
        if !isempty(result.log_weights)
            ess_total = sum(exp.(result.log_weights .- logsumexp(result.log_weights)).^2)
            ess_effective = 1.0 / ess_total
        else
            ess_effective = NaN
        end
        println("\nEffective Sample Size: $(round(ess_effective, digits=2)) / $(result.n_iterations)")
        
        # Acceptance rate
        if result.acceptance_rate > 0
            println("Acceptance Rate: $(round(result.acceptance_rate * 100, digits=2))%")
        end
        
        # Jump count statistics (for Merton model)
        if !isempty(result.jump_count_estimates)
            println("Estimated Jump Counts: Mean = $(round(mean(result.jump_count_estimates), digits=2)), " *
                   "Std = $(round(std(result.jump_count_estimates), digits=2))")
        end
        
        println()
    end
    
    println("="^80)
end

"""Compute log-sum-exp for numerical stability"""
function logsumexp(x::Vector{Float64})
    if isempty(x)
        return 0.0
    end
    max_x = maximum(x)
    return max_x + log(sum(exp.(x .- max_x)))
end

"""
    compare_inference_methods(results::Dict, true_params::Union{Nothing, Vector{Float64}}=nothing)

Compare inference methods with optional ground truth.
"""
function compare_inference_methods(results::Dict, true_params::Union{Nothing, Vector{Float64}}=nothing)
    println("\n" * "="^80)
    println("INFERENCE METHOD COMPARISON")
    println("="^80)
    
    methods = collect(keys(results))
    n_methods = length(methods)
    
    # Table header
    header = @sprintf "%-20s %15s %15s %15s" "Method" "ESS/iter" "Accept. Rate" "Parameters"
    if true_params !== nothing
        header *= @sprintf "%20s" "MAE vs Truth"
    end
    println(header)
    println("-"^80)
    
    for method in methods
        result = results[method]
        
        if !isempty(result.log_weights)
            ess_per_iter = 1.0 / sum(exp.(result.log_weights .- logsumexp(result.log_weights)).^2) / result.n_iterations
        else
            ess_per_iter = NaN
        end
        
        accept_str = result.acceptance_rate > 0 ? 
                     @sprintf("%.2f%%", result.acceptance_rate * 100) : "N/A"
        
        param_str = @sprintf "%.6f" result.posterior_mean[1]
        
        mae_str = ""
        if true_params !== nothing && length(true_params) > 0
            mae = mean(abs.(result.posterior_mean[1:min(length(true_params), length(result.posterior_mean))] .- 
                           true_params[1:min(length(true_params), length(result.posterior_mean))]))
            mae_str = @sprintf "%15.6f" mae
        end
        
        row = @sprintf "%-20s %15.4f %15s %15s" method ess_per_iter accept_str param_str
        if true_params !== nothing
            row *= mae_str
        end
        println(row)
    end
    
    println("="^80)
end

end # module InferenceMetrics
