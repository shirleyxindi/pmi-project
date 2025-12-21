"""
    MertonJumpDiffusionModel
"""

module MertonJumpDiffusionModel

using Distributions
using Random

export simulate_merton_paths, merton_likelihood, extract_jump_times,
       detect_jumps_bayesian, analyze_jump_statistics, generate_challenging_scenarios,
       compare_scenarios

"""
    simulate_merton_paths(S0, μ, σ, λ, μ_J, σ_J, T, num_steps)

Simulate stock price paths using the Merton Jump Diffusion model.

The dynamics follow:
dS_t = μ S_t dt + σ S_t dW_t + S_t dJ_t

where J_t is a jump process with Poisson arrivals at rate λ and 
log-normal jump sizes.

# Arguments
- `S0`: Initial stock price
- `μ`: Drift (expected return)
- `σ`: Volatility (diffusion component)
- `λ`: Jump intensity (Poisson rate)
- `μ_J`: Mean of log jump size
- `σ_J`: Standard deviation of log jump size
- `T`: Time to maturity (in years)
- `num_steps`: Number of discrete time steps

# Returns
Tuple (stock_prices, jump_times, jump_sizes) where:
- stock_prices: Vector of stock prices over time
- jump_times: Times at which jumps occurred
- jump_sizes: Sizes of jumps (multiplicative factors)
"""
function simulate_merton_paths(S0::Float64, μ::Float64, σ::Float64, 
                               λ::Float64, μ_J::Float64, σ_J::Float64,
                               T::Float64, num_steps::Int)
    dt = T / num_steps
    S = zeros(num_steps + 1)
    S[1] = S0
    
    jump_times = Float64[]
    jump_sizes = Float64[]
    
    t = 0.0
    for i in 1:num_steps
        t = i * dt
        
        # Continuous diffusion component
        Z_W = randn()
        S[i + 1] = S[i] * exp((μ - 0.5 * σ^2) * dt + σ * sqrt(dt) * Z_W)
        
        # Jump component (Poisson process)
        num_jumps = rand(Poisson(λ * dt))
        
        for _ in 1:num_jumps
            jump_log_size = μ_J + σ_J * randn()
            jump_size = exp(jump_log_size)
            S[i + 1] *= jump_size
            
            push!(jump_times, t)
            push!(jump_sizes, jump_size)
        end
    end
    
    return S, jump_times, jump_sizes
end

"""
    merton_likelihood(observed_prices, S0, T, r, μ, σ, λ, μ_J, σ_J, 
                     observation_noise, num_steps)

Compute likelihood of observed prices under Merton Jump Diffusion model.

This likelihood computation integrates over all possible jump configurations
that could have produced the observed price sequence.

# Arguments
- `observed_prices`: Vector of observed prices at discrete time points
- `S0`: Initial price
- `T`: Total time period
- `r`: Risk-free rate
- `μ`: Drift parameter
- `σ`: Diffusion volatility
- `λ`: Jump intensity
- `μ_J`: Mean log jump size
- `σ_J`: Std dev of log jump size
- `observation_noise`: Observation error std dev
- `num_steps`: Number of time steps in observation

# Returns
Log-likelihood of observations
"""
function merton_likelihood(observed_prices::Vector{Float64}, S0::Float64,
                          T::Float64, r::Float64, μ::Float64, σ::Float64,
                          λ::Float64, μ_J::Float64, σ_J::Float64,
                          observation_noise::Float64, num_steps::Int)
    
    # Simplified likelihood using approximation
    # In practice, we would need importance sampling or particle filtering
    # for exact inference due to the discrete jump events
    
    dt = T / num_steps
    log_likelihood = 0.0
    
    for i in 1:length(observed_prices)-1
        price_return = log(observed_prices[i + 1] / observed_prices[i])
        
        # Expected return under Merton model (approximately)
        expected_return = (μ - 0.5 * σ^2) * dt + λ * μ_J * dt
        
        # Variance including jump risk
        variance = (σ^2 + λ * (μ_J^2 + σ_J^2)) * dt
        
        # Observation likelihood
        log_likelihood += logpdf(Normal(expected_return, sqrt(variance + observation_noise^2)), 
                                price_return)
    end
    
    return log_likelihood
end

"""
    extract_jump_times(S_path, price_threshold)

Detect potential jumps in a price path based on large movements.

This is a heuristic method for jump detection. A more principled approach
would use particle filtering.

# Arguments
- `S_path`: Vector of prices
- `price_threshold`: Fraction threshold for detecting a jump (e.g., 0.05 = 5% move)

# Returns
Indices where jumps are detected
"""
function extract_jump_times(S_path::Vector{Float64}, price_threshold::Float64)
    jump_indices = Int[]
    
    for i in 2:length(S_path)
        return_magnitude = abs(log(S_path[i] / S_path[i-1]))
        if return_magnitude > price_threshold
            push!(jump_indices, i)
        end
    end
    
    return jump_indices
end

"""
    detect_jumps_bayesian(S_path, dt, σ_baseline, λ_baseline)

Bayesian jump detection using adaptive threshold.
Returns list of likely jump indices based on posterior probability.
"""
function detect_jumps_bayesian(S_path::Vector{Float64}, dt::Float64, 
                              σ_baseline::Float64, λ_baseline::Float64)
    jump_indices = Int[]
    
    # Expected variance under no-jump scenario
    diffusion_std = σ_baseline * sqrt(dt)
    
    # Jump probability threshold
    jump_prob_threshold = 0.3  # At least 30% chance of jump
    
    for i in 2:length(S_path)
        log_return = log(S_path[i] / S_path[i-1])
        
        # Likelihood under jump vs no-jump
        # P(Jump | return) ∝ P(return | Jump) * P(Jump)
        p_no_jump = pdf(Normal(0, diffusion_std), log_return)
        p_jump = pdf(Normal(0, diffusion_std * 3), log_return)  # Wider dist with jump
        
        # Posterior probability of jump
        p_jump_posterior = (p_jump * λ_baseline * dt) / 
                          (p_jump * λ_baseline * dt + p_no_jump * (1 - λ_baseline * dt))
        
        if p_jump_posterior > jump_prob_threshold
            push!(jump_indices, i)
        end
    end
    
    return jump_indices
end

"""
    analyze_jump_statistics(jump_sizes)

Analyze empirical jump size distribution.
Returns summary statistics of jump sizes.
"""
function analyze_jump_statistics(jump_sizes::Vector{Float64})
    if length(jump_sizes) == 0
        return nothing
    end
    
    log_jumps = log.(jump_sizes)
    
    return (
        count = length(jump_sizes),
        mean = mean(jump_sizes),
        std = std(jump_sizes),
        log_mean = mean(log_jumps),
        log_std = std(log_jumps),
        min = minimum(jump_sizes),
        max = maximum(jump_sizes)
    )
end

"""
    generate_challenging_scenarios(S0, T, r, num_obs)

Generate 4 challenging scenarios for inference stress-testing:
1. MANY_JUMPS: High λ (many small jumps) - easy to infer λ
2. FEW_LARGE: Low λ (few large jumps) - HARDEST case, rare events
3. CRASH_RISK: Asymmetric negative jumps - realistic crash-a-phobia
4. NOISE_DOMINATED: High observation noise - jumps hidden in noise
"""
function generate_challenging_scenarios(S0::Float64, T::Float64, r::Float64, num_obs::Int=250)
    scenarios = Dict()
    
    # Scenario 1: Many small jumps (λ is easy to infer)
    μ1, σ1 = 0.08, 0.10
    λ1, μ_J1, σ_J1 = 3.0, 0.01, 0.02  # 3 jumps/year, tiny jumps
    obs_noise1 = 0.2
    S1, jt1, js1 = simulate_merton_paths(S0, μ1, σ1, λ1, μ_J1, σ_J1, T, num_obs)
    obs1 = S1[1:num_obs] .+ randn(num_obs) .* obs_noise1
    scenarios["MANY_JUMPS"] = (
        prices=obs1, 
        params=(μ=μ1, σ=σ1, λ=λ1, μ_J=μ_J1, σ_J=σ_J1, obs_noise=obs_noise1),
        jump_info=(times=jt1, sizes=js1),
        description="High jump intensity (λ=3), tiny jumps → Easy to detect λ, hard to infer sizes"
    )
    
    # Scenario 2: Few large jumps (λ is hard to infer - rare events!)
    μ2, σ2 = 0.10, 0.12
    λ2, μ_J2, σ_J2 = 0.3, -0.05, 0.08  # ~0.3 jumps/year, large downward jumps
    obs_noise2 = 0.3
    S2, jt2, js2 = simulate_merton_paths(S0, μ2, σ2, λ2, μ_J2, σ_J2, T, num_obs)
    obs2 = S2[1:num_obs] .+ randn(num_obs) .* obs_noise2
    scenarios["FEW_LARGE"] = (
        prices=obs2,
        params=(μ=μ2, σ=σ2, λ=λ2, μ_J=μ_J2, σ_J=σ_J2, obs_noise=obs_noise2),
        jump_info=(times=jt2, sizes=js2),
        description="Low jump intensity (λ=0.3), large jumps → HARDEST: rare events, inference challenge"
    )
    
    # Scenario 3: Crash risk (asymmetric jumps - realistic)
    μ3, σ3 = 0.12, 0.15
    λ3, μ_J3, σ_J3 = 0.8, -0.15, 0.10  # Downside jumps (crash-a-phobia)
    obs_noise3 = 0.25
    S3, jt3, js3 = simulate_merton_paths(S0, μ3, σ3, λ3, μ_J3, σ_J3, T, num_obs)
    obs3 = S3[1:num_obs] .+ randn(num_obs) .* obs_noise3
    scenarios["CRASH_RISK"] = (
        prices=obs3,
        params=(μ=μ3, σ=σ3, λ=λ3, μ_J=μ_J3, σ_J=σ_J3, obs_noise=obs_noise3),
        jump_info=(times=jt3, sizes=js3),
        description="Asymmetric downside jumps → Realistic, tests skewness inference"
    )
    
    # Scenario 4: Noise-dominated (subtle jumps hidden in observation error)
    μ4, σ4 = 0.09, 0.11
    λ4, μ_J4, σ_J4 = 0.5, 0.02, 0.04  # Upside jumps, harder to see with noise
    obs_noise4 = 1.0  # HIGH NOISE - jumps barely visible!
    S4, jt4, js4 = simulate_merton_paths(S0, μ4, σ4, λ4, μ_J4, σ_J4, T, num_obs)
    obs4 = S4[1:num_obs] .+ randn(num_obs) .* obs_noise4
    scenarios["NOISE_DOMINATED"] = (
        prices=obs4,
        params=(μ=μ4, σ=σ4, λ=λ4, μ_J=μ_J4, σ_J=σ_J4, obs_noise=obs_noise4),
        jump_info=(times=jt4, sizes=js4),
        description="High observation noise (σ_obs=1.0) → Jumps hidden in noise, tests robustness"
    )
    
    return scenarios
end

"""
    compare_scenarios(scenarios, S0, T)

Print detailed comparison of all challenging scenarios for analysis.
"""
function compare_scenarios(scenarios::Dict, S0::Float64=100.0, T::Float64=1.0)
    println("\n" * "="^80)
    println("MERTON MODEL: CHALLENGING INFERENCE SCENARIOS")
    println("="^80)
    
    for (scenario_name, data) in scenarios
        println("\n$(scenario_name):")
        println("  Description: $(data.description)")
        
        params = data.params
        println("  Parameters:")
        println("    μ=$(params.μ), σ=$(params.σ), λ=$(params.λ)")
        println("    μ_J=$(params.μ_J), σ_J=$(params.σ_J)")
        println("    Observation noise: $(params.obs_noise)")
        
        jumps = data.jump_info
        println("  Actual jumps in data: $(length(jumps.times))")
        if length(jumps.times) > 0
            jump_stats = analyze_jump_statistics(jumps.sizes)
            println("    Mean jump size: $(round(jump_stats.mean, digits=3))")
            println("    Std jump size: $(round(jump_stats.std, digits=3))")
        end
        
        # Likelihood at true parameters
        lik = merton_likelihood(data.prices, S0, T, 0.05, params.μ, params.σ,
                               params.λ, params.μ_J, params.σ_J, 
                               params.obs_noise, length(data.prices))
        println("  Log-likelihood at true params: $(round(lik, digits=2))")
    end
end

end  # module MertonJumpDiffusionModel  