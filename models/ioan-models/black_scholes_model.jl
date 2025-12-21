"""
    BlackScholesModel
"""

module BlackScholesModel

using Distributions
using DifferentialEquations

export simulate_bs_paths, bs_call_price, bs_put_price, bs_likelihood,
       verify_put_call_parity, generate_test_scenarios, analyze_likelihood_surface

"""
    simulate_bs_paths(S0, μ, σ, T, num_steps)

Simulate stock price paths using the Black-Scholes model.

# Arguments
- `S0`: Initial stock price
- `μ`: Drift (expected return)
- `σ`: Volatility (standard deviation of returns)
- `T`: Time to maturity (in years)
- `num_steps`: Number of discrete time steps

# Returns
Vector of stock prices over time
"""
function simulate_bs_paths(S0::Float64, μ::Float64, σ::Float64, T::Float64, num_steps::Int)
    dt = T / num_steps
    S = zeros(num_steps + 1)
    S[1] = S0
    
    for i in 1:num_steps
        Z = randn()
        S[i + 1] = S[i] * exp((μ - 0.5 * σ^2) * dt + σ * sqrt(dt) * Z)
    end
    
    return S
end

"""
    bs_call_price(S, K, T, r, σ)

Compute Black-Scholes call option price.

# Arguments
- `S`: Current stock price
- `K`: Strike price
- `T`: Time to maturity (in years)
- `r`: Risk-free rate
- `σ`: Volatility

# Returns
Call option price
"""
function bs_call_price(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    
    call = S * cdf(Normal(), d1) - K * exp(-r * T) * cdf(Normal(), d2)
    return call
end

"""
    bs_put_price(S, K, T, r, σ)

Compute Black-Scholes put option price.

# Arguments
- `S`: Current stock price
- `K`: Strike price
- `T`: Time to maturity (in years)
- `r`: Risk-free rate
- `σ`: Volatility

# Returns
Put option price
"""
function bs_put_price(S::Float64, K::Float64, T::Float64, r::Float64, σ::Float64)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    
    put = K * exp(-r * T) * cdf(Normal(), -d2) - S * cdf(Normal(), -d1)
    return put
end

"""
    bs_likelihood(observed_prices, S, K, T, r, σ, observation_noise)

Compute likelihood of observed option prices given model parameters.

# Arguments
- `observed_prices`: Vector of observed option prices
- `S`: Current stock price
- `K`: Vector of strike prices
- `T`: Vector of times to maturity
- `r`: Risk-free rate
- `σ`: Volatility (parameter to infer)
- `observation_noise`: Standard deviation of price observation error

# Returns
Log-likelihood of the observations
"""
function bs_likelihood(observed_prices::Vector{Float64}, S::Float64, 
                      K::Vector{Float64}, T::Vector{Float64}, r::Float64, 
                      σ::Float64, observation_noise::Float64)
    log_likelihood = 0.0
    
    for (i, (k, t)) in enumerate(zip(K, T))
        predicted_price = bs_call_price(S, k, t, r, σ)
        residual = observed_prices[i] - predicted_price
        log_likelihood += logpdf(Normal(0, observation_noise), residual)
    end
    
    return log_likelihood
end

"""
    verify_put_call_parity(S, K, T, r, call_price, put_price)

Verify put-call parity relationship: C - P = S - K*exp(-rT)
Returns (parity_holds, difference)
"""
function verify_put_call_parity(S::Float64, K::Float64, T::Float64, r::Float64,
                               call_price::Float64, put_price::Float64)
    theoretical = S - K * exp(-r * T)
    observed = call_price - put_price
    difference = abs(observed - theoretical)
    holds = difference < 1e-6
    return holds, difference
end

"""
    generate_test_scenarios(S, r, true_σ, obs_noise)

Generate 5 comprehensive test scenarios:
1. ATM (At-The-Money): Tests baseline
2. OTM (Out-of-The-Money): Tests tail behavior
3. ITM (In-The-Money): Tests deep ITM
4. Mixed: Diverse strikes and maturities
5. Pairs: Call-put pairs for put-call parity testing
"""
function generate_test_scenarios(S::Float64, r::Float64, true_σ::Float64, obs_noise::Float64)
    scenarios = Dict()
    
    # Scenario 1: At-The-Money options
    atm_K = [S]  # Strike = Current price
    atm_T = [0.25, 0.5, 1.0]
    atm_prices = Float64[]
    for (k, t) in zip(atm_K, atm_T)
        price = bs_call_price(S, k, t, r, true_σ)
        push!(atm_prices, price + randn() * obs_noise)
    end
    scenarios["ATM"] = (K=atm_K, T=atm_T, prices=atm_prices, 
                        description="At-the-money options")
    
    # Scenario 2: Out-of-The-Money (OTM) options
    otm_K = [S * 1.05, S * 1.10, S * 1.15]  # 5%, 10%, 15% OTM
    otm_T = [0.25, 0.5, 0.75]
    otm_prices = Float64[]
    for (k, t) in zip(otm_K, otm_T)
        price = bs_call_price(S, k, t, r, true_σ)
        push!(otm_prices, price + randn() * obs_noise)
    end
    scenarios["OTM"] = (K=otm_K, T=otm_T, prices=otm_prices,
                        description="Out-of-the-money options")
    
    # Scenario 3: In-The-Money (ITM) options
    itm_K = [S * 0.95, S * 0.90, S * 0.85]  # 5%, 10%, 15% ITM
    itm_T = [0.25, 0.5, 0.75]
    itm_prices = Float64[]
    for (k, t) in zip(itm_K, itm_T)
        price = bs_call_price(S, k, t, r, true_σ)
        push!(itm_prices, price + randn() * obs_noise)
    end
    scenarios["ITM"] = (K=itm_K, T=itm_T, prices=itm_prices,
                        description="In-the-money options")
    
    # Scenario 4: Mixed - Diverse strikes and maturities (good for inference)
    mixed_K = [0.90*S, 0.95*S, S, 1.05*S, 1.10*S]
    mixed_T = [0.1, 0.25, 0.5, 1.0, 2.0]
    mixed_prices = Float64[]
    for (k, t) in zip(mixed_K, mixed_T)
        price = bs_call_price(S, k, t, r, true_σ)
        push!(mixed_prices, price + randn() * obs_noise)
    end
    scenarios["MIXED"] = (K=mixed_K, T=mixed_T, prices=mixed_prices,
                          description="Mixed strikes and maturities")
    
    # Scenario 5: Call and Put pairs (tests put-call parity)
    pair_K = [0.95*S, S, 1.05*S]
    pair_T = [0.5, 1.0]
    call_prices = Float64[]
    put_prices = Float64[]
    for k in pair_K
        for t in pair_T
            call = bs_call_price(S, k, t, r, true_σ)
            put = bs_put_price(S, k, t, r, true_σ)
            push!(call_prices, call + randn() * obs_noise)
            push!(put_prices, put + randn() * obs_noise)
        end
    end
    scenarios["PAIRS"] = (K=pair_K, T=pair_T, calls=call_prices, puts=put_prices,
                          description="Call-put pairs (put-call parity)")
    
    return scenarios
end

"""
    analyze_likelihood_surface(observed_prices, S, K, T, r, obs_noise)

Detailed analysis of likelihood surface for inference diagnostics.
Returns fine-grained likelihood surface and confidence intervals.
"""
function analyze_likelihood_surface(observed_prices::Vector{Float64}, S::Float64,
                                   K::Vector{Float64}, T::Vector{Float64},
                                   r::Float64, obs_noise::Float64)
    
    # Test over fine grid
    σ_values = range(0.05, 1.0, length=100)
    likelihoods = [bs_likelihood(observed_prices, S, K, T, r, σ, obs_noise) 
                   for σ in σ_values]
    
    # Find maximum likelihood estimator
    max_lik, max_idx = findmax(likelihoods)
    mle_σ = σ_values[max_idx]
    
    # Find 95% confidence interval (within 1.92 log-likelihood units)
    threshold = max_lik - 1.92
    ci_mask = likelihoods .>= threshold
    if any(ci_mask)
        ci_lower = σ_values[findfirst(ci_mask)]
        ci_upper = σ_values[findlast(ci_mask)]
    else
        ci_lower = ci_upper = mle_σ
    end
    
    return (σ_values=σ_values, likelihoods=likelihoods, 
            mle_σ=mle_σ, ci_lower=ci_lower, ci_upper=ci_upper)
end

end  # module BlackScholesModel
