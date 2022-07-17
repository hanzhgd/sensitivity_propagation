# utils.jl
using Distributions
using Statistics
using Roots
using Plots
using DataFrames
using LaTeXStrings
using Optim
using LinearAlgebra
using ForwardDiff

ϕ(x, y; H=1) = max.(y .- x, 0) .- H .* max.(x .- y, 0)
ϕ′(x, y; H=1) = max.(x .- y, 0) .- H .* max.(y .- x, 0)
ψ(x, y; K=1) = max.(y .- x, 0) .- K .* max.(x .- y, 0)
ψ′(x, y; K=1) = max.(x .- y, 0) .- K .* max.(y .- x, 0)

function data_generation(N=10000; H=exp(0.1), K=exp(0.5))
    N = 10000 # number of total observations

    # unobserved confounder for external validity
    U = rand(Normal(), N)

    # DGP for S₁ and G. They all depend on U.
    S₁ = 5 * (rand(N) .< (0.5 .- 0.3 * (U .> 0))) .+ 5
    G = rand(N) .> (1 ./ (1 .+ exp.(-log(H) .* (U .> 0))))

    # unobserved confounder for latent unconfoundedness
    V = rand(Normal(), N)

    # DGP for Y₁ and D. They all depend on V.
    Y₁ = S₁ + V + randn(N)
    D = similar(G)

    # randomization in the experimental sample
    D[G.==1] = rand((0, 1), sum(G .== 1))

    # K-latent-confoundedness
    D[G.==0] = rand(sum(G .== 0)) .>
               (1 ./ (1 .+ exp.(-log(K) .* (V[G.==0] .> 0) .- 0.1 .* S₁[G.==0])))

    df = DataFrame(G=G, S1=S₁, D=D, Y1=Y₁, S=D .* S₁, Y=D .* Y₁)
end


function lower(m; K=exp(0.5), H=exp(0.1), df=df)
    ϕ(x, y) = max.(y .- x, 0) .- H .* max.(x .- y, 0)
    ϕ′(x, y) = max.(x .- y, 0) .- H .* max.(y .- x, 0)
    ψ(x, y) = max.(y .- x, 0) .- K .* max.(x .- y, 0)
    ψ′(x, y) = max.(x .- y, 0) .- K .* max.(y .- x, 0)

    f(s) = m -> mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ s) .* ψ(m, df.Y))
    g(s) = m -> mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ s) .* ψ′(m, df.Y))

    θ1minushat_s1 = find_zero(f(5), 2.0)
    θ1minushat_s2 = find_zero(f(10), 2.0)

    return mean((df.D .== 1) .* (df.G .== 1) .*
                (df.S .== 5) .* ϕ(m, θ1minushat_s1)) + mean((df.D .== 1) .* (df.G .== 1) .*
                                                            (df.S .== 10) .* ϕ(m, θ1minushat_s2))
end


function upper(m; K=exp(0.5), H=exp(0.1), df=df)
    ϕ(x, y) = max.(y .- x, 0) .- H .* max.(x .- y, 0)
    ϕ′(x, y) = max.(x .- y, 0) .- H .* max.(y .- x, 0)
    ψ(x, y) = max.(y .- x, 0) .- K .* max.(x .- y, 0)
    ψ′(x, y) = max.(x .- y, 0) .- K .* max.(y .- x, 0)

    f(s) = m -> mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ s) .* ψ(m, df.Y))
    g(s) = m -> mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ s) .* ψ′(m, df.Y))

    θ1plushat_s1 = find_zero(g(5), 2.0)
    θ1plushat_s2 = find_zero(g(10), 2.0)

    return mean((df.D .== 1) .* (df.G .== 1) .*
                (df.S .== 5) .* ϕ′(m, θ1plushat_s1)) + mean((df.D .== 1) .* (df.G .== 1) .*
                                                            (df.S .== 10) .* ϕ′(m, θ1plushat_s2))
end

function compute_lower_Y1(df; K=1, H=1)

    f(s; df=df, K=K) = m -> mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ s) .* ψ(m, df.Y; K=K))

    θ1minushat_s1 = find_zero(f(5; df=df, K=K), 2.0)
    θ1minushat_s2 = find_zero(f(10; df=df, K=K), 2.0)

    μ1minus = find_zero(m -> lower(m; df=df, K=K, H=H), 2.0) / mean(df[df.G.==0, :D] .== 0) -
              (θ1minushat_s1 * mean(df[(df.D.==1).*(df.G.==0), :S] .== 5) +
               θ1minushat_s2 * mean(df[(df.D.==1).*(df.G.==0), :S] .== 10)) *
              mean(df[df.G.==0, :D] .== 1) /
              mean(df[df.G.==0, :D] .== 0)

    lower_Y1 = mean(df[(df.G.==0).*(df.D.==1), :Y]) *
               mean(df[df.G.==0, :D] .== 1) +
               μ1minus *
               mean(df[df.G.==0, :D] .== 0)
end

function compute_upper_Y1(df; K=exp(0.5), H=exp(0.1))

    g(s; df=df, K=K) = m -> mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ s) .* ψ′(m, df.Y; K=K))

    θ1plushat_s1 = find_zero(g(5; df=df, K=K), 2.0)
    θ1plushat_s2 = find_zero(g(10; df=df, K=K), 2.0)

    μ1plus = find_zero(m -> upper(m; df=df, K=K, H=H), 2.0) / mean(df[df.G.==0, :D] .== 0) - (θ1plushat_s1 * mean(df[(df.D.==1).*(df.G.==0), :S] .== 5) +
                                                                                              θ1plushat_s2 * mean(df[(df.D.==1).*(df.G.==0), :S] .== 10)) *
                                                                                             mean(df[df.G.==0, :D] .== 1) /
                                                                                             mean(df[df.G.==0, :D] .== 0)

    upper_Y1 = mean(df[(df.G.==0).*(df.D.==1), :Y]) *
               mean(df[df.G.==0, :D] .== 1) +
               μ1plus *
               mean(df[df.G.==0, :D] .== 0)
end

function lower2(m; K=exp(0.5), H=exp(0.1), df=df)
    ϕ(x, y) = max.(y .- x, 0) .- H .* max.(x .- y, 0)
    ϕ′(x, y) = max.(x .- y, 0) .- H .* max.(y .- x, 0)
    ψ(x, y) = max.(y .- x, 0) .- K .* max.(x .- y, 0)
    ψ′(x, y) = max.(x .- y, 0) .- K .* max.(y .- x, 0)

    f(s) = m -> mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ s) .* ψ(m, df.Y))
    g(s) = m -> mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ s) .* ψ′(m, df.Y))

    θ0minushat_s1 = find_zero(f(5), 0.0)
    θ0minushat_s2 = find_zero(f(10), 0.0)

    return mean((df.D .== 0) .* (df.G .== 1) .*
                (df.S .== 5) .* ϕ(m, θ0minushat_s1)) + mean((df.D .== 0) .* (df.G .== 1) .*
                                                            (df.S .== 10) .* ϕ(m, θ0minushat_s2))
end

function upper2(m; K=exp(0.5), H=exp(0.1), df=df)
    ϕ(x, y) = max.(y .- x, 0) .- H .* max.(x .- y, 0)
    ϕ′(x, y) = max.(x .- y, 0) .- H .* max.(y .- x, 0)
    ψ(x, y) = max.(y .- x, 0) .- K .* max.(x .- y, 0)
    ψ′(x, y) = max.(x .- y, 0) .- K .* max.(y .- x, 0)

    f(s) = m -> mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ s) .* ψ(m, df.Y))
    g(s) = m -> mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ s) .* ψ′(m, df.Y))

    θ0plushat_s1 = find_zero(g(5), 0.0)
    θ0plushat_s2 = find_zero(g(10), 0.0)

    return mean((df.D .== 0) .* (df.G .== 1) .*
                (df.S .== 5) .* ϕ′(m, θ0plushat_s1)) + mean((df.D .== 0) .* (df.G .== 1) .*
                                                            (df.S .== 10) .* ϕ′(m, θ0plushat_s2))
end


function compute_lower_Y0(df; K=1, H=1)

    f(s; df=df, K=K) = m -> mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ s) .* ψ(m, df.Y; K=K))

    θ0minushat_s1 = find_zero(f(5; df=df, K=K), 0.0)
    θ0minushat_s2 = find_zero(f(10; df=df, K=K), 0.0)

    μ0minus = find_zero(m -> lower2(m; df=df, K=K, H=H), 0.0) / mean(df[df.G.==0, :D] .== 1) -
              (θ0minushat_s1 * mean(df[(df.D.==0).*(df.G.==0), :S] .== 5) +
               θ0minushat_s2 * mean(df[(df.D.==0).*(df.G.==0), :S] .== 10)) *
              mean(df[df.G.==0, :D] .== 0) /
              mean(df[df.G.==0, :D] .== 1)

    lower_Y0 = mean(df[(df.G.==0).*(df.D.==0), :Y]) *
               mean(df[df.G.==0, :D] .== 0) +
               μ0minus *
               mean(df[df.G.==0, :D] .== 1)
end


function compute_upper_Y0(df; K=exp(0.5), H=exp(0.1))

    g(s; df=df, K=K) = m -> mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ s) .* ψ′(m, df.Y; K=K))

    θ0plushat_s1 = find_zero(g(5; df=df, K=K), 0.0)
    θ0plushat_s2 = find_zero(g(10; df=df, K=K), 0.0)

    μ0plus = find_zero(m -> upper2(m; df=df, K=K, H=H), 0.0) / mean(df[df.G.==0, :D] .== 1) -
             (θ0plushat_s1 * mean(df[(df.D.==0).*(df.G.==0), :S] .== 5) +
              θ0plushat_s2 * mean(df[(df.D.==0).*(df.G.==0), :S] .== 10)) *
             mean(df[df.G.==0, :D] .== 0) /
             mean(df[df.G.==0, :D] .== 1)

    upper_Y0 = mean(df[(df.G.==0).*(df.D.==0), :Y]) *
               mean(df[df.G.==0, :D] .== 0) +
               μ0plus *
               mean(df[df.G.==0, :D] .== 1)
end

function data_generation2(N=10000; H=exp(0.1), K=exp(0.5))
    τ = 1

    # unobserved confounder for external validity
    U = rand(Normal(), N)

    # DGP for S₁ and G. They all depend on U.
    S₀ = 5 * (rand(N) .< (0.5 .+ 0.15 * (U .> 0))) .+ 5
    S₁ = 5 * (rand(N) .< (0.5 .+ 0.3 * (U .> 0))) .+ 5
    G = rand(N) .< (1 ./ (1 .+ exp.(-log(H) .* (U .> 0))))

    # unobserved confounder for latent unconfoundedness
    V = rand(Normal(), N)

    # DGP for Y₁ and D. They all depend on V.
    Y₀ = S₀ + 5 * V + randn(N)
    Y₁ = Y₀ .+ τ
    D = similar(G)

    # randomization in the experimental sample
    D[G.==1] = rand((0, 1), sum(G .== 1))

    # K-latent-confoundedness
    D[G.==0] = rand(sum(G .== 0)) .<
               (1 ./ (1 .+ exp.(-log(K) .* (V[G.==0] .> 0) - 0.5 .* (U[G.==0] .> 0))))

    df = DataFrame(G=G, S1=S₁, S0=S₀, D=D, Y1=Y₁, Y0=Y₀,
        S=D .* S₁ .+ (1 .- D) .* S₀, Y=D .* Y₁ .+ (1 .- D) .* Y₀)
end




############## Confidence Interval ################
function moment_τ_bds(param, data; K=1, H=1)
    df = data
    p1, θ11, θ12, ζ1, θ01, θ02, ζ0, q1, r1, m1, m0 = param

    [mean(p1 .- df[df.G.==0, :D])
        mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ 5) .* ψ(θ11, df.Y; K=K))
        mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ 10) .* ψ(θ12, df.Y; K=K))
        mean((df.D .== 1) .* (df.G .== 1) .* (df.S .≈ 5) .* ϕ(ζ1, θ11; H=H)) + mean((df.D .== 1) .* (df.G .== 1) .* (df.S .≈ 10) .* ϕ(ζ1, θ12; H=H))
        mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ 5) .* ψ′(θ01, df.Y; K=K))
        mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ 10) .* ψ′(θ02, df.Y; K=K))
        mean((df.D .== 0) .* (df.G .== 1) .* (df.S .≈ 5) .* ϕ′(ζ0, θ01; H=H)) + mean((df.D .== 0) .* (df.G .== 1) .* (df.S .≈ 10) .* ϕ′(ζ0, θ02; H=H))
        q1 - mean(df[(df.G.==0).*(df.D.==1), :S] .== 5)
        # q2 - mean(df[ (df.G.==0) .* (df.D .==1), :S] .== 10);
        r1 - mean(df[(df.G.==0).*(df.D.==0), :S] .== 5)
        # r2 - mean(df[ (df.G.==0) .* (df.D .==0), :S] .== 10);
        mean((df.D .== 1) .* (df.G .== 0) .* (df.Y .- m1))
        mean((df.D .== 0) .* (df.G .== 0) .* (df.Y .- m0))]
end

function cov_moment_τ_bds(param, data; K=1, H=1)
    df = data
    p1, θ11, θ12, ζ1, θ01, θ02, ζ0, q1, r1, m1, m0 = param

    n = size(df, 1)

    mom = hcat((df.G .== 0) .* (p1 .- df[:, :D]),
        (df.D .== 1) .* (df.G .== 0) .* (df.S .≈ 5) .* ψ(θ11, df.Y; K=K),
        (df.D .== 1) .* (df.G .== 0) .* (df.S .≈ 10) .* ψ(θ12, df.Y; K=K),
        (df.D .== 1) .* (df.G .== 1) .* (df.S .== 5) .* ϕ(ζ1, θ11; H=H) + (df.D .== 1) .* (df.G .== 1) .* (df.S .== 10) .* ϕ(ζ1, θ12; H=H),
        (df.D .== 0) .* (df.G .== 0) .* (df.S .≈ 5) .* ψ′(θ01, df.Y; K=K),
        (df.D .== 0) .* (df.G .== 0) .* (df.S .≈ 10) .* ψ′(θ02, df.Y; K=K),
        (df.D .== 0) .* (df.G .== 1) .* (df.S .== 5) .* ϕ′(ζ0, θ01; H=H) + (df.D .== 0) .* (df.G .== 1) .* (df.S .== 10) .* ϕ′(ζ0, θ02; H=H),
        (df.G .== 0) .* (df.D .== 1) .* (q1 .- (df[:, :S] .== 5)),
        # (df.G .== 0) .* (df.D .== 1) .* (q2 .- (df[:, :S] .== 10)),
        (df.G .== 0) .* (df.D .== 0) .* (r1 .- (df[:, :S] .== 5)),
        # (df.G .== 0) .* (df.D .== 0) .* (r2 .- (df[:, :S] .== 10)),
        (df.D .== 1) .* (df.G .== 0) .* (df.Y .- m1),
        (df.D .== 0) .* (df.G .== 0) .* (df.Y .- m0))

    result = zeros(11, 11)
    for i in 1:n
        result += 1 / n * mom[i, :] * mom[i, :]'
    end

    result
end

function CI_τ_lb(data; K=1, H=1)

    param_init = [0.5, 5.0, 10.0, 7.5, 5.0, 10.0, 7.5, 0.3, 0.3, 7.0, 8.0]

    loss(param) = norm(moment_τ_bds(param, data; K=K, H=H))
    result = Optim.optimize(loss, param_init)
    param_est = result.minimizer

    τ_lb_func((p1, θ11, θ12, ζ1, θ01, θ02, ζ0, q1, r1, m1, m0)) = (ζ1 - p1 * (q1 * θ11 + (1 - q1) * θ12)) + p1 * m1 -
                                                                  ((ζ0 - (1 - p1) * (r1 * θ01 + (1 - r1) * θ02)) + (1 - p1) * m0)

    G = ForwardDiff.jacobian(x -> moment_τ_bds(x, data; K=K, H=H), param_est)[1:11, 1:11]
    V = cov_moment_τ_bds(param_est, data; K=K, H=H)[1:11, 1:11]

    grad_τ_lb = ForwardDiff.gradient(τ_lb_func, param_est[1:11])

    asym_var_τ_lb = grad_τ_lb' * inv(G' * inv(V) * G) * grad_τ_lb

    std_err_τ_lb = sqrt(asym_var_τ_lb) / sqrt(size(df2, 1))

    est = τ_lb_func(param_est[1:11])

    #return compute_lower_Y1(df2; K = K, H = H) - compute_upper_Y0(df2; K = K, H = H), std_err_τ_lb
    return est, std_err_τ_lb
end


function moment_τ_upper_bds(param, data; K=1, H=1)
    df = data
    p1, θ11, θ12, ζ1, θ01, θ02, ζ0, q1, r1, m1, m0 = param

    [p1 - mean(df[df.G.==0, :D])
        mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ 5) .* ψ(θ01, df.Y; K=K))
        mean((df.D .== 0) .* (df.G .== 0) .* (df.S .≈ 10) .* ψ(θ02, df.Y; K=K))
        mean((df.D .== 0) .* (df.G .== 1) .* (df.S .== 5) .* ϕ(ζ0, θ01; H=H)) + mean((df.D .== 0) .* (df.G .== 1) .* (df.S .== 10) .* ϕ(ζ0, θ02; H=H))
        mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ 5) .* ψ′(θ11, df.Y; K=K))
        mean((df.D .== 1) .* (df.G .== 0) .* (df.S .≈ 10) .* ψ′(θ12, df.Y; K=K))
        mean((df.D .== 1) .* (df.G .== 1) .* (df.S .== 5) .* ϕ′(ζ1, θ11; H=H)) + mean((df.D .== 1) .* (df.G .== 1) .* (df.S .== 10) .* ϕ′(ζ1, θ12; H=H))
        q1 - mean(df[(df.G.==0).*(df.D.==1), :S] .== 5)
        # q2 - mean(df[ (df.G.==0) .* (df.D .==1), :S] .== 10);
        r1 - mean(df[(df.G.==0).*(df.D.==0), :S] .== 5)
        # r2 - mean(df[ (df.G.==0) .* (df.D .==0), :S] .== 10);
        mean((df.D .== 1) .* (df.G .== 0) .* (df.Y .- m1))
        mean((df.D .== 0) .* (df.G .== 0) .* (df.Y .- m0))]
end


function cov_moment_τ_upper_bds(param, data; K=1, H=1)
    df = data
    p1, θ11, θ12, ζ1, θ01, θ02, ζ0, q1, r1, m1, m0 = param

    n = size(df, 1)

    mom = hcat((df.G .== 0) .* (p1 .- df[:, :D]),
        (df.D .== 0) .* (df.G .== 0) .* (df.S .≈ 5) .* ψ(θ01, df.Y; K=K),
        (df.D .== 0) .* (df.G .== 0) .* (df.S .≈ 10) .* ψ(θ02, df.Y; K=K),
        (df.D .== 0) .* (df.G .== 1) .* (df.S .== 5) .* ϕ(ζ0, θ01; H=H) + (df.D .== 0) .* (df.G .== 1) .* (df.S .== 10) .* ϕ(ζ0, θ02; H=H),
        (df.D .== 1) .* (df.G .== 0) .* (df.S .≈ 5) .* ψ′(θ11, df.Y; K=K),
        (df.D .== 1) .* (df.G .== 0) .* (df.S .≈ 10) .* ψ′(θ12, df.Y; K=K),
        (df.D .== 1) .* (df.G .== 1) .* (df.S .== 5) .* ϕ′(ζ1, θ11; H=H) + (df.D .== 1) .* (df.G .== 1) .* (df.S .== 10) .* ϕ′(ζ1, θ12; H=H),
        (df.G .== 0) .* (df.D .== 1) .* (q1 .- (df[:, :S] .== 5)),
        # (df.G .== 0) .* (df.D .== 1) .* (q2 .- (df[:, :S] .== 10)),
        (df.G .== 0) .* (df.D .== 0) .* (r1 .- (df[:, :S] .== 5)),
        # (df.G .== 0) .* (df.D .== 0) .* (r2 .- (df[:, :S] .== 10)),
        (df.D .== 1) .* (df.G .== 0) .* (df.Y .- m1),
        (df.D .== 0) .* (df.G .== 0) .* (df.Y .- m0))

    result = zeros(11, 11)
    for i in 1:n
        result += 1 / n * mom[i, :] * mom[i, :]'
    end

    result
end

function CI_τ_ub(data; K=1, H=1)

    param_init = [0.5, 5.0, 10.0, 7.5, 5.0, 10.0, 7.5, 0.3, 0.3, 7.0, 8.0]

    loss(param) = norm(moment_τ_upper_bds(param, data; K=K, H=H))
    result = Optim.optimize(loss, param_init)
    param_est = result.minimizer

    τ_ub_func((p1, θ11, θ12, ζ1, θ01, θ02, ζ0, q1, r1, m1, m0)) = (ζ1 - p1 * (q1 * θ11 + (1 - q1) * θ12)) + p1 * m1 -
                                                                  ((ζ0 - (1 - p1) * (r1 * θ01 + (1 - r1) * θ02)) + (1 - p1) * m0)

    G = ForwardDiff.jacobian(x -> moment_τ_upper_bds(x, data; K=K, H=H), param_est)[1:11, 1:11]
    V = cov_moment_τ_upper_bds(param_est, data; K=K, H=H)[1:11, 1:11]

    grad_τ_ub = ForwardDiff.gradient(τ_ub_func, param_est[1:11])

    asym_var_τ_ub = grad_τ_ub' * inv(G' * inv(V) * G) * grad_τ_ub

    std_err_τ_ub = sqrt(asym_var_τ_ub) / sqrt(size(df2, 1))

    est = τ_ub_func(param_est[1:11])

    #return compute_upper_Y1(df2; K = K, H = H) - compute_lower_Y0(df2; K = K, H = H), std_err_τ_lb
    return est, std_err_τ_ub
end