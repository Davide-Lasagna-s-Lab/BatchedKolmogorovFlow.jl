using Test
using LinearAlgebra
using Random
using FFTW
using Flows
using OpenKolmogorovFlow
using KolmogorovShootingGPU

const KS = KolmogorovShootingGPU

# ---- a small grid and a few smooth fields near the laminar flow, as FTFields of OKF ----
const n  = 8
const m  = 12
const Re = 40.0
const kf = 4
const dt = 0.01

function okf_fields(B; seed=1, laminar=true)
    Random.seed!(seed)
    return [begin
                a = rand(6)
                u = Field(m, (x, y) -> sin(a[1] + x)*cos(2y + a[2]) + 0.5*sin(3x + 2y + a[3]) +
                                       0.3*cos(x - 4y + a[4]) + 0.2*sin(5x + a[5])*cos(y + a[6]))
                Ω = FFT(u, n)
                laminar && (Ω .+= laminarflow(n, m, Re, kf))
                Ω
            end for i in 1:B]
end

# the batch of the data of the fields
batch(Ωs) = cat([parent(Ω) for Ω in Ωs]...; dims=3)

# the largest difference relative to the largest coefficient
reldiff(A, B) = maximum(abs, A .- B) / maximum(abs, B)

# the compact batch of a batch of fields
compact(P, X) = KS._compress!(P, similar(X, n + 1, 2n + 1, size(X, 3)), X, KS._forward(P, size(X, 3)).explicit)

# the active modes of OKF, without the mean
active() = [Float64(abs(k) <= n && j <= n && !(k == 0 && j == 0)) for k in KS._ks(m), j in KS._js(m)]

# OKF and Flows: the equation, its tangent, and their flows with CB3R2R3e
const ex, im_ = splitexim(ForwardEquation(n, m, Re, kf, FFTW.ESTIMATE))
const lex, lim = splitexim(LinearisedEquation(n, m, Re, TangentMode(), FFTW.ESTIMATE))
const G = flow(ex, im_, CB3R2R3e(FTField(n, m)), TimeStepConstant(dt))
const L = flow(couple(ex, lex), couple(im_, lim), CB3R2R3e(couple(FTField(n, m), FTField(n, m))),
               TimeStepConstant(dt))

@testset "terms of the equation and of its tangent" begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    X  = batch(Ωs)
    V  = batch(Λs)
    P  = KolmogorovPropagator(n, m, Re, kf; batch=3)

    # ---- explicit terms: the transforms differ, so equal to round-off ----
    Xc = compact(P, X)
    Vc = compact(P, V)
    E  = KS.ExplicitTerm(P.template, n, m, kf, 3)
    Et = KS.ExplicitTerm(P.template, n, m, kf, 3; base=E)
    dX = E(0.0, Xc, similar(Xc))
    dV = Et(0.0, Xc, dX, Vc, similar(Vc))
    for i in 1:3
        d = ex(0.0, copy(Ωs[i]), FTField(n, m))
        @test reldiff(dX[:, :, i], compact(P, parent(d)[:, :, :])[:, :, 1]) < 1e-14
        e = lex(0.0, copy(Ωs[i]), d, copy(Λs[i]), FTField(n, m))
        @test reldiff(dV[:, :, i], compact(P, parent(e)[:, :, :])[:, :, 1]) < 1e-14
    end

    # ---- implicit term and its solve: the same arithmetic, so equal ----
    Y = mul!(similar(Xc), P.implicit, Xc)
    Z = Flows.ImcA!(P.implicit, 0.3, Xc, similar(Xc))
    W = KS._implicit!(similar(Xc), P.implicit, 0.3, Xc)
    for i in 1:3
        @test compact(P, parent(mul!(FTField(n, m), im_, Ωs[i]))[:, :, :])[:, :, 1] == Y[:, :, i]
        @test compact(P, parent(Flows.ImcA!(im_, 0.3, Ωs[i], FTField(n, m)))[:, :, :])[:, :, 1] ≈ Z[:, :, i] rtol=1e-14
    end
    @test Flows.ImcA!(P.implicit, 0.3, Y, similar(Xc)) ≈ W rtol=1e-14

    # ---- the right hand side ----
    eq = ForwardEquation(n, m, Re, kf, FFTW.ESTIMATE)
    R  = rhs!(P, similar(X), X)
    for i in 1:3
        @test reldiff(R[:, :, i], parent(eq(0.0, copy(Ωs[i]), FTField(n, m)))) < 1e-14
    end
end

@testset "propagate!, slice by slice as OKF + Flows" begin
    Ωs = okf_fields(4)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, batch=4)
    @test collect(keys(P.cache)) == [4]

    # ---- a span that is a whole number of steps, and one that is not, as the segments of a search ----
    for τ in (1.0, 2.345)
        X = propagate!(P, batch(Ωs), τ)
        for i in 1:4
            Ω = G(copy(Ωs[i]), (0, τ))
            @test reldiff(X[:, :, i], parent(Ω)) < 1e-13
            @test sqrt(normdiff(Ω, Ωs[i])) > 1e-2
        end
    end

    # ---- any batch size: two of the fields alone, and one field repeated ----
    X = propagate!(P, batch(Ωs), 2.345)
    Y = propagate!(P, batch(Ωs[1:2]), 2.345)
    Z = propagate!(P, batch([Ωs[3] for i in 1:5]), 2.345)
    @test reldiff(Y, X[:, :, 1:2]) < 1e-13
    @test all(reldiff(Z[:, :, j], X[:, :, 3]) < 1e-13 for j in 1:5)
    @test sort(collect(keys(P.cache))) == [2, 4, 5]

    # ---- a view of a batch ----
    Y = batch(Ωs)
    propagate!(P, view(Y, :, :, 1:2), 2.345)
    @test reldiff(Y[:, :, 1:2], X[:, :, 1:2]) < 1e-13
    @test Y[:, :, 3:4] == batch(Ωs[3:4])

    # ---- zero inactive modes and zero mean ----
    mask = active()
    @test all(X[:, :, 1][mask .== 0] .== 0)

    # ---- entries outside the fields of OKF are projected out: inactive modes, mean, the j = 0 column not Hermitian ----
    Random.seed!(3)
    J  = randn(ComplexF64, 2m + 2, m + 2, 4) .* (1 .- mask)
    J[2:(m + 1), 1, :] .+= randn(ComplexF64, m, 4)
    J[(2m + 2):-1:(m + 3), 1, :] .= .-conj.(J[2:(m + 1), 1, :])
    Y  = propagate!(P, batch(Ωs) .+ J, 2.345)
    @test reldiff(Y, X) < 1e-13
end

@testset "tangent!, slice by slice as OKF + Flows " begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt)

    # ---- the coupled tangent equation of OKF ----
    τ = 1.234
    X = batch(Ωs)
    V = batch(Λs)
    tangent!(P, X, V, τ)
    for i in 1:3
        Ω = copy(Ωs[i])
        Λ = copy(Λs[i])
        L(couple(Ω, Λ), (0, τ))
        @test reldiff(X[:, :, i], parent(Ω)) < 1e-13
        @test reldiff(V[:, :, i], parent(Λ)) < 1e-12
    end

    # ---- the derivative of the flow: a centred difference of propagate! ----
    h  = 1e-5
    Xp = propagate!(P, batch(Ωs) .+ h .* batch(Λs), τ)
    Xm = propagate!(P, batch(Ωs) .- h .* batch(Λs), τ)
    @test reldiff((Xp .- Xm) ./ (2h), V) < 1e-7

    # ---- the tangent vanishes outside the fields of OKF ----
    mask = active()
    Random.seed!(4)
    J = randn(ComplexF64, 2m + 2, m + 2, 3) .* (1 .- mask)
    J[2:(m + 1), 1, :] .+= randn(ComplexF64, m, 3)
    J[(2m + 2):-1:(m + 3), 1, :] .= .-conj.(J[2:(m + 1), 1, :])
    tangent!(P, batch(Ωs), J, τ)
    @test maximum(abs, J) == 0
end

@testset "shift, generator and inner product      " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)
    P  = KolmogorovPropagator(n, m, Re, kf)

    # ---- the shift along x is OKF's xshift!, and along y its yshift! by quarter periods ----
    s  = 0.731
    Xs = shift!(P, copy(X), s)
    Xy = shift!(P, copy(X), s, 2π/kf)
    for i in 1:3
        @test parent(xshift!(copy(Ωs[i]), s)) ≈ Xs[:, :, i] rtol=1e-15
        @test parent(yshift!(xshift!(copy(Ωs[i]), s), 1)) ≈ Xy[:, :, i] rtol=1e-14
    end

    # ---- a view of the last field, and trailing batch dimensions ----
    Y = copy(X)
    shift!(P, view(Y, :, :, 3:3), s)
    @test Y[:, :, 1:2] == X[:, :, 1:2]
    @test Y[:, :, 3] == Xs[:, :, 3]
    W = reshape(copy(X), size(X, 1), size(X, 2), 1, 3)
    shift!(P, W, s)
    @test reshape(W, size(X)) == Xs

    # ---- OKF's shift! is still there ----
    @test parent(shift!(copy(Ωs[1]), s, 1)) ≈ Xy[:, :, 1] rtol=1e-14

    # ---- the generator is OKF's ddx! ----
    D = dshift!(P, similar(X), X)
    for i in 1:3
        @test parent(ddx!(FTField(n, m), Ωs[i])) == D[:, :, i]
    end

    # ---- the inner product of OKF, summed over the fields ----
    Y = batch(okf_fields(3; seed=3))
    @test realdot(X, Y) ≈ sum(dot(Ωs[i], FTField(Y[:, :, i], n)) for i in 1:3) rtol=1e-14
    @test realdot(X, X) ≈ sum(norm(Ω)^2 for Ω in Ωs) rtol=1e-14
    @test realdot(X, X) ≈ Re * sum(dissrate(Ω, Re) for Ω in Ωs) rtol=1e-14

    # ---- checks ----
    @test_throws ArgumentError KolmogorovPropagator(8, 6, Re, kf)
    @test_throws ArgumentError KolmogorovPropagator(n, m, Re, 9)
end
