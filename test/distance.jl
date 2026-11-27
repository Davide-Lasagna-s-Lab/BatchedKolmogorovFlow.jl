# The distance up to the symmetries and the moduli of the coefficients: against the distance of
# kf_recurrences_tight.jl on pairs of a trajectory, the bound from below, the invariance under the
# symmetries, and the residual of the shift that the distance returns.

# The distance of kf_recurrences_tight.jl (code of the paper on long orbits, 26 Nov), lines 119 to 203,
# at the resolution of the tests, as the reference: its classes of k mod 4, its phases i^(q sy), its
# FFT at 256 shifts and its Newton steps, with sy = 0, 1, 2, 3
function script_distance(a::FTFields, b::FTFields)
    L     = 256
    CLASS = [mod(k <= n + 1 ? k - 1 : k - 2n - 2, 4) + 1 for k in 1:(2n + 1)]
    PHASE = [ComplexF64(im^mod(q*s, 4)) for q in 0:3, s in 0:3]
    Gq    = zeros(ComplexF64, n + 1, 4)
    Gs    = zeros(ComplexF64, n + 1)
    C     = zeros(ComplexF64, L, 4)
    A     = cpu(a)
    B     = cpu(b)
    na    = 0.0
    nb    = 0.0
    for k in 1:(2n + 1), j in 1:(n + 1)
        w             = j == 1 ? 1.0 : 2.0
        Gq[j, CLASS[k]] += w*conj(A[j, k, 1])*B[j, k, 1]
        na           += w*abs2(A[j, k, 1])
        nb           += w*abs2(B[j, k, 1])
    end
    for s in 0:3, q in 1:4, j in 1:(n + 1)
        C[j, s + 1] += PHASE[q, s + 1]*Gq[j, q]
    end
    FFTW.bfft!(C, 1)
    c, at = findmax(real, C)
    sy    = at[2] - 1
    σ     = 2π*(at[1] - 1)/L
    for q in 1:4, j in 1:(n + 1)
        Gs[j] += PHASE[q, sy + 1]*Gq[j, q]
    end
    for _ in 1:4
        e, p, f′, f″ = cis(σ), one(ComplexF64), 0.0, 0.0
        for j in 1:n
            p  *= e
            g   = Gs[j + 1]*p
            f′ -= j*imag(g)
            f″ -= j^2*real(g)
        end
        f″ < 0 || break
        σ -= f′/f″
    end
    e, p, f = cis(σ), one(ComplexF64), real(Gs[1])
    for j in 1:n
        p *= e
        f += real(Gs[j + 1]*p)
    end
    if f < c
        f = c
        σ = 2π*(at[1] - 1)/L
    end

    return sqrt(max(na + nb - 2f, 0.0)/na), rem(σ, 2π, RoundNearest), sy
end

# the norm of one field
fieldnorm(a) = sqrt(products(a, a)[1])

@testset "the distance up to the symmetries      " begin
    # ---- eight states of one trajectory, 2.5 apart, on the CPU ----
    G = NonlinearPropagator(n, m, Re, kf; B=1, dt=dt)
    x = FTFields(okf_fields(1; seed=4, laminar=false))
    G(x, 20.0)
    U = [copy(G(x, 2.5)) for _ in 1:8]
    D = Distance(n, kf)

    for i in 1:8, l in (i + 1):8
        a = U[i]
        b = U[l]
        d, σ, sy = D(a, b)

        # ---- the distance of the script, the same shift and the same translation modulo kf ----
        d₀, σ₀, sy₀ = script_distance(a, b)
        @test d ≈ d₀ rtol=1e-12
        @test σ ≈ σ₀ atol=1e-10
        @test mod(sy - sy₀, kf) == 0

        # ---- the shift that it returns reaches it: ‖S(σ, sy) b - a‖/‖a‖ = d ----
        r = plus(XShift(; sy=sy, kf=kf)(copy(b), σ), -1, a)
        @test fieldnorm(r)/fieldnorm(a) ≈ d rtol=1e-10

        # ---- the moduli bound it from below, and their norm is that of the field ----
        @test norm(moduli(b) .- moduli(a)) <= d*norm(moduli(a))
        @test norm(moduli(a)) ≈ fieldnorm(a) rtol=1e-14

        # ---- the same distance when b is shifted and translated ----
        for (s₁, σ₁) in ((1, 0.37), (2, -2.1), (-1, 1.3))
            @test D(a, XShift(; sy=s₁, kf=kf)(copy(b), σ₁))[1] ≈ d rtol=1e-10
        end
    end

    # ---- a field and its own shift: no distance, and the inverse shift and translation ----
    a = U[3]
    b = XShift(; sy=1, kf=kf)(copy(a), 0.83)
    d, σ, sy = D(a, b)
    @test d < 1e-7
    @test σ ≈ -0.83 atol=1e-8
    @test mod(sy + 1, kf) == 0

    # ---- the translations given, along x alone: the distance can only be larger ----
    Dx = Distance(n, kf; sy=0:0)
    @test all(Dx(U[1], U[l])[1] >= D(U[1], U[l])[1] - 1e-14 for l in 2:8)
    @test Dx(U[1], U[2])[3] == 0
    @test D.sy == -1:2
end
