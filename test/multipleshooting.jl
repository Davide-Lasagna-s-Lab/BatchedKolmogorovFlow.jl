# The propagator in the searches of MultipleShooting.jl, and the seeds of examples/orbits.jl.

@testset "make_seeds, on the grid of the recurrence" begin
    Ωs = okf_fields(1)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)
    x  = batch(Ωs)
    T  = 5.37

    # ---- five segments of 1.074, not a whole number of steps ----
    X = make_seeds(P, (X=FTFields(Ωs), T=T, s=0.0, kind=:seeds), 1.0)
    @test size(X, 3) == 5
    @test parent(X) isa ARRAY
    @test cpu(X)[:, :, 1] == cpu(x)[:, :, 1]

    # ---- each seed lies on the trajectory by whole steps, the rest of a step taken on a copy ----
    for q in 2:5
        t = (q - 1)*T/5
        w = floor(Int, t/dt)
        y = propagate!(P, propagate!(P, copy(x), w*dt), t - w*dt)
        @test reldiff(cpu(X)[:, :, q], y) < 1e-13
    end

    # ---- going on from each shorter step leaves that trajectory ----
    z = copy(x)
    for q in 1:4
        propagate!(P, z, T/5)
    end
    @test reldiff(cpu(X)[:, :, 5], z) > 1e3*reldiff(cpu(X)[:, :, 5], propagate!(P, propagate!(P, copy(x), 429*dt), 4T/5 - 429*dt))

    # ---- dense seeds are the states of the file, with no integration ----
    Y = FTFields(cat(cpu(x), 2 .* cpu(x); dims=3))
    @test cpu(make_seeds(P, (X=Y, T=2.2, s=0.0, kind=:seeds), 1.1)) == cpu(Y)
end

@testset "as a propagator of MultipleShooting    " begin
    Ωs = okf_fields(4)
    Λs = okf_fields(4; seed=2, laminar=false)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)
    z  = MultiVector(batch(Ωs), [2.0, 0.3])
    v  = MultiVector(batch(Λs), [0.0, -0.2])
    F  = Residual(P)
    J  = Jacobian(P, z)

    # ---- the inner product of the unknowns is that of the fields ----
    @test dot(z, v) ≈ realdot(z.X, v.X) + dot(z.d, v.d)

    # ---- the product with the Jacobian against the centred difference of the residual: the seeds
    #      and the shift to rounding of the difference, the period up to the error of the scheme ----
    difference(v, ε) = (F(similar(z), z .+ ε .* v) .- F(similar(z), z .- ε .* v)) ./ (2ε)
    @test reldiff((J*v).X, difference(v, 1e-6).X) < 1e-7
    w = MultiVector(fill!(similar(z.X), 0), [1.0, 0.0])
    @test reldiff((J*w).X, difference(w, 1e-6).X) < 1e-4

    # ---- the phase conditions, in time and in space, in the inner product of the fields ----
    Jv = J*v
    u₁ = view(z.X, :, :, 1:1)
    v₁ = view(v.X, :, :, 1:1)
    @test MultipleShooting.period(Jv) ≈ realdot(rhs!(P, similar(u₁), u₁), v₁)
    @test MultipleShooting.shift(Jv, 1) ≈ realdot(dshift!(P, similar(u₁), u₁, 1), v₁)

    # ---- the column of the Reynolds number, from forced!, against the centred difference of the
    #      residual at the propagators of the family ----
    family(R) = KolmogorovPropagator(P, R)
    c = MultipleShooting._forced(P, z)
    d = (Residual(family(Re + 1e-4))(similar(z), z) .- Residual(family(Re - 1e-4))(similar(z), z)) ./ 2e-4
    @test reldiff(c.X, d.X) < 1e-4

    # ---- the bases of the subspace iteration are fields, orthonormal in the inner product ----
    Random.seed!(1)
    S = Subspace(P, z; k′=2)
    iterate!(S)
    @test all(all(products(S.V[a], S.V[a]) .≈ 1) for a in 1:2)
    @test maximum(abs, products(S.V[1], S.V[2])) < 1e-12
    @test all(reldiff(KS.apply_symmetry!(copy(S.V[a])), S.V[a]) < 1e-14 for a in 1:2)
end
