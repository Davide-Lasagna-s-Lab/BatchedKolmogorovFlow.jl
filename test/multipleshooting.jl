# The objects of the dynamics in the searches of MultipleShooting.jl, and the seeds of
# examples/orbits.jl.

@testset "make_seeds, on the grid of the recurrence" begin
    Ωs = okf_fields(1)
    P  = dynamics(1)
    x  = batch(Ωs)
    T  = 5.37

    # ---- five segments of 1.074, not a whole number of steps ----
    X = make_seeds(P.G, (X=FTFields(Ωs), T=T, s=0.0, kind=:seeds), 1.0)
    @test size(X, 3) == 5
    @test parent(X) isa ARRAY
    @test cpu(X)[:, :, 1] == cpu(x)[:, :, 1]

    # ---- each seed lies on the trajectory by whole steps, the rest of a step taken on a copy ----
    for q in 2:5
        t = (q - 1)*T/5
        w = floor(Int, t/dt)
        y = P.G(P.G(copy(x), w*dt), t - w*dt)
        @test reldiff(cpu(X)[:, :, q], y) < 1e-13
    end

    # ---- going on from each shorter step leaves that trajectory ----
    z = copy(x)
    for q in 1:4
        P.G(z, T/5)
    end
    @test reldiff(cpu(X)[:, :, 5], z) > 1e3*reldiff(cpu(X)[:, :, 5], P.G(P.G(copy(x), 429*dt), 4T/5 - 429*dt))

    # ---- dense seeds are the states of the file, with no integration ----
    Y = FTFields(cat(cpu(x), 2 .* cpu(x); dims=3))
    @test cpu(make_seeds(P.G, (X=Y, T=2.2, s=0.0, kind=:seeds), 1.1)) == cpu(Y)
end

@testset "the dynamics in MultipleShooting        " begin
    Ωs = okf_fields(4)
    Λs = okf_fields(4; seed=2, laminar=false)
    D  = dynamics(4)
    S  = XShift()
    z  = ShootingVector(batch(Ωs), [2.0, 0.3])
    v  = ShootingVector(batch(Λs), [0.0, -0.2])
    F  = Residual(D.G; S=S)
    J  = Jacobian(D.L, D.F, z; S=S, dS=S)

    # ---- the inner product of the unknowns is that of the fields ----
    @test dot(z, v) ≈ dot(z.X, v.X) + dot(z.d, v.d)

    # ---- the product with the Jacobian against the centred difference of the residual: the seeds
    #      and the shift to rounding of the difference, the period up to the error of the scheme ----
    difference(v, ε) = (F(similar(z), z .+ ε .* v) .- F(similar(z), z .- ε .* v)) ./ (2ε)
    @test reldiff((J*v).X, difference(v, 1e-6).X) < 1e-7
    w = ShootingVector(fill!(similar(z.X), 0), [1.0, 0.0])
    @test reldiff((J*w).X, difference(w, 1e-6).X) < 1e-4

    # ---- the phase conditions, in time and in space, in the inner product of the fields, at the
    #      first seed ----
    Jv = J*v
    f₁ = view(D.F(similar(z.X), z.X), :, :, 1:1)
    u₁ = view(z.X, :, :, 1:1)
    v₁ = view(v.X, :, :, 1:1)
    @test MultipleShooting.period(Jv) ≈ dot(f₁, v₁)
    @test MultipleShooting.shift(Jv, 1) ≈ dot(S(similar(u₁), u₁, 1), v₁)

    # ---- the column of the Reynolds number, from the forced tangent, against the centred
    #      difference of the residual at the flows of the family ----
    c = MultipleShooting._forced(D.Lp, S, z)
    d = (Residual(at(D, Re + 1e-4).G; S=S)(similar(z), z) .-
         Residual(at(D, Re - 1e-4).G; S=S)(similar(z), z)) ./ 2e-4
    @test reldiff(c.X, d.X) < 1e-4

    # ---- the bases of the subspace iteration are fields, orthonormal in the inner product ----
    Random.seed!(1)
    B = Subspace(dynamics(4; k′=2).L, z; k′=2, S=S)
    iterate!(B)
    @test all(all(products(B.V[k₁], B.V[k₁]) .≈ 1) for k₁ in 1:2)
    @test maximum(abs, products(B.V[1], B.V[2])) < 1e-12
    @test all(reldiff(KS.apply_symmetry!(copy(B.V[k₁])), B.V[k₁]) < 1e-14 for k₁ in 1:2)
end
