# The adjoint propagator, the transpose of the tangent in the inner product of the fields.

@testset "the adjoint, the transpose of the tangent" begin
    X = random_fields(3; seed=1)
    V = random_fields(3; seed=2)
    W = random_fields(3; seed=3)
    P = dynamics(3)

    # ---- the identity of the transpose, field by field: τ a whole number of steps and not (100
    #      steps, and 124 with the last one shorter), a single step shorter than dt, and no step ----
    for τ in (1.0, 1.234, 0.004, 0.0)
        a = products(P.L(copy(X), copy(V), τ), W)
        b = products(V, P.A(copy(W), stages(P.G, X, τ)))
        @test all(relerr(a, b) .< 1e-12)
    end

    # ---- and on what is not a field, which both project first ----
    Vj = plus(V, 1, junk(3; seed=5))
    Wj = plus(W, 1, junk(3; seed=6))
    S  = stages(P.G, X, 1.234)
    a  = products(P.L(copy(X), copy(Vj), 1.234), Wj)
    b  = products(Vj, P.A(copy(Wj), S))
    @test all(relerr(a, b) .< 1e-12)
    @test cpu(P.A(copy(Wj), S)) == cpu(P.A(KS.apply_symmetry!(copy(Wj)), S))

    # ---- stages leaves the fields at the start as they are, and keeps the four stages of each of
    #      the 124 steps, of the type of the batches ----
    X₀ = copy(X)
    S  = stages(P.G, X, 1.234)
    @test cpu(X) == cpu(X₀)
    @test length(S.xs) == 124 && all(length(stage) == 4 for stage in S.xs)
    @test parent(S.xs[1][1]) isa ARRAY && size(S.xs[1][1]) == size(X)

    # ---- the result is a batch of fields, and the stages serve any number of calls, as they are ----
    Z = P.A(copy(W), S)
    @test cpu(KS.apply_symmetry!(copy(Z))) == cpu(Z)
    @test cpu(P.A(copy(W), S)) == cpu(Z)
    @test length(S.xs) == 124

    # ---- linear ----
    @test reldiff(P.A(plus(W, 2.5, V), S), plus(Z, 2.5, P.A(copy(V), S))) < 1e-13

    # ---- its fields shared at another Reynolds number, and its flow made again there ----
    Q = AdjointPropagator(P.A, 2Re)
    @test Q.W === P.A.W && Q.ψ !== P.A.ψ && Q.Re == 2Re
end

@testset "the adjoint, field by field as OKF + Flows" begin
    Ωs = okf_fields(3)
    Ψs = okf_fields(3; seed=4, laminar=false)
    P  = dynamics(3)

    # ---- the adjoint of OKF in the discrete mode of Flows, back over the stages of the whole
    #      interval, cached by a forward run ----
    for τ in (1.0, 1.234)
        Z = P.A(batch(Ψs), stages(P.G, batch(Ωs), τ))
        for i in 1:3
            cache = RAMStageCache(4, OKF.FTField(n, m))
            G(copy(Ωs[i]), (0, τ), cache)
            Ψ = LA(copy(Ψs[i]), cache)
            @test reldiff(cpu(Z)[:, :, i], coefficients(Ψ)) < 1e-12
        end
    end
end

@testset "the adjoint: views, batch sizes, another Re, composition" begin
    X = random_fields(3; seed=1)
    V = random_fields(3; seed=2)
    W = random_fields(3; seed=3)
    P  = dynamics(3)
    P₁ = dynamics(1)
    P₅ = dynamics(5)
    τ = 1.234
    Z = P.A(copy(W), stages(P.G, X, τ))

    # ---- views of a batch: the stages of its field alone, and its adjoint field alone, the others
    #      left as they are ----
    Y = copy(W)
    P₁.A(view(Y, :, :, 2:2), stages(P₁.G, view(X, :, :, 2:2), τ))
    @test reldiff(cpu(Y)[:, :, 2:2], cpu(Z)[:, :, 2:2]) < 1e-13
    @test cpu(Y)[:, :, [1, 3]] == cpu(W)[:, :, [1, 3]]

    # ---- a batch of one, and one field repeated ----
    Z₁ = P₁.A(dev(FTFields(cpu(W)[:, :, 3:3])), stages(P₁.G, dev(FTFields(cpu(X)[:, :, 3:3])), τ))
    @test reldiff(Z₁, cpu(Z)[:, :, 3:3]) < 1e-13
    X₅ = dev(FTFields(repeat(cpu(X)[:, :, 1:1], 1, 1, 5)))
    W₅ = dev(FTFields(repeat(cpu(W)[:, :, 1:1], 1, 1, 5)))
    Z₅ = P₅.A(W₅, stages(P₅.G, X₅, τ))
    @test all(reldiff(cpu(Z₅)[:, :, j], cpu(Z)[:, :, 1]) < 1e-13 for j in 1:5)

    # ---- the propagator at another Reynolds number shares the workspaces and gives the results of
    #      one made there, the transpose of its own tangent ----
    Q = at(P, 35.0)
    R = dynamics(3; reynolds=35.0)
    @test cpu(Q.A(copy(W), stages(Q.G, X, τ))) == cpu(R.A(copy(W), stages(R.G, X, τ)))
    a = products(Q.L(copy(X), copy(V), τ), W)
    b = products(V, Q.A(copy(W), stages(Q.G, X, τ)))
    @test all(relerr(a, b) .< 1e-12)
    @test reldiff(Q.A(copy(W), stages(Q.G, X, τ)), Z) > 1e-3
    @test cpu(P.A(copy(W), stages(P.G, X, τ))) == cpu(Z)

    # ---- over two intervals one after the other: the adjoint of the second, from the fields at its
    #      start, then that of the first, the transpose of the tangent over both ----
    τ₁ = 0.5
    τ₂ = τ - τ₁
    X₁ = P.G(copy(X), τ₁)
    Z₂ = P.A(P.A(copy(W), stages(P.G, X₁, τ₂)), stages(P.G, X, τ₁))
    @test reldiff(Z₂, Z) < 1e-12
end
