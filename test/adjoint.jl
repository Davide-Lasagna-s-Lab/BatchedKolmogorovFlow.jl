# adjoint!, the transpose of tangent! in the inner product of the fields.

@testset "adjoint!, the transpose of tangent!    " begin
    X = random_fields(3; seed=1)
    V = random_fields(3; seed=2)
    W = random_fields(3; seed=3)
    P = KolmogorovPropagator(n, m, Re, kf; B=3, k′=1, dt=dt, device=DEVICE, adjoint=true)

    # ---- the identity of the transpose, field by field: τ a whole number of steps and not (100
    #      steps, and 124 with the last one shorter), a single step shorter than dt, and no step ----
    for τ in (1.0, 1.234, 0.004, 0.0)
        a = products(tangent!(P, copy(X), copy(V), τ), W)
        b = products(V, adjoint!(P, stages(P, X, τ), copy(W)))
        @test all(relerr(a, b) .< 1e-12)
    end

    # ---- and on what is not a field, which both project first ----
    Vj = plus(V, 1, junk(3; seed=5))
    Wj = plus(W, 1, junk(3; seed=6))
    S  = stages(P, X, 1.234)
    a  = products(tangent!(P, copy(X), copy(Vj), 1.234), Wj)
    b  = products(Vj, adjoint!(P, S, copy(Wj)))
    @test all(relerr(a, b) .< 1e-12)
    @test cpu(adjoint!(P, S, copy(Wj))) == cpu(adjoint!(P, S, KS.apply_symmetry!(copy(Wj))))

    # ---- stages leaves the fields at the start as they are, and keeps the four stages of each of
    #      the 124 steps, of the type of the batches ----
    X₀ = copy(X)
    S  = stages(P, X, 1.234)
    @test cpu(X) == cpu(X₀)
    @test length(S.xs) == 124 && all(length(stage) == 4 for stage in S.xs)
    @test parent(S.xs[1][1]) isa ARRAY && size(S.xs[1][1]) == size(X)

    # ---- the result is a batch of fields, and the stages serve any number of calls, as they are ----
    Z = adjoint!(P, S, copy(W))
    @test cpu(KS.apply_symmetry!(copy(Z))) == cpu(Z)
    @test cpu(adjoint!(P, S, copy(W))) == cpu(Z)
    @test length(S.xs) == 124

    # ---- linear ----
    @test reldiff(adjoint!(P, S, plus(W, 2.5, V)), plus(Z, 2.5, adjoint!(P, S, copy(V)))) < 1e-13

    # ---- the workspace of adjoint! only when asked for, and shared at another Reynolds number ----
    Q = KolmogorovPropagator(n, m, Re, kf; B=3, k′=1, dt=dt, device=DEVICE)
    @test Q.adjoint === nothing
    @test_throws ArgumentError adjoint!(Q, stages(Q, X, 0.05), copy(W))
    @test KolmogorovPropagator(P, 2Re).adjoint === P.adjoint
end

@testset "adjoint!, field by field as OKF + Flows " begin
    Ωs = okf_fields(3)
    Ψs = okf_fields(3; seed=4, laminar=false)
    P  = KolmogorovPropagator(n, m, Re, kf; B=3, dt=dt, device=DEVICE, adjoint=true)

    # ---- the adjoint of OKF in the discrete mode of Flows, back over the stages of the whole
    #      interval, cached by a forward run ----
    for τ in (1.0, 1.234)
        Z = adjoint!(P, stages(P, batch(Ωs), τ), batch(Ψs))
        for i in 1:3
            cache = RAMStageCache(4, OKF.FTField(n, m))
            G(copy(Ωs[i]), (0, τ), cache)
            Ψ = LA(copy(Ψs[i]), cache)
            @test reldiff(cpu(Z)[:, :, i], coefficients(Ψ)) < 1e-12
        end
    end
end

@testset "adjoint!: views, batch sizes, another Re, composition" begin
    X = random_fields(3; seed=1)
    V = random_fields(3; seed=2)
    W = random_fields(3; seed=3)
    P = propagators(n, m, Re, kf; sizes=[(3, 1), (1, 0), (5, 0)], dt=dt, device=DEVICE, adjoint=true)
    τ = 1.234
    Z = adjoint!(P, stages(P, X, τ), copy(W))

    # ---- views of a batch: the stages of its field alone, and its adjoint field alone, the others
    #      left as they are ----
    Y = copy(W)
    adjoint!(P, stages(P, view(X, :, :, 2:2), τ), view(Y, :, :, 2:2))
    @test reldiff(cpu(Y)[:, :, 2:2], cpu(Z)[:, :, 2:2]) < 1e-13
    @test cpu(Y)[:, :, [1, 3]] == cpu(W)[:, :, [1, 3]]

    # ---- a batch of one, and one field repeated ----
    Z₁ = adjoint!(P, stages(P, dev(FTFields(cpu(X)[:, :, 3:3])), τ), dev(FTFields(cpu(W)[:, :, 3:3])))
    @test reldiff(Z₁, cpu(Z)[:, :, 3:3]) < 1e-13
    Z₅ = adjoint!(P, stages(P, dev(FTFields(repeat(cpu(X)[:, :, 1:1], 1, 1, 5))), τ),
                     dev(FTFields(repeat(cpu(W)[:, :, 1:1], 1, 1, 5))))
    @test all(reldiff(cpu(Z₅)[:, :, j], cpu(Z)[:, :, 1]) < 1e-13 for j in 1:5)

    # ---- the propagators at another Reynolds number share the workspaces and give the results of
    #      those made there, the transpose of their own tangent ----
    Q = propagators(P, 35.0)
    R = propagators(n, m, 35.0, kf; sizes=[(3, 1)], dt=dt, device=DEVICE, adjoint=true)
    @test cpu(adjoint!(Q, stages(Q, X, τ), copy(W))) == cpu(adjoint!(R, stages(R, X, τ), copy(W)))
    a = products(tangent!(Q, copy(X), copy(V), τ), W)
    b = products(V, adjoint!(Q, stages(Q, X, τ), copy(W)))
    @test all(relerr(a, b) .< 1e-12)
    @test reldiff(adjoint!(Q, stages(Q, X, τ), copy(W)), Z) > 1e-3
    @test cpu(adjoint!(P, stages(P, X, τ), copy(W))) == cpu(Z)

    # ---- over two intervals one after the other: the adjoint of the second, from the fields at its
    #      start, then that of the first, the transpose of the tangent over both ----
    τ₁ = 0.5
    τ₂ = τ - τ₁
    X₁ = propagate!(P, copy(X), τ₁)
    Z₂ = adjoint!(P, stages(P, X, τ₁), adjoint!(P, stages(P, X₁, τ₂), copy(W)))
    @test reldiff(Z₂, Z) < 1e-12
end
