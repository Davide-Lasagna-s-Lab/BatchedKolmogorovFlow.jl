# The terms of the equation, of its tangent and of its adjoint, one evaluation each.

@testset "terms of the equation and of its tangent" begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    X  = batch(Ωs)
    V  = batch(Λs)

    # ---- explicit terms: the form of the advection and the transforms differ, so equal to rounding ----
    E  = KS.ExplicitTerm(X, m, kf)
    Et = KS.TangentTerm(bundle(X, V), 1, m, kf)
    dX = E(0.0, X, similar(X))

    # the term of the equation and of its tangent on the bundle of Y with W: the two parts
    tangent(Y, W) = (Z = bundle(Y, W); unbundle(Et(0.0, Z, similar(Z)), size(Y, 3)))
    dY, dV = tangent(X, V)
    @test reldiff(dY, dX) < 1e-14
    for i in 1:3
        d = ex(0.0, copy(Ωs[i]), OKF.FTField(n, m))
        @test reldiff(cpu(dX)[:, :, i], coefficients(d)) < 1e-13
        e = lex(0.0, copy(Ωs[i]), d, copy(Λs[i]), OKF.FTField(n, m))
        @test reldiff(cpu(dV)[:, :, i], coefficients(e)) < 1e-13
    end

    # ---- implicit term and its solve ----
    Y = mul!(similar(X), KS.ImplicitTerm(Re), X)
    Z = Flows.ImcA!(KS.ImplicitTerm(Re), 0.3, X, similar(X))
    for i in 1:3
        @test coefficients(mul!(OKF.FTField(n, m), im_, Ωs[i])) ≈ cpu(Y)[:, :, i] rtol=1e-15
        @test coefficients(Flows.ImcA!(im_, 0.3, Ωs[i], OKF.FTField(n, m))) ≈ cpu(Z)[:, :, i] rtol=1e-14
    end

    # ---- the forcing alone is the term at zero, -kf/2 at the two modes j = 0, k = ±kf ----
    F = E(0.0, fill!(similar(X), 0), similar(X))
    @test all(cpu(F)[1, KS._column(kf, n), :] .== -kf/2)
    @test all(cpu(F)[1, KS._column(-kf, n), :] .== -kf/2)
    @test count(!iszero, cpu(F)) == 2*3

    # ---- the advection conserves the enstrophy and the energy: it is orthogonal to ω and to ψ ----
    N = plus(dX, -1, F)
    @test all(abs.(products(N, X)) .< 1e-13 .* sqrt.(products(N, N) .* products(X, X)))
    Ψ = KS.invlaplacian!(similar(X), X)
    @test all(abs.(products(N, Ψ)) .< 1e-13 .* sqrt.(products(N, N) .* products(Ψ, Ψ)))

    # ---- the tangent term is the derivative of the explicit one, which is quadratic, so that a
    #      centred difference of any size gives it to rounding ----
    Ep = E(0.0, plus(X, 1, V), similar(X))
    Em = E(0.0, plus(X, -1, V), similar(X))
    @test reldiff((Ep .- Em) ./ 2, dV) < 1e-13

    # ---- linear in the perturbations, a field from fields, and a batch of one as in a batch ----
    W  = random_fields(3)
    dW = last(tangent(X, W))
    @test reldiff(last(tangent(X, plus(V, 2.5, W))), plus(dV, 2.5, dW)) < 1e-14
    @test cpu(KS.apply_symmetry!(copy(dV))) == cpu(dV)
    X₂  = dev(FTFields(cpu(X)[:, :, 2:2]))
    V₂  = dev(FTFields(cpu(V)[:, :, 2:2]))
    Z₂  = bundle(X₂, V₂)
    Et₁ = KS.TangentTerm(Z₂, 1, m, kf)
    @test reldiff(last(unbundle(Et₁(0.0, Z₂, similar(Z₂)), 1)), cpu(dV)[:, :, 2:2]) < 1e-14

    # ---- two perturbations of every field, in two blocks: each as alone, and the fields once ----
    Z₃  = bundle(X, FTFields(cat(cpu(V), cpu(W); dims=3)) |> dev)
    Et₂ = KS.TangentTerm(Z₃, 2, m, kf)
    dY₂, dVW = unbundle(Et₂(0.0, Z₃, similar(Z₃)), 3)
    @test reldiff(dY₂, dX) < 1e-14
    @test reldiff(cpu(dVW)[:, :, 1:3], dV) < 1e-14
    @test reldiff(cpu(dVW)[:, :, 4:6], dW) < 1e-14

    # ---- the term forced in the Reynolds number adds -Δω/Re² ----
    Ef = KS.ForcedTangentTerm(Et, Re)
    Z  = bundle(X, V)
    dF = last(unbundle(Ef(0.0, Z, similar(Z)), 3))
    ΔX = KS.laplacian!(similar(X), X)
    @test reldiff(plus(dF, -1, dV), ΔX ./ -Re^2) < 1e-14

    # ---- the right hand side, of a batch and of a view of one field ----
    P  = propagators(n, m, Re, kf; sizes=[(3, 0), (1, 0)], array=ARRAY)
    eq = OKF.ForwardEquation(n, m, Re, kf, FFTW.ESTIMATE)
    R  = rhs!(P, similar(X), X)
    for i in 1:3
        @test reldiff(cpu(R)[:, :, i], coefficients(eq(0.0, copy(Ωs[i]), OKF.FTField(n, m)))) < 1e-13
    end
    @test reldiff(rhs!(P, similar(view(X, :, :, 2:2)), view(X, :, :, 2:2)), cpu(R)[:, :, 2:2]) < 1e-13
end

@testset "term of the adjoint                    " begin
    X = random_fields(4; seed=1)
    V = random_fields(4; seed=2)
    W = random_fields(4; seed=3)

    Z  = bundle(X, V)
    Et = KS.TangentTerm(Z, 1, m, kf)
    Ea = KS.AdjointTerm(X, m)
    dV = last(unbundle(Et(0.0, Z, similar(Z)), 4))
    dW = Ea(0.0, X, W, similar(W))

    # ---- the transpose of the tangent term in the inner product, field by field ----
    @test all(relerr(products(dV, W), products(V, dW)) .< 1e-12)

    # ---- the adjoint term of OKF, the formula of the continuous adjoint, field by field ----
    for i in 1:4
        e = aex(0.0, OKF.FTField(X, i, m), OKF.FTField(W, i, m), OKF.FTField(n, m))
        @test reldiff(cpu(dW)[:, :, i], coefficients(e)) < 1e-13
    end

    # ---- linear, a field from fields, the input left as it is, and a batch of one as in a batch ----
    W₀ = copy(W)
    Ea(0.0, X, W, similar(W))
    @test cpu(W) == cpu(W₀)
    @test reldiff(Ea(0.0, X, plus(W, 2.5, V), similar(W)), plus(dW, 2.5, Ea(0.0, X, V, similar(V)))) < 1e-14
    @test cpu(KS.apply_symmetry!(copy(dW))) == cpu(dW)
    X₃  = dev(FTFields(cpu(X)[:, :, 3:3]))
    Ea₁ = KS.AdjointTerm(X₃, m)
    @test reldiff(Ea₁(0.0, X₃, view(W, :, :, 3:3), similar(X₃)), cpu(dW)[:, :, 3:3]) < 1e-14
end
