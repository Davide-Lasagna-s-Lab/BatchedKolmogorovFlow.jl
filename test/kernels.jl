# The terms entry by entry (src/kernels.jl), three loops on the CPU and three kernels on the GPU,
# against their broadcasts: the terms, and the objects of the dynamics made with and without.

@testset "kernels against the broadcasts" begin
    X  = random_fields(3)
    V  = random_fields(3; seed=8)
    Vs = [random_fields(3; seed=20 + a) for a in 1:3]
    W  = random_fields(3; seed=9)
    τ  = 10.5dt

    # ---- the explicit term, and the term of a bundle with one perturbation of every field and
    #      with three, forced or not ----
    E(kernels) = KS.ExplicitTerm(X, m, kf; kernels=kernels)(0.0, X, similar(X))
    @test reldiff(E(true), E(false)) < 1e-14
    for k′ in (1, 3)
        Z          = KS._bundle!(similar(X, ComplexF64, (n + 1, 2n + 1, 3*(1 + k′))), X, Vs[1:k′])
        T(kernels) = KS.TangentTerm(Z, k′, m, kf; kernels=kernels)
        @test reldiff(T(true)(0.0, Z, similar(Z)), T(false)(0.0, Z, similar(Z))) < 1e-14
        @test reldiff(KS.ForcedTangentTerm(T(true), Re)(0.0, Z, similar(Z)),
                      KS.ForcedTangentTerm(T(false), Re)(0.0, Z, similar(Z))) < 1e-14
    end

    # ---- the objects of the dynamics: the flow, the tangent with one perturbation and with
    #      three, the forced tangent from zero, and the right-hand side ----
    D, K   = dynamics(3; kernels=false), dynamics(3)
    D₃, K₃ = dynamics(3; k′=3, kernels=false), dynamics(3; k′=3)
    F      = RightHandSide(n, m, Re, kf; B=3, device=DEVICE, kernels=false)
    zero!  = Y -> fill!(Y, 0)
    @test reldiff(K.G(copy(X), τ), D.G(copy(X), τ)) < 1e-12
    @test reldiff(K.L(copy(X), copy(V), τ), D.L(copy(X), copy(V), τ)) < 1e-12
    Vk = K₃.L(copy(X), copy.(Vs), τ)
    Vd = D₃.L(copy(X), copy.(Vs), τ)
    @test all(reldiff(Vk[a], Vd[a]) < 1e-12 for a in 1:3)
    @test reldiff(K.Lp(copy(X), zero!(similar(X)), τ), D.Lp(copy(X), zero!(similar(X)), τ)) < 1e-12
    @test reldiff(K.F(similar(X), X), F(similar(X), X)) < 1e-13

    # ---- at another Reynolds number, the forcing of the forced tangent with it ----
    D′, K′ = at(D, 55.0), at(K, 55.0)
    @test reldiff(K′.G(copy(X), τ), D′.G(copy(X), τ)) < 1e-12
    @test reldiff(K′.Lp(copy(X), zero!(similar(X)), τ), D′.Lp(copy(X), zero!(similar(X)), τ)) < 1e-12

    # ---- the stages kept for the adjoint ----
    @test reldiff(D.A(copy(W), stages(K.G, X, τ)), D.A(copy(W), stages(D.G, X, τ))) < 1e-12
end
