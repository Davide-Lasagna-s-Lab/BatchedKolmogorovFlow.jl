# The objects of the dynamics: their constructors, the flow, the tangent, the forced tangent, the
# right-hand side and the shift, the properties of the flow, and the workspaces.

# the rotation by π, ω(x, y) → ω(-x, -y): the coefficients conjugated
rotate(X) = FTFields(conj.(X))

# the shift-reflect, ω(x, y) → -ω(-x, y + π/kf): Ω[j, k] → -conj(Ω[j, -k]) e^{i k π/kf}
function shiftreflect(X)
    k = [0:n; -n:-1]
    c = mod.(.-k, 2n + 1) .+ 1
    B = .-conj.(cpu(X)[:, c, :]) .* cis.(k' .* π ./ kf)

    return dev(FTFields(B))
end

@testset "the objects of the dynamics              " begin
    D = dynamics(3; k′=2)

    # ---- what they hold: the size, and the workspaces on the device ----
    @test (D.G.Re, D.G.dt, D.G.B) == (Re, dt, 3)
    @test parent(D.G.whole.X) isa ARRAY{ComplexF64, 3}
    @test (D.L.B, D.L.k′, D.L.forced, D.Lp.forced) == (3, 2, false, true)
    @test size(D.L.chunks[1].Z, 3) == 9
    @test (D.F.B, D.A.B) == (3, 3)
    @test length(dynamics(5; tasks=3).L.chunks) == 3
    @test NonlinearPropagator(n, m, 40, kf; B=1).Re === 40.0
    @test NonlinearPropagator(n, m, Re, kf; B=1).dt == 0.01

    # ---- another size is an error ----
    @test_throws ArgumentError D.G(random_fields(2), 0.1)
    @test_throws ArgumentError D.F(random_fields(2), random_fields(2))
    @test_throws ArgumentError D.L(random_fields(3), random_fields(3), 0.1)
    @test_throws ArgumentError D.L(random_fields(3), [random_fields(3) for k₁ in 1:3], 0.1)
    @test_throws ArgumentError D.A(random_fields(2), stages(dynamics(2).G, random_fields(2), 0.1))

    # ---- at another Reynolds number, with the same workspaces and their flows made again ----
    E = at(D, 35.0)
    @test all(x.Re == 35.0 for x in E) && all(x.Re == Re for x in D)
    @test E.G.chunks[1].X === D.G.chunks[1].X && E.G.chunks[1].ϕ !== D.G.chunks[1].ϕ
    @test E.L.chunks[1].Z === D.L.chunks[1].Z && E.F.X === D.F.X && E.A.W === D.A.W
    @test XShift(XShift(), 35.0) === XShift()

    # ---- checks: no size, a size of no fields or of no perturbations, a device that is not one,
    #      or the GPU without CUDA, a grid that aliases the products, and a forcing outside the
    #      modes ----
    @test_throws UndefKeywordError NonlinearPropagator(n, m, Re, kf)
    @test_throws ArgumentError dynamics(0)
    @test_throws ArgumentError dynamics(1; k′=0)
    @test_throws ArgumentError dynamics(1; device=:tpu)
    DEVICE === :cpu && @test_throws ArgumentError NonlinearPropagator(n, m, Re, kf; B=1, device=:gpu)
    @test_throws ArgumentError NonlinearPropagator(8, 11, Re, kf; B=1)
    @test NonlinearPropagator(8, 12, Re, kf; B=1) isa NonlinearPropagator
    @test_throws ArgumentError NonlinearPropagator(n, m, Re, 9; B=1)
    @test_throws ArgumentError NonlinearPropagator(n, m, Re, 0; B=1)
end

@testset "the flow, field by field as OKF + Flows " begin
    Ωs = okf_fields(4)
    P  = dynamics(4)

    # ---- a span that is a whole number of steps, and one that is not, as the segments of a search ----
    for τ in (1.0, 2.345)
        X = P.G(batch(Ωs), τ)
        for i in 1:4
            Ω = G(copy(Ωs[i]), (0, τ))
            @test reldiff(cpu(X)[:, :, i], coefficients(Ω)) < 1e-12
            @test sqrt(OKF.normdiff(Ω, Ωs[i])) > 1e-2
        end
    end

    # ---- every batch size it was made for: two of the fields alone, one field repeated, and one
    #      alone ----
    X  = P.G(batch(Ωs), 2.345)
    Y  = dynamics(2).G(batch(Ωs[1:2]), 2.345)
    Z  = dynamics(5).G(batch([Ωs[3] for i in 1:5]), 2.345)
    X₁ = dynamics(1).G(batch(Ωs[4:4]), 2.345)
    @test reldiff(Y, cpu(X)[:, :, 1:2]) < 1e-13
    @test all(reldiff(cpu(Z)[:, :, j], cpu(X)[:, :, 3]) < 1e-13 for j in 1:5)
    @test reldiff(X₁, cpu(X)[:, :, 4:4]) < 1e-13

    # ---- a view of a batch ----
    Y = batch(Ωs)
    dynamics(2).G(view(Y, :, :, 1:2), 2.345)
    @test reldiff(cpu(Y)[:, :, 1:2], cpu(X)[:, :, 1:2]) < 1e-13
    @test cpu(Y)[:, :, 3:4] == cpu(batch(Ωs[3:4]))

    # ---- the result is a batch of fields, and what is not a field is projected out first ----
    @test cpu(KS.apply_symmetry!(copy(X))) == cpu(X)
    @test reldiff(P.G(plus(batch(Ωs), 1, junk(4)), 2.345), X) < 1e-13
end

@testset "the flow as a flow                      " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)
    P  = dynamics(3)
    P₁ = dynamics(1)

    # ---- a span of zero leaves the fields, and two spans one after the other make their sum ----
    @test cpu(P.G(copy(X), 0)) == cpu(X)
    Y = P.G(P.G(copy(X), 0.5), 0.734)
    @test reldiff(Y, P.G(copy(X), 1.234)) < 1e-12

    # ---- its steps one by one: a span up to dt is one step of its length, so that the steps of
    #      the flow taken one at a time give its result exactly ----
    steps = [Δt for (t, Δt) in Flows.Steps(0.0, 1.234, dt)]
    @test length(steps) == 124 && all(steps[1:(end - 1)] .== dt) && steps[end] ≈ 0.004
    Z = copy(X)
    for Δt in steps
        P.G(Z, Δt)
    end
    @test cpu(Z) == cpu(P.G(copy(X), 1.234))

    # ---- the laminar flow is a fixed point ----
    Xₗ = batch([OKF.laminarflow(n, m, Re, kf)])
    @test maximum(abs, cpu(P₁.F(similar(Xₗ), Xₗ))) < 1e-14 * maximum(abs, cpu(Xₗ))
    @test reldiff(P₁.G(copy(Xₗ), 1.0), Xₗ) < 1e-14

    # ---- the right-hand side is the derivative of the flow in time: a short step against it, with
    #      an error that halves with the step ----
    R      = P.F(similar(X), X)
    errors = map((1e-3, 5e-4)) do ε
        reldiff(plus(P.G(copy(X), ε), -1, X) ./ ε, R)
    end
    @test errors[1] < 1e-2
    @test 0.45 < errors[2]/errors[1] < 0.55
end

@testset "symmetries of the flow                 " begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    X  = batch(Ωs)
    V  = batch(Λs)
    P  = dynamics(3)
    τ  = 1.234
    s  = 0.731

    # ---- the flow, its tangent and the right hand side commute with the shift along x ----
    Y = P.G(copy(X), τ)
    @test reldiff(P.G(XShift()(copy(X), s), τ), XShift()(copy(Y), s)) < 1e-12
    W = P.L(copy(X), copy(V), τ)
    @test reldiff(P.L(XShift()(copy(X), s), XShift()(copy(V), s), τ), XShift()(copy(W), s)) < 1e-12
    R = P.F(similar(X), X)
    @test reldiff(P.F(similar(X), XShift()(copy(X), s)), XShift()(copy(R), s)) < 1e-13

    # ---- with the rotation by π ----
    @test reldiff(P.G(rotate(X), τ), rotate(Y)) < 1e-12
    @test reldiff(P.F(similar(X), rotate(X)), rotate(R)) < 1e-13

    # ---- and with the shift-reflect, which the forcing keeps ----
    @test reldiff(P.G(shiftreflect(X), τ), shiftreflect(Y)) < 1e-12
    @test reldiff(P.F(similar(X), shiftreflect(X)), shiftreflect(R)) < 1e-13
end

@testset "the tangent, field by field as OKF + Flows" begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    P  = dynamics(3)
    P₁ = dynamics(1)

    # ---- the coupled tangent equation of OKF ----
    τ = 1.234
    X = batch(Ωs)
    V = batch(Λs)
    P.L(X, V, τ)
    for i in 1:3
        Ω = copy(Ωs[i])
        Λ = copy(Λs[i])
        L(couple(Ω, Λ), (0, τ))
        @test reldiff(cpu(X)[:, :, i], coefficients(Ω)) < 1e-12
        @test reldiff(cpu(V)[:, :, i], coefficients(Λ)) < 1e-12
    end

    # ---- the fields go as the flow takes them ----
    @test cpu(X) == cpu(P.G(batch(Ωs), τ))

    # ---- the derivative of the flow: a centred difference of the flow ----
    h  = 1e-5
    Xp = P.G(plus(batch(Ωs), h, batch(Λs)), τ)
    Xm = P.G(plus(batch(Ωs), -h, batch(Λs)), τ)
    @test reldiff((Xp .- Xm) ./ (2h), V) < 1e-7

    # ---- whose error falls as the square of the increment ----
    errors = map((4e-3, 2e-3)) do h
        Xp = P.G(plus(batch(Ωs), h, batch(Λs)), τ)
        Xm = P.G(plus(batch(Ωs), -h, batch(Λs)), τ)
        reldiff((Xp .- Xm) ./ (2h), V)
    end
    @test 3.8 < errors[1]/errors[2] < 4.2

    # ---- linear in the perturbations ----
    U  = random_fields(3)
    TU = P.L(batch(Ωs), copy(U), τ)
    TV = P.L(batch(Ωs), plus(batch(Λs), 2.5, U), τ)
    @test reldiff(TV, plus(V, 2.5, TU)) < 1e-13

    # ---- a batch of one, and a view ----
    V₁ = P₁.L(batch(Ωs[2:2]), batch(Λs[2:2]), τ)
    @test reldiff(V₁, cpu(V)[:, :, 2:2]) < 1e-13
    Y = batch(Ωs)
    W = batch(Λs)
    P₁.L(view(Y, :, :, 3:3), view(W, :, :, 3:3), τ)
    @test reldiff(cpu(W)[:, :, 3:3], cpu(V)[:, :, 3:3]) < 1e-13
    @test cpu(W)[:, :, 1:2] == cpu(batch(Λs[1:2]))

    # ---- the tangent vanishes on what is not a field ----
    J = junk(3; seed=4)
    P.L(batch(Ωs), J, τ)
    @test maximum(abs, cpu(J)) == 0
end

@testset "the other schemes of Flows             " begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    τ  = 0.734

    for name in (:CNRK2, :CB3R2R2, :CB3R2R3c, :CB4R3R4)
        P = dynamics(3; scheme=name)

        # ---- the same scheme of Flows on the fields of OKF, alone and with the tangent ----
        scheme = getfield(Flows, name)
        Gs     = flow(ex, im_, scheme(OKF.FTField(n, m)), TimeStepConstant(dt))
        Ls     = flow(couple(ex, lex), couple(im_, lim),
                      scheme(couple(OKF.FTField(n, m), OKF.FTField(n, m))), TimeStepConstant(dt))

        # ---- the flow and the tangent, field by field ----
        X = P.G(batch(Ωs), τ)
        Y = batch(Ωs)
        V = P.L(Y, batch(Λs), τ)
        for i in 1:3
            Ω = copy(Ωs[i])
            Λ = copy(Λs[i])
            @test reldiff(cpu(X)[:, :, i], coefficients(Gs(copy(Ω), (0, τ)))) < 1e-12
            Ls(couple(Ω, Λ), (0, τ))
            @test reldiff(cpu(Y)[:, :, i], coefficients(Ω)) < 1e-12
            @test reldiff(cpu(V)[:, :, i], coefficients(Λ)) < 1e-12
        end
    end

    # ---- the adjoint is the transpose of the tangent with every scheme ----
    X = batch(Ωs)
    V = random_fields(3; seed=5)
    W = random_fields(3; seed=6)
    for name in (:CNRK2, :CB3R2R2, :CB3R2R3c, :CB4R3R4)
        P  = dynamics(3; scheme=name)
        ψV = P.L(copy(X), copy(V), τ)
        ψW = P.A(copy(W), stages(P.G, X, τ))
        @test all(relerr(products(ψV, W), products(V, ψW)) .< 1e-10)
    end

    # ---- a name that is not a scheme of Flows, nor one of a split system ----
    @test_throws ArgumentError dynamics(3; scheme=:CB9)
    @test_throws ArgumentError dynamics(3; scheme=:RK4)
end

@testset "the exponential schemes of Flows        " begin
    X = batch(okf_fields(3))
    V = random_fields(3; seed=5)
    τ = 1.0
    G(name, h) = NonlinearPropagator(n, m, Re, kf; B=3, dt=h, scheme=name, device=DEVICE)

    # ---- of order two and four, the viscous term exact: the error against a flow of short steps
    #      falls four and sixteen times when the step is halved ----
    R = G(:CB4R3R4, 1e-4)(copy(X), τ)
    for (name, ratio) in ((:ETDRK2, 4), (:ETDRK4, 16))
        e = [reldiff(G(name, h)(copy(X), τ), R) for h in (0.02, 0.01)]
        @test 0.9ratio < e[1]/e[2] < 1.1ratio
    end

    # ---- the tangent is the derivative of the discrete flow, a centred difference of the flow ----
    for name in (:ETDRK2, :ETDRK4)
        L = TangentPropagator(n, m, Re, kf; B=3, dt=dt, scheme=name, device=DEVICE)
        h = 1e-5
        D = (G(name, dt)(plus(X, h, V), τ) .- G(name, dt)(plus(X, -h, V), τ)) ./ (2h)
        @test reldiff(D, L(copy(X), copy(V), τ)) < 1e-7
    end

    # ---- the adjoint, over the stages of the flow, is the transpose of the tangent ----
    W = random_fields(3; seed=6)
    for name in (:ETDRK2, :ETDRK4)
        L  = TangentPropagator(n, m, Re, kf; B=3, dt=dt, scheme=name, device=DEVICE)
        A  = AdjointPropagator(n, m, Re; B=3, scheme=name, device=DEVICE)
        ψV = L(copy(X), copy(V), τ)
        ψW = A(copy(W), stages(G(name, dt), X, τ))
        @test all(relerr(products(ψV, W), products(V, ψW)) .< 1e-10)
    end
end

@testset "the tangent, batches about the same fields" begin
    Ωs = okf_fields(3)
    P  = dynamics(3; k′=2)
    P₁ = dynamics(3)
    τ  = 1.234

    # ---- two batches of three perturbations, each about the three fields ----
    V = random_fields(6)
    Y = batch(Ωs)
    W = copy(V)
    P.L(Y, directions(Y, W), τ)

    # ---- batch by batch as the tangent of the batch alone ----
    for k₁ in 1:2
        block = (3k₁ - 2):3k₁
        V₁    = dev(FTFields(cpu(V)[:, :, block]))
        @test reldiff(cpu(W)[:, :, block], P₁.L(batch(Ωs), V₁, τ)) < 1e-13
    end

    # ---- the fields advanced once, as the flow takes them ----
    @test cpu(Y) == cpu(P.G(batch(Ωs), τ))

    # ---- a batch of perturbations with another number of fields is refused ----
    @test_throws ArgumentError P₁.L(batch(Ωs), random_fields(4), τ)
    @test_throws ArgumentError P.L(batch(Ωs), [random_fields(3), random_fields(4)], τ)
end

@testset "another Reynolds number, the forced tangent" begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    P  = dynamics(3)
    X  = P.G(batch(Ωs), 1.234)

    # ---- the propagator at another Reynolds number shares the workspace, gives the results of one
    #      made there, and leaves the first as it was ----
    Q = at(P, 35.0)
    R = dynamics(3; reynolds=35.0)
    @test Q.G.whole.X === P.G.whole.X && Q.L.chunks[1].Z === P.L.chunks[1].Z
    @test Q.G.Re == 35.0 && P.G.Re == Re
    @test cpu(Q.G(batch(Ωs), 1.234)) == cpu(R.G(batch(Ωs), 1.234))
    @test cpu(Q.L(batch(Ωs), batch(Λs), 1.234)) == cpu(R.L(batch(Ωs), batch(Λs), 1.234))
    @test cpu(Q.Lp(batch(Ωs), batch(Λs), 1.234)) == cpu(R.Lp(batch(Ωs), batch(Λs), 1.234))
    @test cpu(Q.F(similar(X), X)) == cpu(R.F(similar(X), X))
    @test cpu(P.G(batch(Ωs), 1.234)) == cpu(X)
    @test reldiff(Q.G(batch(Ωs), 1.234), X) > 1e-3

    # ---- it costs no workspace ----
    Q.G(batch(Ωs), 2dt)
    @test (@allocated TangentPropagator(P.L, 36.0)) < 1024

    # ---- the forced tangent from zero against the centred difference of the flow in the Reynolds
    #      number, up to the error of the scheme, which falls with the step ----
    errors = map((0.01, 0.005)) do h
        A  = dynamics(3; dt=h)
        V  = A.Lp(batch(Ωs), fill!(similar(X), 0), 1.234)
        ε  = 1e-4
        Xp = at(A, Re + ε).G(batch(Ωs), 1.234)
        Xm = at(A, Re - ε).G(batch(Ωs), 1.234)
        reldiff(V, (Xp .- Xm) ./ (2ε))
    end
    @test errors[1] < 1e-4
    @test errors[2] < errors[1]/2

    # ---- and it is the tangent plus that: from V it adds the tangent of V ----
    V₀ = P.Lp(batch(Ωs), fill!(similar(X), 0), 1.234)
    V₁ = P.Lp(batch(Ωs), batch(Λs), 1.234)
    @test reldiff(V₁ .- V₀, P.L(batch(Ωs), batch(Λs), 1.234)) < 1e-12

    # ---- the fields go as the flow takes them ----
    Y = batch(Ωs)
    P.Lp(Y, batch(Λs), 1.234)
    @test cpu(Y) == cpu(X)
end

@testset "shift and its generator                " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)
    S  = XShift()

    # ---- the shift along x is OKF's xshift! ----
    s  = 0.731
    Xs = S(copy(X), s)
    for i in 1:3
        @test coefficients(OKF.xshift!(copy(Ωs[i]), s)) ≈ cpu(Xs)[:, :, i] rtol=1e-15
    end

    # ---- a view of the last field ----
    Y = copy(X)
    S(view(Y, :, :, 3:3), s)
    @test cpu(Y)[:, :, 1:2] == cpu(X)[:, :, 1:2]
    @test cpu(Y)[:, :, 3] == cpu(Xs)[:, :, 3]

    # ---- a group of isometries: two shifts are their sum, 2π is none, and the inner product stays ----
    @test reldiff(S(S(copy(X), s), 1.2), S(copy(X), s + 1.2)) < 1e-14
    @test reldiff(S(copy(X), 2π), X) < 1e-14
    @test products(Xs, Xs) ≈ products(X, X) rtol=1e-14

    # ---- the generator is the derivative along x, the derivative of the shift, and antisymmetric ----
    D = S(similar(X), X, 1)
    @test cpu(D) == cpu(KS.ddx!(similar(X), X))
    @test_throws ArgumentError S(similar(X), X, 2)
    @test reldiff((S(copy(X), 1e-6) .- S(copy(X), -1e-6)) ./ 2e-6, D) < 1e-9
    @test all(abs.(products(D, X)) .< 1e-14 .* products(X, X))
end

@testset "workspaces, made with the propagator   " begin
    X   = random_fields(256)
    V   = random_fields(256; seed=8)
    W   = random_fields(256; seed=9)
    F   = similar(X)
    one = sizeof(parent(X))
    P   = dynamics(256)

    # ---- a call allocates much less than one batch, over three steps: a few kB for each
    #      call and each step, whatever the size of the batch ----
    @test allocated(() -> P.F(F, X)) < one/10
    @test allocated(() -> P.G(X, 3dt)) < one/10
    @test allocated(() -> P.L(X, V, 3dt)) < one/10
    @test allocated(() -> P.Lp(X, V, 3dt)) < one/10

    # ---- stages allocates the four stages of each step, which it returns, and little more, and
    #      the adjoint over them much less than one batch ----
    S = stages(P.G, X, 3dt)
    P.A(W, S)
    @test allocated(() -> stages(P.G, X, 3dt)) < 4*3*allocated(() -> copy(X)) + one/10
    @test allocated(() -> P.A(W, S)) < one/10

    # ---- with the exponential schemes too, over a span with a shorter last step, at which the
    #      coefficients are made again, mode by mode, into the arrays of the viscous term, with no
    #      array made: the flow, the tangent, the stages and the adjoint ----
    T = KS.ImplicitTerm(X, Re)
    for (name, coeffs) in ((:ETDRK2, Flows.etdrk2_coeffs), (:ETDRK4, Flows.etdrk4_coeffs))
        E = dynamics(256; scheme=name)
        @test all(c -> any(c === a for a in T.coefficients), coeffs(T, dt))
        @test allocated(() -> coeffs(T, dt)) == 0
        @test allocated(() -> E.G(X, 2.5dt)) < one/10
        @test allocated(() -> E.L(X, V, 2.5dt)) < one/10
        S = stages(E.G, X, 2.5dt)
        E.A(W, S)
        @test allocated(() -> E.A(W, S)) < one/10
    end
end

# several tasks advance a batch on the CPU, a chunk of its fields each: the results are those of one task
DEVICE === :cpu && @testset "a batch cut among tasks                " begin
    Ωs = okf_fields(5)
    τ  = 0.734
    for tasks in (2, 3)
        # ---- batches of more fields than tasks, of as many, and of fewer, with one perturbation
        #      and with two, on the tasks and on one ----
        for B in (5, tasks, 1)
            X  = batch(Ωs[1:B])
            V  = random_fields(B; seed=4)
            W  = random_fields(2B; seed=5)
            P  = dynamics(B; tasks=tasks)
            P₁ = dynamics(B)
            Q  = dynamics(B; k′=2, tasks=tasks)
            Q₁ = dynamics(B; k′=2)

            @test reldiff(P.G(copy(X), τ), P₁.G(copy(X), τ)) < 1e-13
            @test reldiff(P.L(copy(X), copy(V), τ), P₁.L(copy(X), copy(V), τ)) < 1e-13
            W₁ = copy(W)
            W₂ = copy(W)
            Q.L(copy(X), directions(X, W₁), τ)
            Q₁.L(copy(X), directions(X, W₂), τ)
            @test reldiff(W₁, W₂) < 1e-13
            @test reldiff(P.Lp(copy(X), copy(V), τ), P₁.Lp(copy(X), copy(V), τ)) < 1e-13

            # ---- the fields are advanced with their perturbations, as by one task ----
            Y = copy(X)
            Q.L(Y, directions(Y, copy(W)), τ)
            @test reldiff(Y, P₁.G(copy(X), τ)) < 1e-13
        end

        # ---- every task has its chunk, the same for the fields and for their perturbations: of 5
        #      fields, 2 and 3, or 1, 2 and 2 ----
        P = dynamics(5; tasks=tasks)
        @test [chunk.fields for chunk in P.L.chunks] == (tasks == 2 ? [1:2, 3:5] : [1:1, 2:3, 4:5])
        @test [chunk.fields for chunk in P.G.chunks] == [chunk.fields for chunk in P.L.chunks]

        # ---- the other methods run on one task, as those of one task ----
        P₁ = dynamics(5)
        X  = batch(Ωs)
        @test cpu(P.F(similar(X), X)) == cpu(P₁.F(similar(X), X))
        W = random_fields(5; seed=6)
        @test reldiff(P.A(copy(W), stages(P.G, X, τ)), P₁.A(copy(W), stages(P₁.G, X, τ))) < 1e-13
    end
end
