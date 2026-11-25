# The propagator: its constructor, propagate!, tangent!, forced!, rhs!, shift! and dshift!, the
# properties of the flow, and its storage.

# the rotation by π, ω(x, y) → ω(-x, -y): the coefficients conjugated
rotate(X) = FTFields(conj.(X))

# the shift-reflect, ω(x, y) → -ω(-x, y + π/kf): Ω[j, k] → -conj(Ω[j, -k]) e^{i k π/kf}
function shiftreflect(X)
    k = [0:n; -n:-1]
    c = mod.(.-k, 2n + 1) .+ 1
    B = .-conj.(cpu(X)[:, c, :]) .* cis.(k' .* π ./ kf)

    return dev(FTFields(B))
end

@testset "the propagator                         " begin
    P = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)

    # ---- what it holds, with no storage until the first call ----
    @test (P.n, P.m, P.Re, P.kf, P.dt) == (n, m, Re, kf, dt)
    @test P.template isa ARRAY{ComplexF64, 3}
    @test all(isempty, P.cache)
    @test length(P.cache) == length(P.tangents) == 1
    @test length(KolmogorovPropagator(n, m, Re, kf; tasks=3).cache) == 3
    @test KolmogorovPropagator(n, m, 40, kf).Re === 40.0
    @test KolmogorovPropagator(n, m, Re, kf).dt == 0.01

    # ---- checks: a grid that aliases the products, and a forcing outside the modes ----
    @test_throws ArgumentError KolmogorovPropagator(8, 11, Re, kf)
    @test KolmogorovPropagator(8, 12, Re, kf) isa KolmogorovPropagator
    @test_throws ArgumentError KolmogorovPropagator(n, m, Re, 9)
    @test_throws ArgumentError KolmogorovPropagator(n, m, Re, 0)
end

@testset "propagate!, field by field as OKF + Flows" begin
    Ωs = okf_fields(4)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)

    # ---- a span that is a whole number of steps, and one that is not, as the segments of a search ----
    for τ in (1.0, 2.345)
        X = propagate!(P, batch(Ωs), τ)
        for i in 1:4
            Ω = G(copy(Ωs[i]), (0, τ))
            @test reldiff(cpu(X)[:, :, i], coefficients(Ω)) < 1e-12
            @test sqrt(OKF.normdiff(Ω, Ωs[i])) > 1e-2
        end
    end

    # ---- any batch size: two of the fields alone, one field repeated, and one alone ----
    X  = propagate!(P, batch(Ωs), 2.345)
    Y  = propagate!(P, batch(Ωs[1:2]), 2.345)
    Z  = propagate!(P, batch([Ωs[3] for i in 1:5]), 2.345)
    X₁ = propagate!(P, batch(Ωs[4:4]), 2.345)
    @test reldiff(Y, cpu(X)[:, :, 1:2]) < 1e-13
    @test all(reldiff(cpu(Z)[:, :, j], cpu(X)[:, :, 3]) < 1e-13 for j in 1:5)
    @test reldiff(X₁, cpu(X)[:, :, 4:4]) < 1e-13
    @test sort(collect(keys(P.cache[1]))) == [1, 2, 4, 5]

    # ---- a view of a batch ----
    Y = batch(Ωs)
    propagate!(P, view(Y, :, :, 1:2), 2.345)
    @test reldiff(cpu(Y)[:, :, 1:2], cpu(X)[:, :, 1:2]) < 1e-13
    @test cpu(Y)[:, :, 3:4] == cpu(batch(Ωs[3:4]))

    # ---- the result is a batch of fields, and what is not a field is projected out first ----
    @test cpu(KS.apply_symmetry!(copy(X))) == cpu(X)
    @test reldiff(propagate!(P, plus(batch(Ωs), 1, junk(4)), 2.345), X) < 1e-13
end

@testset "propagate! as a flow                   " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)

    # ---- a span of zero leaves the fields, and two spans one after the other make their sum ----
    @test cpu(propagate!(P, copy(X), 0)) == cpu(X)
    Y = propagate!(P, propagate!(P, copy(X), 0.5), 0.734)
    @test reldiff(Y, propagate!(P, copy(X), 1.234)) < 1e-12

    # ---- its steps one by one: a span up to dt is one step of its length, so that the steps of
    #      propagate! taken one at a time give its result exactly ----
    steps = [Δt for (t, Δt) in Flows.Steps(0.0, 1.234, dt)]
    @test length(steps) == 124 && all(steps[1:(end - 1)] .== dt) && steps[end] ≈ 0.004
    Z = copy(X)
    for Δt in steps
        propagate!(P, Z, Δt)
    end
    @test cpu(Z) == cpu(propagate!(P, copy(X), 1.234))

    # ---- the laminar flow is a fixed point ----
    Xₗ = batch([OKF.laminarflow(n, m, Re, kf)])
    @test maximum(abs, cpu(rhs!(P, similar(Xₗ), Xₗ))) < 1e-14 * maximum(abs, cpu(Xₗ))
    @test reldiff(propagate!(P, copy(Xₗ), 1.0), Xₗ) < 1e-14

    # ---- rhs! is the derivative of the flow in time: a short step against it, with an error that
    #      halves with the step ----
    R      = rhs!(P, similar(X), X)
    errors = map((1e-3, 5e-4)) do ε
        reldiff(plus(propagate!(P, copy(X), ε), -1, X) ./ ε, R)
    end
    @test errors[1] < 1e-2
    @test 0.45 < errors[2]/errors[1] < 0.55
end

@testset "symmetries of the flow                 " begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    X  = batch(Ωs)
    V  = batch(Λs)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)
    τ  = 1.234
    s  = 0.731

    # ---- the flow, its tangent and the right hand side commute with the shift along x ----
    Y = propagate!(P, copy(X), τ)
    @test reldiff(propagate!(P, shift!(P, copy(X), s), τ), shift!(P, copy(Y), s)) < 1e-12
    W = tangent!(P, copy(X), copy(V), τ)
    @test reldiff(tangent!(P, shift!(P, copy(X), s), shift!(P, copy(V), s), τ), shift!(P, copy(W), s)) < 1e-12
    R = rhs!(P, similar(X), X)
    @test reldiff(rhs!(P, similar(X), shift!(P, copy(X), s)), shift!(P, copy(R), s)) < 1e-13

    # ---- with the rotation by π ----
    @test reldiff(propagate!(P, rotate(X), τ), rotate(Y)) < 1e-12
    @test reldiff(rhs!(P, similar(X), rotate(X)), rotate(R)) < 1e-13

    # ---- and with the shift-reflect, which the forcing keeps ----
    @test reldiff(propagate!(P, shiftreflect(X), τ), shiftreflect(Y)) < 1e-12
    @test reldiff(rhs!(P, similar(X), shiftreflect(X)), shiftreflect(R)) < 1e-13
end

@testset "tangent!, field by field as OKF + Flows " begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)

    # ---- the coupled tangent equation of OKF ----
    τ = 1.234
    X = batch(Ωs)
    V = batch(Λs)
    tangent!(P, X, V, τ)
    for i in 1:3
        Ω = copy(Ωs[i])
        Λ = copy(Λs[i])
        L(couple(Ω, Λ), (0, τ))
        @test reldiff(cpu(X)[:, :, i], coefficients(Ω)) < 1e-12
        @test reldiff(cpu(V)[:, :, i], coefficients(Λ)) < 1e-12
    end

    # ---- the fields go as propagate! takes them ----
    @test cpu(X) == cpu(propagate!(P, batch(Ωs), τ))

    # ---- the derivative of the flow: a centred difference of propagate! ----
    h  = 1e-5
    Xp = propagate!(P, plus(batch(Ωs), h, batch(Λs)), τ)
    Xm = propagate!(P, plus(batch(Ωs), -h, batch(Λs)), τ)
    @test reldiff((Xp .- Xm) ./ (2h), V) < 1e-7

    # ---- whose error falls as the square of the increment ----
    errors = map((4e-3, 2e-3)) do h
        Xp = propagate!(P, plus(batch(Ωs), h, batch(Λs)), τ)
        Xm = propagate!(P, plus(batch(Ωs), -h, batch(Λs)), τ)
        reldiff((Xp .- Xm) ./ (2h), V)
    end
    @test 3.8 < errors[1]/errors[2] < 4.2

    # ---- linear in the perturbations ----
    U  = random_fields(3)
    TU = tangent!(P, batch(Ωs), copy(U), τ)
    TV = tangent!(P, batch(Ωs), plus(batch(Λs), 2.5, U), τ)
    @test reldiff(TV, plus(V, 2.5, TU)) < 1e-13

    # ---- a batch of one, and a view ----
    V₁ = tangent!(P, batch(Ωs[2:2]), batch(Λs[2:2]), τ)
    @test reldiff(V₁, cpu(V)[:, :, 2:2]) < 1e-13
    Y = batch(Ωs)
    W = batch(Λs)
    tangent!(P, view(Y, :, :, 3:3), view(W, :, :, 3:3), τ)
    @test reldiff(cpu(W)[:, :, 3:3], cpu(V)[:, :, 3:3]) < 1e-13
    @test cpu(W)[:, :, 1:2] == cpu(batch(Λs[1:2]))

    # ---- the tangent vanishes on what is not a field ----
    J = junk(3; seed=4)
    tangent!(P, batch(Ωs), J, τ)
    @test maximum(abs, cpu(J)) == 0
end

@testset "the other schemes of Flows             " begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    τ  = 0.734

    for name in (:CNRK2, :CB3R2R2, :CB3R2R3c, :CB4R3R4)
        P = KolmogorovPropagator(n, m, Re, kf; dt=dt, scheme=name, array=ARRAY)

        # ---- the same scheme of Flows on the fields of OKF, alone and with the tangent ----
        scheme = getfield(Flows, name)
        Gs     = flow(ex, im_, scheme(OKF.FTField(n, m)), TimeStepConstant(dt))
        Ls     = flow(couple(ex, lex), couple(im_, lim),
                      scheme(couple(OKF.FTField(n, m), OKF.FTField(n, m))), TimeStepConstant(dt))

        # ---- propagate! and tangent!, field by field ----
        X = propagate!(P, batch(Ωs), τ)
        Y = batch(Ωs)
        V = tangent!(P, Y, batch(Λs), τ)
        for i in 1:3
            Ω = copy(Ωs[i])
            Λ = copy(Λs[i])
            @test reldiff(cpu(X)[:, :, i], coefficients(Gs(copy(Ω), (0, τ)))) < 1e-12
            Ls(couple(Ω, Λ), (0, τ))
            @test reldiff(cpu(Y)[:, :, i], coefficients(Ω)) < 1e-12
            @test reldiff(cpu(V)[:, :, i], coefficients(Λ)) < 1e-12
        end
    end

    # ---- the adjoint is the transpose of the tangent with the schemes that have one ----
    X = batch(Ωs)
    V = random_fields(3; seed=5)
    W = random_fields(3; seed=6)
    for name in (:CNRK2, :CB3R2R2, :CB3R2R3c)
        P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, scheme=name, array=ARRAY)
        ψV = tangent!(P, copy(X), copy(V), τ)
        ψW = adjoint!(P, stages(P, X, τ), copy(W))
        @test all(relerr(products(ψV, W), products(V, ψW)) .< 1e-10)
    end

    # ---- a name that is not a scheme, and the adjoint of the scheme that has none ----
    @test_throws ArgumentError KolmogorovPropagator(n, m, Re, kf; scheme=:RK4)
    P = KolmogorovPropagator(n, m, Re, kf; dt=dt, scheme=:CB4R3R4, array=ARRAY)
    @test_throws ArgumentError adjoint!(P, stages(P, X, τ), copy(W))
end

@testset "tangent!, blocks about the same fields" begin
    Ωs = okf_fields(3)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)
    τ  = 1.234

    # ---- two blocks of three perturbations, each about the three fields ----
    V = random_fields(6)
    Y = batch(Ωs)
    W = tangent!(P, Y, copy(V), τ)

    # ---- block by block as tangent! on the block alone ----
    for a in 1:2
        block = (3a - 2):3a
        Vₐ    = dev(FTFields(cpu(V)[:, :, block]))
        @test reldiff(cpu(W)[:, :, block], tangent!(P, batch(Ωs), Vₐ, τ)) < 1e-13
    end

    # ---- the fields advanced once, as propagate! takes them ----
    @test cpu(Y) == cpu(propagate!(P, batch(Ωs), τ))

    # ---- one storage for the two sizes, built once and looked up after ----
    @test haskey(P.tangents[1], (3, 6))
    @test KS._tangent_storage(P, 3, 6) === KS._tangent_storage(P, 3, 6)

    # ---- perturbations that are not whole blocks of the fields are refused ----
    @test_throws ArgumentError tangent!(P, batch(Ωs), random_fields(4), τ)
end

@testset "another Reynolds number, and forced!   " begin
    Ωs = okf_fields(3)
    Λs = okf_fields(3; seed=2, laminar=false)
    P  = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)
    X  = propagate!(P, batch(Ωs), 1.234)

    # ---- the propagator at another Reynolds number shares the storage, gives the results of one
    #      built there, and leaves the first as it was ----
    Q = KolmogorovPropagator(P, 35.0)
    R = KolmogorovPropagator(n, m, 35.0, kf; dt=dt, array=ARRAY)
    @test typeof(Q) === typeof(P)
    @test Q.cache === P.cache
    @test Q.Re == 35.0 && P.Re == Re
    @test cpu(propagate!(Q, batch(Ωs), 1.234)) == cpu(propagate!(R, batch(Ωs), 1.234))
    @test cpu(tangent!(Q, batch(Ωs), batch(Λs), 1.234)) == cpu(tangent!(R, batch(Ωs), batch(Λs), 1.234))
    @test cpu(forced!(Q, batch(Ωs), batch(Λs), 1.234)) == cpu(forced!(R, batch(Ωs), batch(Λs), 1.234))
    @test cpu(rhs!(Q, similar(X), X)) == cpu(rhs!(R, similar(X), X))
    @test cpu(propagate!(P, batch(Ωs), 1.234)) == cpu(X)
    @test reldiff(propagate!(Q, batch(Ωs), 1.234), X) > 1e-3

    # ---- it costs no plan and no array ----
    propagate!(Q, batch(Ωs), 2dt)
    @test (@allocated KolmogorovPropagator(P, 36.0)) < 256

    # ---- forced! from zero against the centred difference of propagate! in the Reynolds number,
    #      up to the error of the scheme, which falls with the step ----
    errors = map((0.01, 0.005)) do h
        A  = KolmogorovPropagator(n, m, Re, kf; dt=h, array=ARRAY)
        V  = forced!(A, batch(Ωs), fill!(similar(X), 0), 1.234)
        ε  = 1e-4
        Xp = propagate!(KolmogorovPropagator(A, Re + ε), batch(Ωs), 1.234)
        Xm = propagate!(KolmogorovPropagator(A, Re - ε), batch(Ωs), 1.234)
        reldiff(V, (Xp .- Xm) ./ (2ε))
    end
    @test errors[1] < 1e-4
    @test errors[2] < errors[1]/2

    # ---- and it is the tangent plus that: from V it adds the tangent of V ----
    V₀ = forced!(P, batch(Ωs), fill!(similar(X), 0), 1.234)
    V₁ = forced!(P, batch(Ωs), batch(Λs), 1.234)
    @test reldiff(V₁ .- V₀, tangent!(P, batch(Ωs), batch(Λs), 1.234)) < 1e-12

    # ---- the fields go as propagate! takes them ----
    Y = batch(Ωs)
    forced!(P, Y, batch(Λs), 1.234)
    @test cpu(Y) == cpu(X)
end

@testset "shift and its generator                " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)
    P  = KolmogorovPropagator(n, m, Re, kf; array=ARRAY)

    # ---- the shift along x is OKF's xshift!, for a number and for the vector of the shifts ----
    s  = 0.731
    Xs = shift!(P, copy(X), s)
    for i in 1:3
        @test coefficients(OKF.xshift!(copy(Ωs[i]), s)) ≈ cpu(Xs)[:, :, i] rtol=1e-15
    end
    @test cpu(shift!(P, copy(X), view([1.5, s], 2:2))) == cpu(Xs)

    # ---- a view of the last field ----
    Y = copy(X)
    shift!(P, view(Y, :, :, 3:3), s)
    @test cpu(Y)[:, :, 1:2] == cpu(X)[:, :, 1:2]
    @test cpu(Y)[:, :, 3] == cpu(Xs)[:, :, 3]

    # ---- a group of isometries: two shifts are their sum, 2π is none, and the inner product stays ----
    @test reldiff(shift!(P, shift!(P, copy(X), s), 1.2), shift!(P, copy(X), s + 1.2)) < 1e-14
    @test reldiff(shift!(P, copy(X), 2π), X) < 1e-14
    @test products(Xs, Xs) ≈ products(X, X) rtol=1e-14

    # ---- the generator is the derivative along x, the derivative of the shift, and antisymmetric ----
    D = dshift!(P, similar(X), X, 1)
    @test cpu(D) == cpu(KS.ddx!(similar(X), X))
    @test cpu(dshift!(P, similar(X), X)) == cpu(D)
    @test reldiff((shift!(P, copy(X), 1e-6) .- shift!(P, copy(X), -1e-6)) ./ 2e-6, D) < 1e-9
    @test all(abs.(products(D, X)) .< 1e-14 .* products(X, X))
end

@testset "storage, built once and looked up      " begin
    X   = random_fields(256)
    V   = random_fields(256; seed=8)
    W   = random_fields(256; seed=9)
    F   = similar(X)
    one = sizeof(parent(X))
    P   = KolmogorovPropagator(n, m, Re, kf; dt=dt, array=ARRAY)

    # ---- built at the first call with 256 fields, and only looked up after ----
    @test all(isempty, P.cache)
    rhs!(P, F, X)
    @test collect(keys(P.cache[1])) == [256]
    @test KS._storage(P, 256) === KS._storage(P, 256)
    @test allocated(() -> KS._storage(P, 256)) == 0

    # ---- so that a later call allocates much less than one batch, over three steps: a few kB for
    #      each call and each step, whatever the size of the batch ----
    @test allocated(() -> rhs!(P, F, X)) < one/10
    @test allocated(() -> propagate!(P, X, 3dt)) < one/10
    @test allocated(() -> tangent!(P, X, V, 3dt)) < one/10
    @test allocated(() -> forced!(P, X, V, 3dt)) < one/10

    # ---- stages allocates the four stages of each step, which it returns, and little more, and
    #      adjoint! over them much less than one batch ----
    S = stages(P, X, 3dt)
    adjoint!(P, S, W)
    @test allocated(() -> stages(P, X, 3dt)) < 4*3*allocated(() -> copy(X)) + one/10
    @test allocated(() -> adjoint!(P, S, W)) < one/10

    # ---- one storage for each batch size, shared at another Reynolds number ----
    X₂ = view(X, :, :, 1:2)
    rhs!(KolmogorovPropagator(P, 35.0), similar(X₂), X₂)
    @test sort(collect(keys(P.cache[1]))) == [2, 256]
end

# several tasks advance a batch on the CPU, a chunk of its fields each: the results are those of one task
DEVICE == "cpu" && @testset "a batch cut among tasks                " begin
    Ωs = okf_fields(5)
    τ  = 0.734
    P₁ = KolmogorovPropagator(n, m, Re, kf; dt=dt)

    for tasks in (2, 3)
        P = KolmogorovPropagator(n, m, Re, kf; dt=dt, tasks=tasks)

        # ---- batches of more fields than tasks, of as many, and of fewer ----
        for B in (5, tasks, 1)
            X = batch(Ωs[1:B])
            V = random_fields(B; seed=4)
            W = random_fields(2B; seed=5)

            @test reldiff(propagate!(P, copy(X), τ), propagate!(P₁, copy(X), τ)) < 1e-13
            @test reldiff(tangent!(P, copy(X), copy(V), τ), tangent!(P₁, copy(X), copy(V), τ)) < 1e-13
            @test reldiff(tangent!(P, copy(X), copy(W), τ), tangent!(P₁, copy(X), copy(W), τ)) < 1e-13
            @test reldiff(forced!(P, copy(X), copy(V), τ), forced!(P₁, copy(X), copy(V), τ)) < 1e-13

            # ---- the fields are advanced with their perturbations, as by one task ----
            Y = copy(X)
            tangent!(P, Y, copy(W), τ)
            @test reldiff(Y, propagate!(P₁, copy(X), τ)) < 1e-13
        end

        # ---- every task keeps the storage of its chunk: of 5 fields, 2 and 3, or 1, 2 and 2 ----
        @test [sort(collect(keys(C))) for C in P.cache] ==
              (tasks == 2 ? [[1, 2], [1, 3]] : [[1], [1, 2], [1, 2]])

        # ---- the other methods run on one task, as those of one task ----
        X = batch(Ωs)
        @test cpu(rhs!(P, similar(X), X)) == cpu(rhs!(P₁, similar(X), X))
        W = random_fields(5; seed=6)
        @test reldiff(adjoint!(P, stages(P, X, τ), copy(W)), adjoint!(P₁, stages(P₁, X, τ), copy(W))) < 1e-13
    end
end
