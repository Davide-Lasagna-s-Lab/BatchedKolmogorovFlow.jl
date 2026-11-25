# Batches of fields: FTFields and Fields, the symmetry of fields and the inner product.

@testset "batches of fields                      " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)

    # ---- the layout: j down the rows, k across the columns in the order 0, …, n, -n, …, -1 ----
    @test size(X) == (n + 1, 2n + 1, 3)
    @test cpu(X)[3, 2, 1] == Ωs[1][OKF.WaveNumber(1, 2)]
    @test cpu(X)[3, 2n + 1, 2] == Ωs[2][OKF.WaveNumber(-1, 2)]
    @test X isa FTFields{n}
    @test parent(X) isa ARRAY
    @test vec(cpu(X.j)) == 0:n
    @test vec(cpu(X.k)) == [0:n; -n:-1]
    @test similar(X).j === X.j && copy(X).k === X.k && view(X, :, :, 1:2).w === X.w
    @test vec(cpu(X.w)) == [1; fill(2, n)]
    @test all(parent(OKF.FTField(X, i, m)) == parent(Ωs[i]) for i in 1:3)
    @test_throws ArgumentError FTFields(zeros(ComplexF64, 4, 8, 2))

    # ---- a batch of zeros on the device ----
    Z = FTFields(n, 2; device=DEVICE)
    @test Z isa FTFields{n} && size(Z) == (n + 1, 2n + 1, 2) && parent(Z) isa ARRAY
    @test all(iszero, cpu(Z))
    @test_throws ArgumentError FTFields(n, 2; device=:tpu)

    # ---- a batch like it, of more fields, a copy, a view of some fields, and a broadcast into it ----
    @test similar(X) isa FTFields{n}
    @test similar(X, ComplexF64, (n + 1, 2n + 1, 7)) isa FTFields{n}
    @test size(similar(X, ComplexF64, (n + 1, 2n + 1, 7))) == (n + 1, 2n + 1, 7)
    @test !(similar(X, ComplexF64, (n + 3, 2n + 1, 7)) isa FTFields)
    @test !(similar(X, Float64, (n + 1, 2n + 1, 7)) isa FTFields)
    @test cpu(copy(X)) == cpu(X) && parent(copy(X)) !== parent(X)
    @test view(X, :, :, 2:3) isa FTFields{n}
    @test cpu(view(X, :, :, 2:3)) == cpu(X)[:, :, 2:3]
    Z  = similar(X)
    Z .= 2 .* X
    @test Z isa FTFields{n} && cpu(Z) == 2 .* cpu(X)

    # ---- the operations that MultipleShooting makes on a batch besides broadcasting ----
    Y = similar(X)
    @test copyto!(Y, X) === Y && cpu(Y) == cpu(X)
    @test fill!(Y, 0) === Y && all(iszero, cpu(Y))
    @test reverse!(copyto!(Y, X); dims=3) === Y && cpu(Y) == cpu(X)[:, :, 3:-1:1]
    @test Random.randn!(Y) === Y && cpu(Y) != cpu(X)

    # ---- a batch on the grid ----
    u = Fields(X, m)
    @test u isa Fields{m} && size(u) == (2m + 2, 2m + 2, 3) && parent(u) isa ARRAY
    @test Fields(zeros(2m + 2, 2m + 2, 2)) isa Fields{m}
    @test_throws ArgumentError Fields(zeros(4, 6, 2))
    u .= 0.5                 # the values of a new batch are not set, and may be NaN
    v  = similar(u)
    v .= 2 .* u
    @test v isa Fields{m} && cpu(v) == 2 .* cpu(u)
end

@testset "symmetry of fields and inner product   " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)

    # ---- a field is left as it is, and anything else becomes one ----
    @test cpu(KS.apply_symmetry!(copy(X))) == cpu(X)
    Y = KS.apply_symmetry!(plus(X, 1, junk(3)))
    @test reldiff(Y, X) < 1e-15
    @test cpu(Y)[1, 2:(n + 1), :] == conj.(cpu(Y)[1, (2n + 1):-1:(n + 2), :])
    @test all(cpu(Y)[1, 1, :] .== 0)

    # ---- a projection, orthogonal in the inner product: the identity twice, and its own transpose
    #      on batches that are not fields, which makes the adjoint the transpose of the tangent ----
    J₁ = plus(random_fields(3; seed=1), 1, junk(3; seed=1))
    J₂ = plus(random_fields(3; seed=2), 1, junk(3; seed=2))
    S₁ = KS.apply_symmetry!(copy(J₁))
    @test cpu(KS.apply_symmetry!(copy(S₁))) == cpu(S₁)
    @test products(S₁, J₂) ≈ products(J₁, KS.apply_symmetry!(copy(J₂))) rtol=1e-14
    @test all(abs.(products(plus(J₁, -1, S₁), S₁)) .< 1e-14 .* products(J₁, J₁))

    # ---- the inner product of OKF, field by field and summed, on views too ----
    Ψs = okf_fields(3; seed=3)
    Y  = batch(Ψs)
    c  = dots(X, Y)
    @test size(c) == (1, 1, 3)
    @test products(X, Y) ≈ [dot(Ωs[i], Ψs[i]) for i in 1:3] rtol=1e-14
    @test dot(X, Y) ≈ sum(cpu(c))
    @test dot(X, X) ≈ sum(norm(Ω)^2 for Ω in Ωs) rtol=1e-14
    @test dot(X, X) ≈ Re * sum(OKF.dissrate(Ω, Re) for Ω in Ωs) rtol=1e-14
    @test products(view(X, :, :, 2:2), view(Y, :, :, 2:2))[1] ≈ cpu(c)[2]

    # ---- the states of MultipleShooting, the first and the last, are batches of fields, with the
    #      inner product of the fields ----
    @test MultipleShooting.states(X, 2:3) isa FTFields
    @test head(X) isa FTFields && tail(X) isa FTFields
    @test dot(head(X), head(Y)) ≈ cpu(c)[1]
    @test dot(tail(X), tail(Y)) ≈ cpu(c)[3]

    # ---- one reduction, with no array of the size of the batch in between: on many fields it
    #      allocates a small share of one batch of Float64 ----
    Z = random_fields(64; seed=5)
    W = random_fields(64; seed=6)
    @test allocated(() -> dots(Z, W)) < sizeof(parent(Z))/20

    # ---- symmetric, and linear in each argument ----
    @test products(X, Y) ≈ products(Y, X) rtol=1e-15
    @test products(plus(X, 2.5, Y), Y) ≈ products(X, Y) .+ 2.5 .* products(Y, Y) rtol=1e-14
end
