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
    @test similar(X).j === X.j && view(X, :, :, 1:2).k === X.k
    @test all(parent(OKF.FTField(X, i, m)) == parent(Ωs[i]) for i in 1:3)
    @test_throws ArgumentError FTFields(zeros(ComplexF64, 4, 8, 2))

    # ---- a batch like it, a copy, a view of some fields, and a broadcast into it ----
    @test similar(X) isa FTFields{n}
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
    #      on batches that are not fields, which makes adjoint! the transpose of tangent! ----
    J₁ = plus(random_fields(3; seed=1), 1, junk(3; seed=1))
    J₂ = plus(random_fields(3; seed=2), 1, junk(3; seed=2))
    S₁ = KS.apply_symmetry!(copy(J₁))
    @test cpu(KS.apply_symmetry!(copy(S₁))) == cpu(S₁)
    @test dots(S₁, J₂) ≈ dots(J₁, KS.apply_symmetry!(copy(J₂))) rtol=1e-14
    @test all(abs.(dots(plus(J₁, -1, S₁), S₁)) .< 1e-14 .* dots(J₁, J₁))

    # ---- the inner product of OKF, field by field and summed, on views too ----
    Ψs = okf_fields(3; seed=3)
    Y  = batch(Ψs)
    c  = realdots(X, Y)
    @test size(c) == (1, 1, 3)
    @test dots(X, Y) ≈ [dot(Ωs[i], Ψs[i]) for i in 1:3] rtol=1e-14
    @test realdot(X, Y) ≈ sum(cpu(c))
    @test realdot(X, X) ≈ sum(norm(Ω)^2 for Ω in Ωs) rtol=1e-14
    @test realdot(X, X) ≈ Re * sum(OKF.dissrate(Ω, Re) for Ω in Ωs) rtol=1e-14
    @test dots(view(X, :, :, 2:2), view(Y, :, :, 2:2))[1] ≈ cpu(c)[2]

    # ---- symmetric, and linear in each argument ----
    @test dots(X, Y) ≈ dots(Y, X) rtol=1e-15
    @test dots(plus(X, 2.5, Y), Y) ≈ dots(X, Y) .+ 2.5 .* dots(Y, Y) rtol=1e-14
end
