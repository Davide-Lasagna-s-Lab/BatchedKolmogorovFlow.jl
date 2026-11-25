# The derivatives of a batch, and the transforms between Fourier space and the grid.

@testset "derivatives                            " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)
    D  = similar(X)

    # ---- the derivatives of OKF ----
    for i in 1:3
        Ω = Ωs[i]
        @test cpu(KS.ddx!(D, X))[:, :, i] == coefficients(OKF.ddx!(OKF.FTField(n, m), Ω))
        @test cpu(KS.ddy!(D, X))[:, :, i] == coefficients(OKF.ddy!(OKF.FTField(n, m), Ω))
        @test cpu(KS.laplacian!(D, X))[:, :, i] == coefficients(OKF.laplacian!(OKF.FTField(n, m), Ω))
        @test cpu(KS.invlaplacian!(D, X))[:, :, i] ≈ coefficients(OKF.invlaplacian!(OKF.FTField(n, m), Ω)) rtol=1e-15
        @test cpu(KS.invlaplacian!(D, X, 0.3))[:, :, i] ≈ coefficients(OKF.invlaplacian!(OKF.FTField(n, m), Ω, 0.3)) rtol=1e-15
    end

    # ---- in place, the output the input ----
    @test cpu(KS.ddx!(copy(X), copy(X))) == cpu(KS.ddx!(D, X))
    @test cpu(KS.ddy!(copy(X), copy(X))) == cpu(KS.ddy!(D, X))
    @test cpu(KS.invlaplacian!(copy(X), copy(X))) == cpu(KS.invlaplacian!(D, X))

    # ---- the inverses: Δ⁻¹ on fields of zero mean, and (I - cΔ)⁻¹ ----
    @test reldiff(KS.laplacian!(D, KS.invlaplacian!(similar(X), X)), X) < 1e-15
    Z = KS.invlaplacian!(similar(X), X, 0.3)
    @test reldiff(plus(Z, -0.3, KS.laplacian!(D, Z)), X) < 1e-15

    # ---- the second derivatives are the derivatives twice ----
    @test reldiff(KS.laplacian!(D, X),
                  plus(KS.ddx!(similar(X), KS.ddx!(similar(X), X)), 1, KS.ddy!(similar(X), KS.ddy!(similar(X), X)))) < 1e-15

    # ---- the derivatives are antisymmetric in the inner product, the Laplacian symmetric ----
    Y = random_fields(3)
    @test products(KS.ddx!(D, X), Y) ≈ .-products(X, KS.ddx!(similar(Y), Y)) rtol=1e-13
    @test products(KS.ddy!(D, X), Y) ≈ .-products(X, KS.ddy!(similar(Y), Y)) rtol=1e-13
    @test products(KS.laplacian!(D, X), Y) ≈ products(X, KS.laplacian!(similar(Y), Y)) rtol=1e-13
end

@testset "transforms                             " begin
    Ωs = okf_fields(3)
    X  = batch(Ωs)

    # ---- to the grid as OKF, whose fields have y down the rows, and back ----
    ifft! = KS.InverseFFT!(X, m)
    fft!  = KS.ForwardFFT!(X, m)
    u     = ifft!(Fields(X, m), X)
    @test u isa Fields{m}
    for i in 1:3
        @test reldiff(cpu(u)[:, :, i], permutedims(parent(OKF.IFFT(Ωs[i])))) < 1e-14
    end
    @test reldiff(fft!(similar(X), u), X) < 1e-14

    # ---- the forward transform makes fields of anything on the grid, and keeps their modes ----
    Random.seed!(4)
    r = Fields(ARRAY(randn(2m + 2, 2m + 2, 3)))
    R = fft!(similar(X), r)
    @test cpu(KS.apply_symmetry!(copy(R))) == cpu(R)
    @test reldiff(fft!(similar(X), ifft!(similar(u), R)), R) < 1e-14

    # ---- the product of two fields has no aliasing on the modes kept ----
    Y = batch(okf_fields(3; seed=5))
    v = ifft!(similar(u), Y)
    W = fft!(similar(X), Fields(u .* v))
    for i in 1:3
        w = OKF.FFT(OKF.Field(parent(OKF.IFFT(Ωs[i])) .* parent(OKF.IFFT(OKF.FTField(Y, i, m)))), n)
        @test reldiff(cpu(W)[:, :, i], coefficients(w)) < 1e-13
    end

    # ---- the mean over the grid of the product of two fields is their inner product ----
    mean(a) = vec(sum(cpu(a); dims=(1, 2))) ./ (2m + 2)^2
    @test mean(Fields(u .* v)) ≈ products(X, Y) rtol=1e-13

    # ---- each transform is the transpose of the other, in the inner product of the fields and the
    #      mean over the grid: the step of TangentTerm that AdjointTerm reverses ----
    @test mean(Fields(u .* r)) ≈ products(X, R) rtol=1e-13
end
