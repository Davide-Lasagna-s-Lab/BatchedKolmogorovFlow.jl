# read_orbit of examples/orbits.jl: the modes of a file read at a resolution other than its own, of
# a seed file and of an orbit file, are those of the file truncated or padded with zeros.

using HDF5

@testset "read_orbit at other resolutions        " begin
    mktempdir() do dir
        # ---- a seed file of nf = 10 on a grid of 30 rows, and an orbit file of nf = 10 ----
        Random.seed!(11)
        nf   = 10
        rows = 30
        cols = 16
        Ns   = 4
        T    = 7.5
        c    = 0.3

        # the modes of every state, ky in FFT order down the rows of the seeds, zero beyond nf
        modes = zeros(ComplexF64, nf + 1, 2nf + 1, Ns)
        for l in 1:Ns, kx in 0:nf, ky in -nf:nf
            modes[kx + 1, mod(ky, 2nf + 1) + 1, l] = randn(ComplexF64)
        end

        raws = [zeros(ComplexF64, rows, cols) for l in 1:Ns]
        for l in 1:Ns, kx in 0:nf, ky in -nf:nf
            raws[l][mod(ky, rows) + 1, kx + 1] = modes[kx + 1, mod(ky, 2nf + 1) + 1, l]
        end

        seeds = joinpath(dir, "seeds.h5")
        h5open(seeds, "w") do f
            for l in 1:Ns
                f["seed_$(l)_real"] = real.(raws[l])
                f["seed_$(l)_imag"] = imag.(raws[l])
            end
            HDF5.attributes(f)["d1"] = T
            HDF5.attributes(f)["d2"] = 1.0
        end

        orbit = joinpath(dir, "orbit.h5")
        h5open(orbit, "w") do f
            f["real"] = real.(modes)
            f["imag"] = imag.(modes)
            HDF5.attributes(f)["period"] = T
            HDF5.attributes(f)["drift"]  = c
            HDF5.attributes(f)["yshift"] = 0.0
        end

        # ---- the modes of the states in the lab frame, ky in FFT order of 2n + 1, at the resolution n ----
        expected(n, kind) = begin
            E = zeros(ComplexF64, n + 1, 2n + 1, Ns)
            for l in 1:Ns, kx in 0:min(n, nf), ky in -min(n, nf):min(n, nf)
                phase = kind === :orbit ? cis(-kx * c * ((l - 1) * T / Ns)) : 1
                E[kx + 1, mod(ky, 2n + 1) + 1, l] = modes[kx + 1, mod(ky, 2nf + 1) + 1, l] * phase
            end
            E[1, 1, :] .= 0
            E
        end

        # the energy and the enstrophy of a state, from its modes, with the weight of the half plane
        function integrals(X, n)
            j = 0:n
            k = [0:n; -n:-1]
            w = [1; fill(2, n)]
            K = j.^2 .+ k'.^2
            Z = [sum(w .* abs2.(X[:, :, l])) for l in 1:size(X, 3)]
            Ek = [sum(w .* abs2.(X[:, :, l]) ./ max.(K, 1)) for l in 1:size(X, 3)]
            return Ek, Z
        end

        for (path, kind) in ((seeds, :seeds), (orbit, :orbit))
            Xf = parent(read_orbit(path, nf).X)

            # ---- at the resolution of the file, exactly the modes ----
            @test Xf == expected(nf, kind)

            for n in (6, 14, 24)
                X = parent(read_orbit(path, n).X)
                R = read_orbit(path, n)

                # ---- the modes themselves ----
                @test size(X) == (n + 1, 2n + 1, Ns)
                @test X == expected(n, kind)
                @test R.kind === kind
                @test R.T == T

                # ---- the same field as the file at nf, truncated or padded in Fourier space ----
                m  = min(n, nf)
                Xt = zeros(ComplexF64, n + 1, 2n + 1, Ns)
                for kx in 0:m, ky in -m:m
                    Xt[kx + 1, mod(ky, 2n + 1) + 1, :] = Xf[kx + 1, mod(ky, 2nf + 1) + 1, :]
                end
                @test X == Xt

                # ---- energy and enstrophy: those of the file when n ≥ nf, those of the truncation otherwise ----
                Ek, Z = integrals(X, n)
                Ek₀, Z₀ = integrals(Xt, n)
                @test Ek == Ek₀ && Z == Z₀
                # the sums run over more zeros, in another order: equal to rounding
                if n ≥ nf
                    Ekf, Zf = integrals(Xf, nf)
                    @test Ek ≈ Ekf rtol=1e-12
                    @test Z ≈ Zf rtol=1e-12
                end
            end
        end
    end
end
