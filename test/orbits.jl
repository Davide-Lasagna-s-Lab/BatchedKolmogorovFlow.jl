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

@testset "read_orbit, the translation along y    " begin
    mktempdir() do dir
        # ---- a seed file with other_sy, an orbit file with the length yshift = -2π sy/kf of
        #      ReSolver and kf, and a guess of kf_recurrences_tight.jl, for every translation ----
        for sy in -1:2
            seeds = joinpath(dir, "seeds$(sy).h5")
            h5open(seeds, "w") do f
                f["seed_1_real"] = zeros(18, 10)
                f["seed_1_imag"] = zeros(18, 10)
                A = HDF5.attributes(f)
                A["d1"]       = 5.0
                A["d2"]       = 0.1
                A["other_sy"] = sy
            end

            orbit = joinpath(dir, "orbit$(sy).h5")
            h5open(orbit, "w") do f
                f["real"] = zeros(9, 17, 2)
                f["imag"] = zeros(9, 17, 2)
                A = HDF5.attributes(f)
                A["period"] = 5.0
                A["drift"]  = 0.1
                A["yshift"] = -2π*sy/kf
                A["kf"]     = kf
            end

            # ---- a guess of kf_recurrences_tight.jl: other_sy, and yshift = +2π sy/kf ----
            tight = joinpath(dir, "tight$(sy).h5")
            h5open(tight, "w") do f
                f["real"] = zeros(9, 17, 2)
                f["imag"] = zeros(9, 17, 2)
                A = HDF5.attributes(f)
                A["period"]   = 5.0
                A["drift"]    = 0.1
                A["yshift"]   = 2π*sy/kf
                A["other_sy"] = sy
            end

            @test read_orbit(seeds, n).sy === sy
            @test read_orbit(orbit, n).sy === sy
            @test read_orbit(tight, n).sy === sy
        end

        # ---- a seed file with no other_sy closes after no translation ----
        bare = joinpath(dir, "bare.h5")
        h5open(bare, "w") do f
            f["seed_1_real"] = zeros(18, 10)
            f["seed_1_imag"] = zeros(18, 10)
            HDF5.attributes(f)["d1"] = 5.0
            HDF5.attributes(f)["d2"] = 0.1
        end
        @test read_orbit(bare, n).sy === 0
    end
end

@testset "read_orbit, a guess on the integrator's steps" begin
    mktempdir() do dir
        # ---- the states of a guess at nf = 10, ky in FFT order of 2nf + 1, every 50 steps of N = 120 and the end ----
        Random.seed!(12)
        nf      = 10
        indices = [0, 50, 100, 120]
        a       = zeros(ComplexF64, nf + 1, 2nf + 1, 4)
        for l in 1:4, kx in 0:nf, ky in -nf:nf
            a[kx + 1, mod(ky, 2nf + 1) + 1, l] = randn(ComplexF64)
        end

        # ---- a file with every attribute, a file with the core ones alone, and a group of a pack ----
        full = joinpath(dir, "full.h5")
        core = joinpath(dir, "core.h5")
        pack = joinpath(dir, "pack.h5")
        for (path, all) in ((full, true), (core, false), (pack, true))
            h5open(path, "w") do f
                g = path == pack ? create_group(f, "b20-r001-0001") : f
                g["states_real"] = real.(a)
                g["states_imag"] = imag.(a)
                A = HDF5.attributes(g)
                A["d1"]            = 1.2
                A["d2"]            = 0.4
                A["other_sy"]      = -1
                A["other_indices"] = indices
                if all
                    A["other_steps"]    = 120
                    A["other_dt"]       = 0.01
                    A["other_distance"] = 0.07
                end
            end
        end

        R = read_orbit(full, n)
        C = read_orbit(core, n)
        Q = read_orbit(pack * ":b20-r001-0001", n)

        # ---- the attributes, given or derived: N the last index, dt = d1/N ----
        @test (R.kind, R.T, R.s, R.sy, R.steps, R.dt, R.indices, R.distance) ==
              (:steps, 1.2, 0.4, -1, 120, 0.01, indices, 0.07)
        @test (C.steps, C.indices) == (120, indices)
        @test C.dt ≈ 0.01 rtol=1e-15
        @test isnan(C.distance)
        @test cpu(Q.X) == cpu(R.X) && Q.distance == 0.07

        # ---- the modes of the file up to n, at the resolution of the tests, with no mean ----
        E = zeros(ComplexF64, n + 1, 2n + 1, 4)
        for l in 1:4, kx in 0:n, ky in -n:n
            E[kx + 1, mod(ky, 2n + 1) + 1, l] = a[kx + 1, mod(ky, 2nf + 1) + 1, l]
        end
        E[1, 1, :] .= 0
        @test cpu(R.X) == E
        @test cpu(C.X) == E
    end
end
