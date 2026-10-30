# Reading the orbits and the recurrences of the Kolmogorov campaigns into
# batches of seeds in the layout of OpenKolmogorovFlow, as shoot.jl of
# ReSolver.jl/examples/kolmogorov/upo/shooting does. Included by the scripts of
# this folder.
#
# Two formats:
#   - a seed file of NKSearch or of recurrences.jl: datasets seed_i_real and
#     seed_i_imag, the raw coefficients of OKF (ky in FFT order down the rows, kx ≥ 0
#     across the columns), and the attributes d1 (period), d2 (shift of NKSearch)
#     and other_sy (the translation along y by whole wavelengths of the force);
#   - an orbit of the campaigns: datasets real and imag, (n+1) × (2n+1) × Ns with kx
#     along the first dimension and ky in FFT order, and the attributes period,
#     drift and yshift.
#
# In both cases the orbit closes as S(Φ_T(x₁), s) = x₁, with S the shift by s along
# x and by ℓy along y, shift!(P, x, s, ℓy).

using HDF5

"""
    read_orbit(path, n, m, kf) -> (X, T, s, ℓy, kind)

The states of the orbit in `path`, as a complex array `(2m+2, m+2, N₀)` in the
layout of OKF and in the lab frame, its period `T`, its shift `s` along `x` and
its translation `ℓy` along `y`. `kind` is `:seeds` or `:orbit`.
"""
function read_orbit(path::String, n::Int, m::Int, kf::Int)
    # ---- the row of OKF of the wavenumber ky ----
    row(ky) = ky >= 0 ? ky + 1 : 2m + 3 + ky

    h5open(path, "r") do f
        A = HDF5.attributes(f)
        if haskey(f, "seed_1_real")
            # ---- seeds of NKSearch or of recurrences.jl ----
            N₀ = count(k -> endswith(k, "_real"), keys(f))
            X  = zeros(ComplexF64, 2m + 2, m + 2, N₀)
            for i in 1:N₀
                raw = read(f["seed_$(i)_real"]) .+ im .* read(f["seed_$(i)_imag"])
                for kx in 0:n, ky in -n:n
                    X[row(ky), kx + 1, i] = raw[mod(ky, size(raw, 1)) + 1, kx + 1]
                end
            end
            X[1, 1, :] .= 0
            T  = Float64(read(A["d1"]))
            s  = rem(Float64(read(A["d2"])), 2π, RoundNearest)
            sy = haskey(A, "other_sy") ? Int(read(A["other_sy"])) : 0
            return (X=X, T=T, s=s, ℓy=sy*2π/kf, kind=:seeds)
        else
            # ---- an orbit of the campaigns, its instants moved back to the lab frame by the drift ----
            a  = read(f["real"]) .+ im .* read(f["imag"])
            Ns = size(a, 3)
            T  = Float64(read(A["period"]))
            c  = Float64(read(A["drift"]))
            ℓy = haskey(A, "yshift") ? Float64(read(A["yshift"])) : 0.0
            X  = zeros(ComplexF64, 2m + 2, m + 2, Ns)
            for l in 0:(Ns - 1), kx in 0:n, ky in -n:n
                X[row(ky), kx + 1, l + 1] = a[kx + 1, ky >= 0 ? ky + 1 : 2n + 2 + ky, l + 1] *
                                            cis(-kx * c * (l * T / Ns))
            end
            X[1, 1, :] .= 0
            return (X=X, T=T, s=c*T, ℓy=-ℓy, kind=:orbit)
        end
    end
end

"""
    make_seeds(P, orbit, segment) -> X

The seeds of about `T/segment` segments of equal length, a batch on the device
of the propagator `P`. From seeds, `M` a multiple of the `N₀` seeds of the file, each advanced
in batch to fill its own segment; from an orbit of the campaigns, `M` the divisor
of its `Ns` instants closest to `T/segment`, and the instants themselves.
"""
function make_seeds(P, orbit, segment::Real)
    N₀ = size(orbit.X, 3)
    if orbit.kind === :seeds
        # ---- the stored seeds, advanced together to fill their segments: whole steps on the grid of
        #      the recurrence, and the part of a step that is left on a copy, which the next segment
        #      does not continue, because a shorter step would leave the trajectory of the simulation ----
        p = max(1, round(Int, orbit.T / (N₀ * segment)))
        M = N₀ * p
        X = zeros(ComplexF64, size(orbit.X, 1), size(orbit.X, 2), M)
        Y = KolmogorovShootingGPU._device(P.template, orbit.X)
        w = 0
        for q in 1:p
            t = (q - 1) * orbit.T / M
            v = floor(Int, t / P.dt + 1e-6)
            v > w && propagate!(P, Y, (v - w) * P.dt)
            w = v
            X[:, :, q:p:M] .= Array(t > w * P.dt ? propagate!(P, copy(Y), t - w * P.dt) : Y)
        end
    else
        # ---- the instants of the orbit at equal intervals ----
        M = argmin(d -> abs(d - orbit.T / segment), filter(d -> N₀ % d == 0, 1:N₀))
        X = orbit.X[:, :, 1:(N₀ ÷ M):N₀]
    end
    return KolmogorovShootingGPU._device(P.template, X)
end
