# Reading the orbits and the recurrences of the Kolmogorov campaigns into
# batches of seeds of the package, as shoot.jl of
# ReSolver.jl/examples/kolmogorov/upo/shooting does. Included by the scripts of
# this folder.
#
# Two formats:
#   - a seed file of NKSearch or of recurrences.jl: datasets seed_i_real and
#     seed_i_imag, the raw coefficients of OKF (ky in FFT order down the rows, kx ≥ 0
#     across the columns), and the attributes d1 (period), d2 (shift of NKSearch)
#     and other_sy (the translation along y by whole wavelengths of the force);
#   - an orbit of the campaigns: datasets real and imag, (n+1) × (2n+1) × Ns with kx
#     along the first dimension and ky in FFT order, the layout of the batches, and
#     the attributes period, drift and yshift.
#
# In both cases the orbit closes as S(Φ_T(x₁), s) = x₁, with S the shift by s along
# x, XShift()(x, s).

using HDF5

"""
    read_orbit(path, n) -> (X, T, s, kind)

The states of the orbit in `path`, as a batch of `N₀` fields, an `FTFields{n}`, in the lab
frame, its period `T` and its shift `s` along `x`. `kind` is `:seeds` or `:orbit`. An orbit that
closes after a translation along `y` is refused, since the propagator shifts along `x` alone.
"""
function read_orbit(path::String, n::Int)
    # ---- the column of a batch of the wavenumber ky ----
    col(ky) = mod(ky, 2n + 1) + 1

    h5open(path, "r") do f
        A = HDF5.attributes(f)
        if haskey(f, "seed_1_real")
            # ---- seeds of NKSearch or of recurrences.jl ----
            (haskey(A, "other_sy") && Int(read(A["other_sy"])) != 0) &&
                error("the seeds of $(path) close after a translation along y")
            N₀ = count(k -> endswith(k, "_real"), keys(f))
            X  = zeros(ComplexF64, n + 1, 2n + 1, N₀)
            for i in 1:N₀
                raw = read(f["seed_$(i)_real"]) .+ im .* read(f["seed_$(i)_imag"])
                for kx in 0:n, ky in -n:n
                    X[kx + 1, col(ky), i] = raw[mod(ky, size(raw, 1)) + 1, kx + 1]
                end
            end
            X[1, 1, :] .= 0
            T = Float64(read(A["d1"]))
            s = rem(Float64(read(A["d2"])), 2π, RoundNearest)
            return (X=FTFields(X), T=T, s=s, kind=:seeds)
        else
            # ---- an orbit of the campaigns, its instants moved back to the lab frame by the drift ----
            (haskey(A, "yshift") && Float64(read(A["yshift"])) != 0) &&
                error("the orbit of $(path) closes after a translation along y")
            a  = read(f["real"]) .+ im .* read(f["imag"])
            Ns = size(a, 3)
            T  = Float64(read(A["period"]))
            c  = Float64(read(A["drift"]))
            X  = zeros(ComplexF64, n + 1, 2n + 1, Ns)
            for l in 0:(Ns - 1), kx in 0:n, ky in -n:n
                X[kx + 1, col(ky), l + 1] = a[kx + 1, col(ky), l + 1] * cis(-kx * c * (l * T / Ns))
            end
            X[1, 1, :] .= 0
            return (X=FTFields(X), T=T, s=c*T, kind=:orbit)
        end
    end
end

# the batch X on the device of the batches of the flow G, that of its fields
on_device(G, X::FTFields) = FTFields(copyto!(similar(parent(G.whole.X), ComplexF64, size(X)), parent(X)))

"""
    make_seeds(G, orbit, segment) -> X

The seeds of about `T/segment` segments of equal length, a batch on the device
of the flow `G`, a `NonlinearPropagator` of the `N₀` fields of the file.
From seeds, `M` a multiple of the `N₀` seeds of the file, each advanced in batch
to fill its own segment; from an orbit of the campaigns, `M` the divisor of its
`Ns` instants closest to `T/segment`, and the instants themselves.
"""
function make_seeds(G, orbit, segment::Real)
    N₀ = size(orbit.X, 3)
    if orbit.kind === :seeds
        # ---- the stored seeds, advanced together to fill their segments: whole steps on the grid of
        #      the recurrence, and the part of a step that is left on a copy, which the next segment
        #      does not continue, because a shorter step would leave the trajectory of the simulation ----
        p = max(1, round(Int, orbit.T / (N₀ * segment)))
        M = N₀ * p
        X = zeros(ComplexF64, size(orbit.X, 1), size(orbit.X, 2), M)
        Y = on_device(G, orbit.X)
        w = 0
        for q in 1:p
            t = (q - 1) * orbit.T / M
            v = floor(Int, t / G.dt + 1e-6)
            v > w && G(Y, (v - w) * G.dt)
            w = v
            Z = t > w * G.dt ? G(copy(Y), t - w * G.dt) : Y
            X[:, :, q:p:M] .= Array(parent(Z))
        end
    else
        # ---- the instants of the orbit at equal intervals ----
        M = argmin(d -> abs(d - orbit.T / segment), filter(d -> N₀ % d == 0, 1:N₀))
        X = Array(parent(orbit.X))[:, :, 1:(N₀ ÷ M):N₀]
    end
    return on_device(G, FTFields(X))
end
