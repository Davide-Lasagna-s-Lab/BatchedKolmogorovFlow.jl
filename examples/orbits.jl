# Reading the orbits and the recurrences of the Kolmogorov campaigns into
# batches of seeds of the package, as shoot.jl of
# ReSolver.jl/examples/kolmogorov/upo/shooting does. Included by the scripts of
# this folder.
#
# Three formats:
#   - a seed file of NKSearch or of recurrences.jl: datasets seed_i_real and
#     seed_i_imag, the raw coefficients of OKF (ky in FFT order down the rows, kx ≥ 0
#     across the columns), and the attributes d1 (period), d2 (shift of NKSearch)
#     and other_sy (the translation along y by whole wavelengths of the force);
#   - an orbit of the campaigns: datasets real and imag, (n+1) × (2n+1) × Ns with kx
#     along the first dimension and ky in FFT order, the layout of the batches, and
#     the attributes period, drift and the translation along y: other_sy, in wavelengths
#     of the forcing, as kf_recurrences_tight.jl writes it, or else yshift, a length, which
#     is -2π sy/kf in the orbits of ReSolver (with kf among the attributes when it is not
#     zero) and +2π sy/kf in the guesses of kf_recurrences_tight.jl;
#   - a guess on the steps of the integrator (kf_recurrences_tight.jl of 26 Nov and the
#     databases after it), states of the simulation at its own time steps: datasets
#     states_real and states_imag, (n+1) × (2n+1) × Ns in the layout of the batches, and the
#     attributes d1 (the period, N steps of dt), d2, other_sy and other_indices (the step of
#     every state from the start), with other_steps (N), other_dt (dt) and other_distance
#     when the file has them: N is otherwise the last index, the end being stored, and dt is
#     d1/N.
#
# In all cases the orbit closes as S(Φ_T(x₁), s) = x₁, with S the shift by s along
# x and by sy wavelengths 2π/kf of the forcing along y, XShift(; sy, kf)(x, s), so that
# ω(x, y, t + T) = ω(x - s, y - 2π sy/kf, t).

using HDF5

"""
    read_orbit(path, n) -> (X, T, s, sy, kind, ...)

The states of the orbit in `path`, as a batch of `N₀` fields, an `FTFields{n}`, in the lab
frame, its period `T`, its shift `s` along `x` and its translation `sy` along `y`, in
wavelengths of the forcing, the keyword of [`XShift`](@ref). The modes of the file are truncated,
or padded with zeros, to the wavenumbers up to `n`. `kind` is `:seeds`, `:orbit` or `:steps`. A
guess of `kind = :steps` also gives `dt`, the time step of the simulation, `steps`, the N steps of
the recurrence, `indices`, the step of every state from the start, and `distance`, the closing
defect over the norm of the first state, NaN when the file does not give it. A guess left in the
pack of its run is read as `"<pack>.h5:<id>"`, the path that kf_recurrences_lists.jl gives it.
"""
function read_orbit(path::String, n::Int)
    # ---- the column of a batch of the wavenumber ky, of the batch asked for ----
    col(ky) = mod(ky, 2n + 1) + 1

    # ---- the file, and the group of a guess left in a pack, "<pack>.h5:<id>" ----
    parts = split(path, ".h5:")
    file  = length(parts) == 2 ? parts[1] * ".h5" : path
    group = length(parts) == 2 ? String(parts[2]) : ""

    h5open(file, "r") do io
        f = isempty(group) ? io : io[group]
        A = HDF5.attributes(f)
        if haskey(f, "seed_1_real")
            # ---- seeds of NKSearch or of recurrences.jl ----
            N₀ = count(k -> endswith(k, "_real"), keys(f))
            X  = zeros(ComplexF64, n + 1, 2n + 1, N₀)
            for i in 1:N₀
                raw = read(f["seed_$(i)_real"]) .+ im .* read(f["seed_$(i)_imag"])

                # the modes the file holds: kx up to its last column, |ky| up to the ones of its rows
                # that do not alias; the others, if n is larger, are zero, and if it is smaller, left out
                rows = size(raw, 1)
                kxf  = size(raw, 2) - 1
                kyf  = (rows - 1) ÷ 2

                for kx in 0:min(n, kxf), ky in -min(n, kyf):min(n, kyf)
                    X[kx + 1, col(ky), i] = raw[mod(ky, rows) + 1, kx + 1]
                end
            end
            X[1, 1, :] .= 0
            T  = Float64(read(A["d1"]))
            s  = rem(Float64(read(A["d2"])), 2π, RoundNearest)
            sy = haskey(A, "other_sy") ? Int(read(A["other_sy"])) : 0
            return (X=FTFields(X), T=T, s=s, sy=sy, kind=:seeds)
        elseif haskey(f, "states_real")
            # ---- a guess on the steps of the integrator, the states of the simulation itself ----
            a = read(f["states_real"]) .+ im .* read(f["states_imag"])
            X = zeros(ComplexF64, n + 1, 2n + 1, size(a, 3))

            # the file has the layout of a batch of its own n, nf: its modes up to n, or zeros beyond nf
            nf = size(a, 1) - 1
            for l in 1:size(a, 3), kx in 0:min(n, nf), ky in -min(n, nf):min(n, nf)
                X[kx + 1, col(ky), l] = a[kx + 1, mod(ky, 2nf + 1) + 1, l]
            end
            X[1, 1, :] .= 0

            # the steps of the recurrence, the last index unless given, and the time step, d1/N unless
            # given; the distance, NaN when the file does not give it
            T       = Float64(read(A["d1"]))
            indices = Int.(read(A["other_indices"]))
            N       = haskey(A, "other_steps") ? Int(read(A["other_steps"])) : last(indices)
            return (X        = FTFields(X),
                    T        = T,
                    s        = rem(Float64(read(A["d2"])), 2π, RoundNearest),
                    sy       = Int(read(A["other_sy"])),
                    kind     = :steps,
                    dt       = haskey(A, "other_dt") ? Float64(read(A["other_dt"])) : T/N,
                    steps    = N,
                    indices  = indices,
                    distance = haskey(A, "other_distance") ? Float64(read(A["other_distance"])) : NaN)
        else
            # ---- an orbit of the campaigns, its instants moved back to the lab frame by the drift ----
            a  = read(f["real"]) .+ im .* read(f["imag"])
            Ns = size(a, 3)
            T  = Float64(read(A["period"]))
            c  = Float64(read(A["drift"]))
            X  = zeros(ComplexF64, n + 1, 2n + 1, Ns)

            # the translation along y in wavelengths of the forcing: other_sy when the file has it,
            # or else from the length yshift = -2π sy/kf of the orbits of ReSolver
            ℓy = haskey(A, "yshift") ? Float64(read(A["yshift"])) : 0.0
            sy = haskey(A, "other_sy") ? Int(read(A["other_sy"])) :
                 ℓy == 0               ? 0 :
                                         round(Int, -ℓy*read(A["kf"])/2π)

            # the file has the layout of a batch of its own n, nf: ky in FFT order of its 2nf + 1 columns
            nf = size(a, 1) - 1
            for l in 0:(Ns - 1), kx in 0:min(n, nf), ky in -min(n, nf):min(n, nf)
                X[kx + 1, col(ky), l + 1] = a[kx + 1, mod(ky, 2nf + 1) + 1, l + 1] * cis(-kx * c * (l * T / Ns))
            end
            X[1, 1, :] .= 0
            return (X=FTFields(X), T=T, s=c*T, sy=sy, kind=:orbit)
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
