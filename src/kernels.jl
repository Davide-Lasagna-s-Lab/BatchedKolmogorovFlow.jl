# The terms of the equation entry by entry, in place of their broadcasts.
#
# A term made with `kernels = true` is three passes over the arrays between its two transforms,
#
#     the velocity     the fields  →  their velocity among the wavenumbers of the grid, zeros too
#     the products     the velocity on the grid  →  the two products of Basdevant
#     the advection    the transforms of the products  →  the term, its symmetry and its forcings
#
# where the broadcasts of nonlinear.jl and linearised.jl make 13 and 17. On the GPU a pass is a
# kernel, and a kernel costs its launch whatever it computes: for the batches of a search the
# launches, and not the arithmetic, set the time of a step.
#
# This file holds what one entry of each pass is, a function of the arrays and of the indices of
# the entry, and the three loops of the CPU. The extension of the package for CUDA makes a kernel
# of each function (ext/kernels.jl). The explicit term is the term of a bundle with no
# perturbation, k′ = 0, so that the same three serve both.
#
# The arrays. A batch of fields of largest wavenumber n is (n+1, 2n+1, B): the row r holds
# j = r - 1, the column c holds k = c - 1 for c ≤ n + 1 and k = c - 2n - 2 after. An array of all
# the wavenumbers of the grid is (m+2, M, ·), M = 2m + 2: its column c holds k = c - 1 among the
# first and k = c - M - 1 among the last. The arrays of two halves (fft.jl) hold the first of a
# pair in their first half of slices and the second in the other: the entry i of a half and the
# entry i + N of the other are the same place of the same field, N the entries of a half. In a
# bundle every field is followed by its k′ perturbations (linearised.jl): the slice b is in the
# place a = (b - 1) mod (1 + k′) of its group, 0 for a field, and b - a is the field of its group.

# The entry (r, c, b) of the velocity among the wavenumbers of the grid, M columns, from the fields
# Ω, B of them: U = i k ω/(j² + k²) in the first B slices, V = -i j ω/(j² + k²) in the last B,
# zero at the mean and at the wavenumbers that the fields do not have
@inline function _velocity_entry(Ω, n::Int, M::Int, r::Int, c::Int, b::Int)
    B = size(Ω, 3)
    j = r - 1

    # ---- the wavenumber k of the column, and the column q of the fields that holds it ----
    if c <= n + 1
        k = c - 1
        q = c
    elseif c >= M - n + 1
        k = c - M - 1
        q = c - M + 2n + 1
    else
        return zero(ComplexF64)
    end

    d = j*j + k*k
    (j > n || d == 0) &&
        return zero(ComplexF64)

    return @inbounds b <= B ? im*k*Ω[r, q, b]/d : -im*j*Ω[r, q, b - B]/d
end

# The two products at the entry i of a half of the velocity on the grid, u in the first half of
# uv and v in the second: u v and v² - u² for a field, and u v′ + u′ v and 2 (v v′ - u u′) for a
# perturbation, about the velocity (u, v) of the field of its group at the same point
@inline function _products_entry(uv, k′::Int, i::Int)
    N = length(uv) >> 1
    s = size(uv, 1)*size(uv, 2)

    # ---- the place of the slice in its group, and the same point of the field of the group ----
    a = k′ == 0 ? 0 : ((i - 1) ÷ s) % (1 + k′)
    f = i - a*s

    @inbounds begin
        u = uv[f]
        v = uv[f + N]
        a == 0 &&
            return u*v, v*v - u*u
        u′ = uv[i]
        v′ = uv[i + N]
    end

    return u*v′ + u′*v, 2*(v*v′ - u*u′)
end

# The entry (r, c, b) of the term, from the transforms PQ of the two products on all the
# wavenumbers of the grid, not normalised, P in the first half of the slices and Q in the second:
# the advection (j² - k²) P + j k Q, with P and Q over M². In the row j = 0 the entries at k and
# -k are made conjugates, P[0, k] → (P[0, k] + conj(P[0, -k]))/2, where Q has the factor zero, and
# the mean comes out zero. A field has the forcing -kf/2 at j = 0, k = ±kf, and a perturbation
# γ (j² + k²) ω, with ω the field of its group in the bundle Z: the derivative of the viscous term
# in the Reynolds number for γ = 1/Re², nothing for γ = 0
@inline function _advection_entry(PQ, Z, n::Int, kf::Int, k′::Int, γ::Float64,
                                  r::Int, c::Int, b::Int)
    M = size(PQ, 2)
    B = size(PQ, 3) >> 1
    j = r - 1
    k = c <= n + 1 ? c - 1 : c - 2n - 2
    a = k′ == 0 ? 0 : (b - 1) % (1 + k′)

    # ---- the columns of k and of -k among those of the grid ----
    p = k >= 0 ? k + 1 : M + k + 1
    q = k <= 0 ? 1 - k : M - k + 1

    out = @inbounds j == 0 ? -k*k*(PQ[1, p, b] + conj(PQ[1, q, b]))/(2*M*M) :
                             ((j*j - k*k)*PQ[r, p, b] + j*k*PQ[r, p, b + B])/(M*M)

    a == 0 &&
        return j == 0 && abs(k) == kf ? out - kf/2 : out

    return @inbounds out + γ*(j*j + k*k)*Z[r, c, b - a]
end

# The term eq of a bundle Z of k′ perturbations of every field, written into dZdt, by the three
# passes above between the two transforms of eq, on its arrays: on the CPU, three loops. The
# extension for CUDA has the method of the batches on the GPU, three kernels
function _kernels!(dZdt::FTFields{n}, eq, Z::FTFields{n}, k′::Int, γ::Float64) where {n}
    Û   = eq.ifft!.padded
    P̂   = eq.fft!.padded
    uv  = parent(eq.uv)
    pq  = parent(eq.pq)
    Ω   = parent(Z)
    out = parent(dZdt)
    N   = length(uv) >> 1

    # ---- in Fourier space: the velocity, among the wavenumbers of the grid ----
    for I in CartesianIndices(Û)
        @inbounds Û[I] = _velocity_entry(Ω, n, size(Û, 2), I[1], I[2], I[3])
    end

    # ---- to the grid, the products there, and back ----
    LinearAlgebra.mul!(uv, eq.ifft!.plan, Û)
    for i in 1:N
        @inbounds pq[i], pq[i + N] = _products_entry(uv, k′, i)
    end
    LinearAlgebra.mul!(P̂, eq.fft!.plan, pq)

    # ---- in Fourier space: the advection and the forcings ----
    for I in CartesianIndices(out)
        @inbounds out[I] = _advection_entry(P̂, Ω, n, eq.kf, k′, γ, I[1], I[2], I[3])
    end

    return dZdt
end
