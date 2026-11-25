# The transforms between a batch of fields in Fourier space, FTFields{n}, and the same fields on
# the grid, Fields{m}.
#
# The grid is finer than the modes need, 2m + 2 ≥ 3n + 1, so that the product of two fields on
# the grid has the right coefficients on the modes kept, with no aliasing. The transforms act on
# the coefficients of all the wavenumbers of the grid, an array (m+2, 2m+2, B), with j = 0, …, m+1
# down the rows and k in the order 0, …, m, -m-1, …, -1 across the columns: the modes of the
# fields are copied into it, among zeros, before an inverse transform, and read from it after a
# forward one.
#
# The terms transform their batches two at a time, the two components of the velocity, the two
# products: as one batch of 2B fields, whose two halves they are, by one execution of a plan in
# place of two. The work is the same, in half the launches on the GPU, where they cost more than
# the work for the batches of a search.

# a batch of twice the fields of X, on its device and with its wavenumbers and weights, for two
# batches like X in its two halves
_twice(X::FTFields) = similar(X, ComplexF64, (size(X, 1), size(X, 2), 2size(X, 3)))

# the two halves of a batch along its last dimension, views, in Fourier space or on the grid, or
# of a part of a bundle (linearised.jl)
function _halves(A::AbstractArray{T, N}) where {T, N}
    B = size(A, N) ÷ 2
    c = ntuple(_ -> Colon(), Val(N - 1))

    return view(A, c..., 1:B), view(A, c..., (B + 1):2B)
end

# the modes of the fields U among the wavenumbers of the grid, the others zero
function _pad!(padded::AbstractArray, U::FTFields{n}) where {n}
    M = size(padded, 2)

    # ---- zeros at the wavenumbers that the fields do not have ----
    fill!(padded, 0)

    # ---- the columns of k = 0, …, n, which are the first of both arrays ----
    view(padded, 1:(n + 1), 1:(n + 1), :) .= view(parent(U), :, 1:(n + 1), :)

    # ---- the columns of k = -n, …, -1, which are the last of both arrays ----
    view(padded, 1:(n + 1), (M - n + 1):M, :) .= view(parent(U), :, (n + 2):(2n + 1), :)

    return padded
end

# the modes of the fields U from the coefficients of the grid, over its number of points
function _unpad!(U::FTFields{n}, padded::AbstractArray) where {n}
    M = size(padded, 2)

    # ---- the columns of k = 0, …, n ----
    view(parent(U), :, 1:(n + 1), :) .= view(padded, 1:(n + 1), 1:(n + 1), :) ./ M^2

    # ---- the columns of k = -n, …, -1 ----
    view(parent(U), :, (n + 2):(2n + 1), :) .= view(padded, 1:(n + 1), (M - n + 1):M, :) ./ M^2

    return U
end

"""
    InverseFFT!(X, m)

The transform of batches of fields like `X` to the grid of `2m+2` points along each direction,
with its plan and its arrays: `ifft!(u, U)` writes the values of the fields `U` on the grid into
the batch `u`, a `Fields{m}`, and returns it.

# Arguments
- `X::FTFields`: a batch of fields, for its size and its device.
- `m::Int`: the grid has `2m+2` points along each direction.
"""
struct InverseFFT!{C, P}
    padded::C # the coefficients of all the wavenumbers of the grid, (m+2, 2m+2, B)
      plan::P # the complex-to-real transform over the first two dimensions, not normalised

    function InverseFFT!(X::FTFields, m::Int)
        # ---- the array of the coefficients of the grid, on the device of X ----
        padded = similar(X, ComplexF64, (m + 2, 2m + 2, size(X, 3)))

        # ---- the transform over x and y, all the fields of the batch in one call ----
        plan = AbstractFFTs.plan_brfft(padded, 2m + 2, (1, 2))

        return new{typeof(padded), typeof(plan)}(padded,
                                                 plan)
    end
end

function (ifft!::InverseFFT!)(u::Fields, U::FTFields)
    # ---- the modes of the fields among the wavenumbers of the grid ----
    _pad!(ifft!.padded, U)

    # ---- to the grid. The transform destroys its input, which the next call writes again ----
    LinearAlgebra.mul!(parent(u), ifft!.plan, ifft!.padded)

    return u
end

"""
    ForwardFFT!(X, m)

The transform from the grid of `2m+2` points along each direction to batches of fields like `X`,
with its plan and its arrays: `fft!(U, u)` writes into the batch `U` the modes of the fields `u`
on the grid, a `Fields{m}`, with the symmetry of fields, and returns `U`. The modes that `U` does
not have are dropped.

# Arguments
- `X::FTFields`: a batch of fields, for its size and its device.
- `m::Int`: the grid has `2m+2` points along each direction.
"""
struct ForwardFFT!{C, P}
    padded::C # the coefficients of all the wavenumbers of the grid, (m+2, 2m+2, B)
      plan::P # the real-to-complex transform over the first two dimensions, not normalised

    function ForwardFFT!(X::FTFields, m::Int)
        # ---- the array of the coefficients of the grid, on the device of X ----
        padded = similar(X, ComplexF64, (m + 2, 2m + 2, size(X, 3)))

        # ---- the transform over x and y, planned on a batch on the grid ----
        plan = AbstractFFTs.plan_rfft(parent(Fields(X, m)), (1, 2))

        return new{typeof(padded), typeof(plan)}(padded,
                                                 plan)
    end
end

function (fft!::ForwardFFT!)(U::FTFields, u::Fields)
    # ---- the coefficients of all the wavenumbers of the grid ----
    LinearAlgebra.mul!(fft!.padded, fft!.plan, parent(u))

    # ---- those of the modes of the fields, normalised ----
    _unpad!(U, fft!.padded)

    # ---- the row j = 0 Hermitian to rounding, and zero mean ----
    return apply_symmetry!(U)
end
