# A batch of vorticity fields in Fourier space.
#
# A field is real and periodic on [0, 2π)², with the Fourier coefficients
#
#     ω(x, y) = Σⱼ Σₖ Ω[j, k] e^{i (j x + k y)},   j, k = -n, …, n.
#
# Since ω is real, Ω[-j, -k] is the conjugate of Ω[j, k], and only the coefficients with j ≥ 0
# are kept. A batch holds B fields with the same largest wavenumber n, in one complex array of
# size (n+1, 2n+1, B): field b is the slice [:, :, b], with the wavenumber j = 0, …, n along x
# down the rows, and the wavenumber k along y across the columns, in the order
# 0, 1, …, n, -n, …, -1. Two conditions make such a slice a field: in the row j = 0 the entries at
# k and -k are conjugates, and the mean, at j = k = 0, is zero.
#
# This is the FTField{n, m} of OpenKolmogorovFlow without the modes that it keeps at zero, and
# transposed: x runs along the first dimension, which is the one that the real-to-complex
# transforms on the GPU halve. As there, the largest wavenumber is a parameter of the type. The
# batch also holds its wavenumbers, computed once when it is made.

"""
    FTFields(data)
    FTFields(Ωs)

A batch of `B` vorticity fields in Fourier space, with the largest wavenumber `n` in its type,
`FTFields{n}`. It is an array of size `(n+1, 2n+1, B)`: `X[j+1, c, b]` is the coefficient of
field `b` at the wavenumber `j = 0, …, n` along x and at the wavenumber `k` along y, with the
columns `c` in the order `k = 0, 1, …, n, -n, …, -1`. It wraps the array of its coefficients, on
the CPU or on the GPU, which `parent` returns, and broadcasts as that array does.

Its properties `X.j` and `X.k` are the wavenumbers of its rows and of its columns, a column and
a row on the device of the coefficients, so that they broadcast against the batch:
`im .* X.j .* X` is the derivative of the fields along x. The batches made from `X`, by
`similar`, `copy` and `view`, share them.

# Arguments
- `data`: the coefficients, a complex array `(n+1, 2n+1, B)`, kept, not copied.
- `Ωs`: or a vector of `B` fields of OpenKolmogorovFlow, `FTField{n, m}`, copied to a batch on
  the CPU. `FTField(X, b, m)` gives field `b` of a batch back.
"""
struct FTFields{n, A<:AbstractArray{ComplexF64, 3}, W<:AbstractMatrix{Float64}} <: AbstractArray{ComplexF64, 3}
    data::A # the coefficients, (n+1, 2n+1, B)
       j::W # the wavenumbers along x of the rows, a column (n+1, 1): 0, …, n
       k::W # the wavenumbers along y of the columns, a row (1, 2n+1): 0, …, n, -n, …, -1

    function FTFields(data::AbstractArray{ComplexF64, 3})
        # ---- input checks ----
        size(data, 2) == 2*size(data, 1) - 1 ||
            throw(ArgumentError("the coefficients must have size (n+1, 2n+1, B)"))

        # ---- the largest wavenumber, from the number of rows ----
        n = size(data, 1) - 1

        # ---- the wavenumbers, computed once, on the device of the coefficients, copied there
        #      from vectors of Float64 on the CPU, which every array type takes ----
        j = copyto!(similar(data, Float64, (n + 1, 1)), Float64[0:n;])
        k = copyto!(similar(data, Float64, (1, 2n + 1)), Float64[0:n; -n:-1])

        return new{n, typeof(data), typeof(j)}(data,
                                               j,
                                               k)
    end

    # a batch on other coefficients, with the wavenumbers of the batch X, shared and not copied
    FTFields(data::AbstractArray{ComplexF64, 3}, X::FTFields{n}) where {n} =
        new{n, typeof(data), typeof(X.j)}(data,
                                          X.j,
                                          X.k)
end


# ---------------------------------------------------------------------------- #
# from and to the fields of OpenKolmogorovFlow                                 #
# ---------------------------------------------------------------------------- #

# the column of a batch that holds the wavenumber k along y
_column(k::Int, n::Int) = mod(k, 2n + 1) + 1

function FTFields(Ωs::AbstractVector{<:OpenKolmogorovFlow.FTField{n}}) where {n}
    data = zeros(ComplexF64, n + 1, 2n + 1, length(Ωs))

    # ---- the active modes of every field, each at its row j and at the column of its k ----
    for (b, Ω) in enumerate(Ωs), j in 0:n, k in -n:n
        data[j + 1, _column(k, n), b] = Ω[OpenKolmogorovFlow.WaveNumber(k, j)]
    end

    return FTFields(data)
end

function OpenKolmogorovFlow.FTField(X::FTFields{n}, b::Int, m::Int) where {n}
    # ---- the coefficients of field b, on the CPU ----
    data = Array(view(parent(X), :, :, b))

    # ---- into the active modes of a field of OKF, whose other modes stay zero ----
    Ω = OpenKolmogorovFlow.FTField(n, m)
    for j in 0:n, k in -n:n
        Ω[OpenKolmogorovFlow.WaveNumber(k, j)] = data[j + 1, _column(k, n)]
    end

    return Ω
end


# ---------------------------------------------------------------------------- #
# the batch as an array                                                        #
# ---------------------------------------------------------------------------- #

# the array of the coefficients, and its size
Base.parent(X::FTFields) = X.data
Base.size(X::FTFields)   = size(X.data)

# one coefficient, for the CPU: on the GPU the operations below act on the whole array
Base.@propagate_inbounds Base.getindex(X::FTFields, I::Vararg{Int, 3}) = X.data[I...]
Base.@propagate_inbounds Base.setindex!(X::FTFields, a, I::Vararg{Int, 3}) = (X.data[I...] = a)

# a batch like X and a copy of it, with its wavenumbers
Base.similar(X::FTFields) = FTFields(similar(X.data), X)
Base.copy(X::FTFields)    = FTFields(copy(X.data), X)

# an array of another type or size on the device of X, as the fields on the grid; or, when the
# type and the first two sizes are those of a batch, a batch of fields like those of X with the
# wavenumbers of X, as MultipleShooting makes for the k directions of a subspace step packed
# together
function Base.similar(X::FTFields{n}, ::Type{T}, dims::Dims) where {n, T}
    batch = T === ComplexF64 && length(dims) == 3 && dims[1:2] == (n + 1, 2n + 1)
    batch && return FTFields(similar(X.data, T, dims), X)

    return similar(X.data, T, dims)
end

# the fields b of a batch, as a batch that shares its coefficients. `selectdim`, by which
# MultipleShooting takes the first and the last seed, passes slices for the colons
Base.view(X::FTFields, ::Union{Colon, Base.Slice}, ::Union{Colon, Base.Slice}, b::AbstractUnitRange) =
    FTFields(view(X.data, :, :, b), X)

# the operations that MultipleShooting makes on a batch besides broadcasting, each on the array
# of the coefficients
Base.copyto!(X::FTFields, Y::FTFields) = (copyto!(X.data, Y.data); X)
Base.fill!(X::FTFields, a)             = (fill!(X.data, a); X)
Base.reverse!(X::FTFields; dims)       = (reverse!(X.data; dims=dims); X)
Random.randn!(X::FTFields)             = (Random.randn!(X.data); X)

# broadcasting: a batch in an expression stands for the array of its coefficients, and an
# expression assigned to a batch is written into that array
Base.Broadcast.broadcastable(X::FTFields) = X.data
Base.Broadcast.materialize!(X::FTFields, bc::Base.Broadcast.Broadcasted) =
    (Base.Broadcast.materialize!(X.data, bc); X)

# a broadcast whose leaves are batches, as Flows.jl builds one for each part of a coupled state,
# written into a batch: the same broadcast on the arrays of the coefficients, so that it runs on
# their device and not entry by entry
Base.copyto!(X::FTFields, bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{3}}) =
    (Base.Broadcast.materialize!(X.data, _coefficients(bc)); X)

# the broadcast bc with every batch among its leaves replaced by the array of its coefficients
_coefficients(bc::Base.Broadcast.Broadcasted) = Base.Broadcast.broadcasted(bc.f, map(_coefficients, bc.args)...)
_coefficients(X::FTFields)                    = X.data
_coefficients(x)                              = x


# ---------------------------------------------------------------------------- #
# the symmetry of a field, and the inner product                               #
# ---------------------------------------------------------------------------- #

"""
    apply_symmetry!(X) -> X

Give the slices of the batch `X` the symmetry of a field: in the row j = 0 the entries at k and
-k become conjugates, by their mean, and the mean of every field is set to zero. The identity on
fields. As `_apply_symmetry` and the mask of OpenKolmogorovFlow.
"""
function apply_symmetry!(X::FTFields{n}) where {n}
    data = parent(X)

    # ---- the row j = 0: the entries at k = 1, …, n, and those at -1, …, -n in the same order ----
    pos = view(data, 1, 2:(n + 1), :)
    neg = view(data, 1, (2n + 1):-1:(n + 2), :)

    # ---- each pair replaced by its Hermitian part ----
    #      Ω[0, k] = (Ω[0, k] + conj(Ω[0, -k]))/2,   Ω[0, -k] = conj(Ω[0, k])
    pos .= (pos .+ conj.(neg)) ./ 2
    neg .= conj.(pos)

    # ---- zero mean ----
    view(data, 1, 1, :) .= 0

    return X
end

"""
    realdots(X, Y) -> c

The inner products of the fields of the batches `X` and `Y` one by one, the mean over the domain
of the product of the two vorticity fields,

    c[1, 1, b] = ⟨ω₁ ω₂⟩ = Σⱼ Σₖ conj(Ω₁[j, k]) Ω₂[j, k],   j, k = -n, …, n,

as `dot` of two fields of OpenKolmogorovFlow. `c` is a real array of size `(1, 1, B)` on the
device of `X`, from one reduction: the form that `MultiVector(X, d; dots=realdots)` of
MultipleShooting takes.

# Arguments
- `X`, `Y`: two batches of fields of the same size.
"""
function realdots(X::FTFields{n}, Y::FTFields{n}) where {n}
    # ---- the weight of a row: the modes with j ≥ 1 stand for their conjugates too, and count
    #      twice; the row j = 0 holds both k and -k, and counts once ----
    weight = ifelse.(X.j .== 0, 1, 2)

    # ---- the sum over the modes of every field, the fields kept apart ----
    return sum(weight .* real.(conj.(X) .* Y); dims=(1, 2))
end

"""
    realdot(X, Y) -> Float64

The sum over the fields of the inner products [`realdots`](@ref) of the batches `X` and `Y`.
`realdot(X, X)` is the mean of ω² summed over the fields.
"""
realdot(X::FTFields{n}, Y::FTFields{n}) where {n} = sum(realdots(X, Y))
