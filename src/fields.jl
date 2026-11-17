# A batch of fields on the grid.
#
# On the grid a field is its values at the points
#
#     x_p = 2π p/(2m+2),   y_q = 2π q/(2m+2),   p, q = 0, …, 2m+1.
#
# A batch holds B fields on the same grid, in one real array of size (2m+2, 2m+2, B): field b is
# the slice [:, :, b], with x down the rows and y across the columns. This is the Field{m} of
# OpenKolmogorovFlow, transposed, as the batches in Fourier space of ftfields.jl are. As there,
# the size of the grid is a parameter of the type.

"""
    Fields(data)
    Fields(X, m)

A batch of `B` fields on the grid of `2m+2` points along each direction, with `m` in its type,
`Fields{m}`. It is an array of size `(2m+2, 2m+2, B)`: `u[p+1, q+1, b]` is the value of field
`b` at `x = 2π p/(2m+2)` and `y = 2π q/(2m+2)`. It wraps the array of its values, on the CPU or
on the GPU, which `parent` returns, and broadcasts as that array does, so that `u .* v` is the
product of the fields of two batches, point by point.

# Arguments
- `data`: the values, a real array `(2m+2, 2m+2, B)`, kept, not copied.
- `X::FTFields`, `m::Int`: or a batch in Fourier space and the size of the grid, for a batch of
  as many fields on the device of `X`, its values not set.
"""
struct Fields{m, A<:AbstractArray{Float64, 3}} <: AbstractArray{Float64, 3}
    data::A # the values on the grid, (2m+2, 2m+2, B)

    function Fields(data::AbstractArray{Float64, 3})
        # ---- input checks ----
        (size(data, 1) == size(data, 2) && iseven(size(data, 1))) ||
            throw(ArgumentError("the values must have size (2m+2, 2m+2, B)"))

        # ---- the size of the grid, from the number of rows ----
        m = size(data, 1) >> 1 - 1

        return new{m, typeof(data)}(data)
    end
end

# as many fields as the batch X has, on the grid of 2m+2 points and on the device of X
Fields(X::FTFields, m::Int) = Fields(similar(X, Float64, (2m + 2, 2m + 2, size(X, 3))))


# ---------------------------------------------------------------------------- #
# the batch as an array                                                        #
# ---------------------------------------------------------------------------- #

# the array of the values, and its size
Base.parent(u::Fields) = u.data
Base.size(u::Fields)   = size(u.data)

# one value, for the CPU: on the GPU the operations act on the whole array
Base.@propagate_inbounds Base.getindex(u::Fields, I::Vararg{Int, 3}) = u.data[I...]
Base.@propagate_inbounds Base.setindex!(u::Fields, a, I::Vararg{Int, 3}) = (u.data[I...] = a)

# a batch like u and a copy of it
Base.similar(u::Fields) = Fields(similar(u.data))
Base.copy(u::Fields)    = Fields(copy(u.data))

# broadcasting: a batch in an expression stands for the array of its values, and an expression
# assigned to a batch is written into that array
Base.Broadcast.broadcastable(u::Fields) = u.data
Base.Broadcast.materialize!(u::Fields, bc::Base.Broadcast.Broadcasted) =
    (Base.Broadcast.materialize!(u.data, bc); u)
