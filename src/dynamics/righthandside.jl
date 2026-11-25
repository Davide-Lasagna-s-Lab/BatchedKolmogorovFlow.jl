# The right-hand side of the vorticity equation on batches of fields, the right-hand side F of
# MultipleShooting.jl: the time derivative of every field, for the column of the period and the
# phase condition in time.

"""
    RightHandSide(n, m, Re, kf; B, device=:cpu)
    RightHandSide(F, Re)

The time derivative of the vorticity equation, advection, forcing and viscous term, called as
`F(dX, X)`: the time derivative of every field of the batch `X`, `B` of them, written into `dX`,
and `dX` returned. `X` is first given the symmetry of fields, on a copy, and is left as it is.

Its fields and its term are made with it, so that a call allocates no batch, and a batch of
another number of fields is an error. The second form is `F` at another Reynolds number, which
shares them.

# Arguments
- `n`, `m`, `Re`, `kf`, `B`, `device`: those of the [`NonlinearPropagator`](@ref).
"""
struct RightHandSide{XT<:FTFields, ET<:ExplicitTerm}
          Re::Float64   # the Reynolds number
           B::Int       # the fields of a batch
           X::XT        # the fields, with the symmetry of fields
    explicit::ET        # the advection and the forcing, for batches like X

    function RightHandSide(n::Int,
                           m::Int,
                          Re::Real,
                          kf::Int;
                           B::Int,
                      device::Symbol=:cpu)
        # ---- input checks ----
        2m + 2 >= 3n + 1 ||
            throw(ArgumentError("the grid must dealias the products, 2m + 2 at least 3n + 1"))
        1 <= kf <= n ||
            throw(ArgumentError("the forcing wavenumber must be in 1:n"))
        B >= 1 ||
            throw(ArgumentError("a batch has B ≥ 1 fields"))
        device in (:cpu, :gpu) ||
            throw(ArgumentError("device must be :cpu or :gpu"))

        # ---- the fields, and the explicit term for batches like them ----
        X        = _batch(n, B, device)
        explicit = ExplicitTerm(X, m, kf)

        return new{typeof(X), typeof(explicit)}(Float64(Re),
                                                B,
                                                X,
                                                explicit)
    end

    # F at another Reynolds number, with its fields and its term, shared
    RightHandSide(F::RightHandSide{XT, ET}, Re::Real) where {XT, ET} =
        new{XT, ET}(Float64(Re),
                    F.B,
                    F.X,
                    F.explicit)
end

function (F::RightHandSide)(dX::FTFields, X::FTFields)
    # ---- input checks ----
    size(X, 3) == F.B ||
        throw(ArgumentError("a right-hand side of $(F.B) fields called with $(size(X, 3))"))

    # ---- the fields, with the symmetry of fields ----
    Ω = apply_symmetry!(copyto!(F.X, X))

    # ---- advection and forcing ----
    F.explicit(0.0, Ω, dX)

    # ---- plus the viscous term, Δω/Re ----
    dX .-= (Ω.j.^2 .+ Ω.k.^2) .* Ω ./ F.Re

    return dX
end
