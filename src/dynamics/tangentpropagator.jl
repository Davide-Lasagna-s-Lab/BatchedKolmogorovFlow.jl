# The flow of Kolmogorov flow with its tangent, the tangent L of MultipleShooting.jl, for the
# parts of a search that advance perturbations: the Jacobian, one perturbation of every field, and
# the subspace iteration and its preconditioner, k′ perturbations of every field. Forced by the
# derivative of the right-hand side in the Reynolds number, it is the forced tangent Lp of the
# sensitivity and of the column of the parameter. Every field is advanced with its perturbations
# in one batch, the bundle of linearised.jl.

"""
    TangentPropagator(n, m, Re, kf; B, k′=1, forced=false, dt=0.01, scheme=:CB3R2R3e,
                      device=:cpu, tasks=1)
    TangentPropagator(L, Re)

The flow of two-dimensional Kolmogorov flow on batches of `B` vorticity fields with its tangent,
`k′` perturbations of every field, called as `L(X, V, τ)`: the fields `X` and the perturbations
`V` advanced in place by `τ` together, each perturbation by the tangent equation about its field,
with the scheme and the steps of the [`NonlinearPropagator`](@ref), so that `V` becomes the
derivative of the discrete flow applied to `V`, and `V` returned.

`V` is a batch like `X`, one perturbation of every field, for a product of the Jacobian of
MultipleShooting.jl, with `k′ = 1`, or a vector of `k′` such batches, the directions of its
subspace step. Every field goes with its `k′` perturbations in one batch, and is advanced once for
all of them, so that the `k′` directions cost one integration of the equation and `k′` of its
tangent. Batch by batch the result is that of the propagator of one batch, to rounding. The
fields and the perturbations are first given the symmetry of fields, and may be views of batches.

With `forced = true` the tangent equation is forced by the derivative of the right-hand side in the
Reynolds number, `-Δω/Re²`, the forced tangent `Lp` of MultipleShooting.jl: from `V = 0` it gives
the derivative of the end points in the Reynolds number. The scheme treats the forcing as an
explicit term, so that this is the derivative of the flow up to the error of the scheme.

The workspaces are made with the propagator, so that a call allocates no batch, and a call of
another size is an error. The second form is the propagator `L` at another Reynolds number, which
shares its workspaces and costs nothing.

# Arguments
- `n`, `m`, `Re`, `kf`, `B`, `dt`, `scheme`, `device`, `tasks`: those of the
  [`NonlinearPropagator`](@ref). The tasks advance each its chunk of fields with their
  perturbations.
- `k′::Int=1`: the perturbations of every field.
- `forced::Bool=false`: whether the tangent equation is forced by `-Δω/Re²`.
"""
struct TangentPropagator{K}
        Re::Float64     # the Reynolds number
        dt::Float64     # the time step
         B::Int         # the fields of a batch
        k′::Int         # the perturbations of every field
    forced::Bool        # whether the tangent equation is forced by -Δω/Re²
    chunks::Vector{K}   # the chunks of the fields, one per task, each with its flow

    # A chunk (chunks.jl) is a named tuple (fields, Z, ϕ): the range of its fields, the bundle of
    # as many fields with their k′ perturbations (linearised.jl), and the flow of the tangent
    # equation of the bundle, forced or not, made once, with its term and the registers of its
    # scheme.
    function TangentPropagator(n::Int,
                               m::Int,
                              Re::Real,
                              kf::Int;
                               B::Int,
                              k′::Int=1,
                          forced::Bool=false,
                              dt::Real=0.01,
                          scheme::Symbol=:CB3R2R3e,
                          device::Symbol=:cpu,
                           tasks::Int=1)
        # ---- input checks ----
        _check(n, m, B, device; kf=kf, scheme=scheme)
        k′ >= 1 ||
            throw(ArgumentError("a tangent propagator has k′ ≥ 1 perturbations of every field"))

        # ---- the chunk of the fields `fields`: the bundle of as many fields with their k′
        #      perturbations, and the flow of the tangent equation that advances it, forced or
        #      not ----
        function chunk(fields)
            Z    = FTFields(n, (1 + k′)*length(fields); device=device)
            term = TangentTerm(Z, k′, m, kf)
            ϕ    = Flows.flow(forced ? ForcedTangentTerm(term, Re) : term,
                              ImplicitTerm(Z, Re),
                              Flows.imex_schemes[scheme](Z),
                              Flows.TimeStepConstant(dt))
            return (fields = fields,
                    Z      = Z,
                    ϕ      = ϕ)
        end

        # ---- the chunks of the tasks ----
        chunks = [chunk(fields) for fields in _ranges(B, tasks)]

        return new{eltype(chunks)}(Float64(Re),
                                   Float64(dt),
                                   B,
                                   k′,
                                   forced,
                                   chunks)
    end

    # the propagator L at another Reynolds number, with its bundles, terms and registers, shared,
    # and its flows made again there
    TangentPropagator(L::TangentPropagator{K}, Re::Real) where {K} =
        new{K}(Float64(Re),
               L.dt,
               L.B,
               L.k′,
               L.forced,
               [(c..., ϕ = _reynolds(c.ϕ, Re)) for c in L.chunks])
end

(L::TangentPropagator)(X::FTFields, V::FTFields, τ::Real) = (_advance!(L, X, (V,), τ); V)

(L::TangentPropagator)(X::FTFields, V::AbstractVector{<:FTFields}, τ::Real) = _advance!(L, X, V, τ)

# the fields X advanced by τ, and the perturbations V with them, k′ batches of as many as the
# fields, a vector or a tuple, by the tangent equation, forced or not
function _advance!(L::TangentPropagator, X::FTFields, V, τ::Real)
    # ---- input checks ----
    (size(X, 3), length(V)) == (L.B, L.k′) ||
        throw(ArgumentError("a propagator of $(L.B) fields with $(L.k′) perturbations each, " *
                            "called with $(size(X, 3)) fields with $(length(V))"))
    all(v -> size(v, 3) == size(X, 3), V) ||
        throw(ArgumentError("every batch of perturbations must have as many as the fields"))

    # ---- every task advances its chunk of the fields, each with its perturbations: with one
    #      task, the whole batch ----
    _foreach_chunk(L.chunks) do chunk
        # ---- every field of the chunk with its perturbations copied into the bundle of its
        #      flow, with the symmetry of fields: the flow advances the fields and their
        #      perturbations as one batch, which the fields of the call and the batches of their
        #      perturbations are not, and works on the bundle it was made for, as the flow of the
        #      NonlinearPropagator does ----
        apply_symmetry!(_bundle!(chunk.Z, X, V, chunk.fields))

        # ---- advanced together, the fields once, and back ----
        chunk.ϕ(chunk.Z, (0, τ))
        _unbundle!(X, V, chunk.Z, chunk.fields)
    end

    return V
end
