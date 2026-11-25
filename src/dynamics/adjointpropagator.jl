# The transpose of the tangent of the discrete flow, the adjoint, back over the stages that a
# NonlinearPropagator keeps (nonlinearpropagator.jl), by the adjoint of the scheme of Flows.jl.

"""
    AdjointPropagator(n, m, Re; B, scheme=:CB3R2R3e, device=:cpu)
    AdjointPropagator(A, Re)

The transpose of the tangent in the inner product of the fields, [`dots`](@ref), called as
`A(W, S)` over the stages `S` of [`stages`](@ref): `W` becomes the transpose of the derivative of
the discrete flow, over the interval and from the fields of `S`, applied to `W`, and is returned.
For any batches `V` and `W`, field by field and to rounding,

    dots(L(copy(X), V, τ), W) == dots(V, A(W, stages(G, X, τ)))

with `G` and `L` the [`NonlinearPropagator`](@ref) and the [`TangentPropagator`](@ref) of the same
flow. The adjoint scheme runs backwards over `S`, from the last step to the first, and leaves `S`
as it is, so that one forward pass serves every `W`: a call costs one backward pass. `W` has as
many fields as the batch that made `S`, `B`, each about the field in the same place, and is first
given the symmetry of fields, as the tangent gives it to its input, so that the transpose
includes that projection. `W` may be a view of a batch.

Its fields and its flow are made with it. The second form is `A` at another Reynolds number,
which shares the fields, the term and the registers of its flow.

# Arguments
- `n`, `m`, `Re`, `B`, `device`: those of the [`NonlinearPropagator`](@ref).
- `scheme::Symbol=:CB3R2R3e`: the scheme of the flow that made the stages, one of
  `Flows.imex_schemes` with a discrete adjoint.
"""
struct AdjointPropagator{WT<:FTFields, ΨT<:Flows.Flow}
    Re::Float64     # the Reynolds number
     B::Int         # the fields of a batch
     W::WT          # the adjoint fields
     ψ::ΨT          # the flow of the adjoint equation back over stages, for batches like W, made
                    # once, with its term and the registers of the scheme in the discrete mode

    function AdjointPropagator(n::Int,
                               m::Int,
                              Re::Real;
                               B::Int,
                          scheme::Symbol=:CB3R2R3e,
                          device::Symbol=:cpu)
        # ---- input checks ----
        2m + 2 >= 3n + 1 ||
            throw(ArgumentError("the grid must dealias the products, 2m + 2 at least 3n + 1"))
        B >= 1 ||
            throw(ArgumentError("a batch has B ≥ 1 fields"))
        device in (:cpu, :gpu) ||
            throw(ArgumentError("device must be :cpu or :gpu"))
        haskey(Flows.imex_schemes, scheme) ||
            throw(ArgumentError("scheme must be one of Flows.imex_schemes, " *
                                "$(sort(collect(keys(Flows.imex_schemes))))"))

        # ---- the adjoint fields, the adjoint term for batches like them, and the registers of
        #      the scheme in the mode of the discrete adjoint ----
        W = _batch(n, B, device)
        ψ = Flows.flow(AdjointTerm(W, m),
                       ImplicitTerm(Re),
                       Flows.imex_schemes[scheme](W, Flows.DiscreteMode(true)),
                       Flows.TimeStepFromCache())

        return new{typeof(W), typeof(ψ)}(Float64(Re),
                                         B,
                                         W,
                                         ψ)
    end

    # A at another Reynolds number, with its fields, its term and its registers, shared, and its
    # flow made again there
    AdjointPropagator(A::AdjointPropagator{WT, ΨT}, Re::Real) where {WT, ΨT} =
        new{WT, ΨT}(Float64(Re),
                    A.B,
                    A.W,
                    _reynolds(A.ψ, Re))
end

function (A::AdjointPropagator)(W::FTFields, S::Flows.RAMStageCache)
    # ---- input checks ----
    size(W, 3) == A.B ||
        throw(ArgumentError("an adjoint propagator of $(A.B) fields called with $(size(W, 3))"))

    # ---- the adjoint fields in those of the flow, with the symmetry of fields ----
    apply_symmetry!(copyto!(A.W, W))

    # ---- back over the stages, from the last step to the first ----
    A.ψ(A.W, S)

    return copyto!(W, A.W)
end
