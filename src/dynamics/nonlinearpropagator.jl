# The flow of Kolmogorov flow on batches of fields, the flow G of MultipleShooting.jl: all the
# fields of a batch advanced together by an implicit–explicit scheme of Flows.jl with constant
# steps, CB3R2R3e unless another is asked for, the viscous term implicit.
#
# The objects of the dynamics of a search, this flow, the tangent (tangentpropagator.jl), the
# right-hand side (righthandside.jl) and the shift (xshift.jl), are callables, each with its task
# and the workspaces of that task, made with it, so that a call allocates no batch. The adjoint
# (adjointpropagator.jl) is one more, which a search does not call. They share the arrays of a
# device (devices.jl), and the propagators the chunks of their tasks on the CPU (chunks.jl).

"""
    NonlinearPropagator(n, m, Re, kf; B, dt=0.01, scheme=:CB3R2R3e, device=:cpu, tasks=1)
    NonlinearPropagator(G, Re)

The flow of two-dimensional Kolmogorov flow on batches of `B` vorticity fields, called as
`G(X, τ)`: every field of the batch `X` advanced in place by the time `τ`, all together, or chunk
by chunk on the tasks, and `X` returned. The batch is first given the symmetry of fields, by
`apply_symmetry!`, and may be a view of a batch. A batch is an [`FTFields`](@ref), the Fourier
coefficients of `B` fields, on the CPU or on the GPU. Field by field the results are those of
OpenKolmogorovFlow with the same scheme and steps, to rounding. [`stages`](@ref) keeps the stages
of the steps, for an [`AdjointPropagator`](@ref).

The propagator holds the workspaces of batches of `B` fields, made with it, so that a call
allocates no batch. A batch of another number of fields is an error. The second form is the
propagator `G` at another Reynolds number, which shares its workspaces and costs nothing. Two
propagators that share their workspaces must not run at the same time.

# Arguments
- `n::Int`: the largest wavenumber of the fields.
- `m::Int`: the grid of the products has `2m+2` points along each direction, at least `3n + 1`,
  which keeps them free of aliasing.
- `Re::Real`: the Reynolds number.
- `kf::Int`: the wavenumber of the forcing, from 1 to `n`.
- `B::Int`: the fields of a batch, required.
- `dt::Real=0.01`: the time step. A time that is not a whole number of steps ends with a shorter
  one.
- `scheme::Symbol=:CB3R2R3e`: the implicit–explicit scheme of Flows.jl, one of
  `Flows.imex_schemes`, `:CNRK2`, `:CB3R2R2`, `:CB3R2R3e`, `:CB3R2R3c` or `:CB4R3R4`, of two,
  three, four, four and six stages per step and of order two, two, three, three and four. The
  exponential schemes of `Flows.imex_schemes` need the coefficients of their linear part, which
  the viscous term does not give yet.
- `device::Symbol=:cpu`: where the batches live, `:cpu`, on `Array`s, or `:gpu`, on `CuArray`s,
  once CUDA is loaded, `using CUDA`, which is not a dependency of the package.
- `tasks::Int=1`: how many tasks advance a batch on the CPU. An operation on an array runs on
  one core, so that one task advances a whole batch on one core. With more tasks the batch is cut
  into as many chunks of fields, and every task advances its chunk as a batch of its own, on its
  own thread, while the others advance theirs. The fields of a batch do not act on one another,
  so that the result is that of one task, to rounding. Start Julia with at least as many threads,
  `julia -t 16` for `tasks=16`.
"""
struct NonlinearPropagator{K, E}
         n::Int         # the largest wavenumber of the fields
         m::Int         # the grid has 2m+2 points along each direction
        Re::Float64     # the Reynolds number
        kf::Int         # the wavenumber of the forcing
        dt::Float64     # the time step
    scheme::Symbol      # the scheme of Flows.jl
         B::Int         # the fields of a batch
    chunks::Vector{K}   # the chunks of the fields, one per task, each with its flow
     whole::E           # the fields and the flow of the whole batch, of stages, on one task

    # A chunk (chunks.jl) is a named tuple (fields, X, ϕ): the range of its fields, a batch of as
    # many, and the flow of the equation of the chunk, made once, with its explicit term and the
    # registers of its scheme. The whole batch is a named tuple (X, ϕ).
    function NonlinearPropagator(n::Int,
                                 m::Int,
                                Re::Real,
                                kf::Int;
                                 B::Int,
                                dt::Real=0.01,
                            scheme::Symbol=:CB3R2R3e,
                            device::Symbol=:cpu,
                             tasks::Int=1)
        # ---- input checks ----
        2m + 2 >= 3n + 1 ||
            throw(ArgumentError("the grid must dealias the products, 2m + 2 at least 3n + 1"))
        1 <= kf <= n ||
            throw(ArgumentError("the forcing wavenumber must be in 1:n"))
        B >= 1 ||
            throw(ArgumentError("a batch has B ≥ 1 fields"))
        device in (:cpu, :gpu) ||
            throw(ArgumentError("device must be :cpu or :gpu"))
        haskey(Flows.imex_schemes, scheme) ||
            throw(ArgumentError("scheme must be one of Flows.imex_schemes, " *
                                "$(sort(collect(keys(Flows.imex_schemes))))"))

        # ---- b fields, and the flow of the equation for batches like them ----
        function equation(b)
            X = _batch(n, b, device)
            ϕ = Flows.flow(ExplicitTerm(X, m, kf),
                           ImplicitTerm(Re),
                           Flows.imex_schemes[scheme](X),
                           Flows.TimeStepConstant(dt))
            return (X = X,
                    ϕ = ϕ)
        end

        # ---- the chunks of the tasks, each with its fields and its flow ----
        chunks = [(fields = fields, equation(length(fields))...) for fields in _ranges(B, tasks)]

        # ---- the whole batch, for stages, which runs on one task ----
        whole = length(chunks) == 1 ? (X = chunks[1].X, ϕ = chunks[1].ϕ) : equation(B)

        return new{eltype(chunks), typeof(whole)}(n,
                                                m,
                                                Float64(Re),
                                                kf,
                                                Float64(dt),
                                                scheme,
                                                B,
                                                chunks,
                                                whole)
    end

    # the propagator G at another Reynolds number, with its fields, terms and registers, shared,
    # and its flows made again there
    NonlinearPropagator(G::NonlinearPropagator{K, E}, Re::Real) where {K, E} =
        new{K, E}(G.n,
                  G.m,
                  Float64(Re),
                  G.kf,
                  G.dt,
                  G.scheme,
                  G.B,
                  [(c..., ϕ = _reynolds(c.ϕ, Re)) for c in G.chunks],
                  (G.whole..., ϕ = _reynolds(G.whole.ϕ, Re)))
end

function (G::NonlinearPropagator)(X::FTFields, τ::Real)
    # ---- input checks ----
    size(X, 3) == G.B ||
        throw(ArgumentError("a propagator of $(G.B) fields called with $(size(X, 3))"))

    # ---- every task advances its chunk of the fields: with one task, the whole batch ----
    _foreach_chunk(G.chunks) do chunk
        Xc = view(X, :, :, chunk.fields)

        # ---- the fields of the chunk in those of its flow, with the symmetry of fields ----
        apply_symmetry!(copyto!(chunk.X, Xc))

        # ---- advanced, and back ----
        chunk.ϕ(chunk.X, (0, τ))
        copyto!(Xc, chunk.X)
    end

    return X
end

"""
    stages(G::NonlinearPropagator, X, τ) -> S

The stages of every step of the flow `G` over `τ` from the fields `X`, kept for an
[`AdjointPropagator`](@ref), which goes back over them. `X` holds the fields at the start of the
interval and is left as it is. The steps are those of `G(X, τ)`. The stages are made once and
serve any number of calls of the adjoint.

`S` holds the stages of each of the `N` steps over `τ`, four with the default scheme, `4N`
batches like `X`, in the stage cache of Flows.jl. For one field with `n = 33` and steps of 0.01
this is 14.6 GB over 1000 time units, whatever the number of segments the orbit is cut into.
"""
function stages(G::NonlinearPropagator, X::FTFields, τ::Real)
    # ---- input checks ----
    size(X, 3) == G.B ||
        throw(ArgumentError("a propagator of $(G.B) fields called with $(size(X, 3))"))

    whole = G.whole

    # ---- the fields in those of the flow, with the symmetry of fields ----
    apply_symmetry!(copyto!(whole.X, X))

    # ---- advanced, the stages of every step kept, in copies that Flows.jl makes ----
    S = Flows.RAMStageCache(Flows.nstages(whole.ϕ.meth), whole.X)
    whole.ϕ(whole.X, (0, τ), S)

    return S
end
