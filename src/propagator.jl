# The propagator of Kolmogorov flow for MultipleShooting.jl: the methods of its interface on
# batches of fields, all the fields of a batch advanced together by an implicit–explicit scheme
# of Flows.jl with constant steps, CB3R2R3e unless another is asked for, the viscous term
# implicit, and the transpose of its tangent, the adjoint of the discrete flow.
#
# The file has two parts. First the propagator and the methods of the interface, which are what a
# search calls. Then what serves them: the tasks of a propagator on the CPU, its storage, and the
# advance of fields with their perturbations.

"""
    KolmogorovPropagator(n, m, Re, kf; dt=0.01, scheme=:CB3R2R3e, array=Array, tasks=1)
    KolmogorovPropagator(P, Re)

The flow of two-dimensional Kolmogorov flow on batches of vorticity fields, for the searches of
MultipleShooting.jl: [`propagate!`](@ref), [`tangent!`](@ref), [`forced!`](@ref), [`rhs!`](@ref), [`shift!`](@ref) and [`dshift!`](@ref), and the transpose of
the tangent, [`adjoint!`](@ref), over the stages kept by [`stages`](@ref). A batch is an
[`FTFields`](@ref) on an array of type `array`, the Fourier coefficients of `B` fields. Field by
field the results are those of OpenKolmogorovFlow with the same scheme and steps, to rounding.

The storage of the batches of `B` fields, the terms with their plans and arrays and the registers
of the scheme, lives in the propagator. It is built once, at the first call with `B` fields, and
every later call with `B` fields only looks it up, so that a call allocates no batch, but for the
stages that [`stages`](@ref) returns. What the tangent equation adds for `K` perturbations of `B`
fields, and what `adjoint!` adds, are built at their own first calls, so that a search that never
calls them holds none of it.

The second form is the propagator of `P` at another Reynolds number. It shares the terms, the
plans and the storage of `P`, and costs nothing, so that `Re -> KolmogorovPropagator(P, Re)` is
the family of propagators that the searches with a free parameter of MultipleShooting.jl take.
Two propagators that share their storage must not run at the same time.

# Arguments
- `n::Int`: the largest wavenumber of the fields.
- `m::Int`: the grid of the products has `2m+2` points along each direction, at least `3n + 1`,
  which keeps them free of aliasing.
- `Re::Real`: the Reynolds number.
- `kf::Int`: the wavenumber of the forcing, from 1 to `n`.
- `dt::Real=0.01`: the time step. A time that is not a whole number of steps ends with a shorter
  one.
- `scheme::Symbol=:CB3R2R3e`: the implicit–explicit scheme of Flows.jl, `:CNRK2`, `:CB3R2R2`,
  `:CB3R2R3e`, `:CB3R2R3c` or `:CB4R3R4`, of two, three, four, four and six stages per step and
  of order two, two, three, three and four. Flows.jl has no adjoint of `:CB4R3R4`, so that
  [`adjoint!`](@ref) is not available with it.
- `array::Type=Array`: the type of the batches, `Array`, or `CuArray` once CUDA is loaded.
- `tasks::Int=1`: how many tasks advance a batch on the CPU. An operation on an array runs on
  one core, so that one task advances a whole batch on one core. With more tasks the batch is cut
  into as many chunks of fields, and every task advances its chunk as a batch of its own, on its
  own thread, while the others advance theirs. The fields of a batch do not act on one another,
  so that the result is that of one task, to rounding. This is how [`propagate!`](@ref),
  [`tangent!`](@ref) and [`forced!`](@ref) use the cores of a CPU. The other
  methods run on one task. Start Julia with at least as many threads, `julia -t 16` for
  `tasks=16`.
"""
struct KolmogorovPropagator{A<:AbstractArray}
           n::Int            # the largest wavenumber of the fields
           m::Int            # the grid has 2m+2 points along each direction
          Re::Float64        # the Reynolds number
          kf::Int            # the wavenumber of the forcing
          dt::Float64        # the time step
      scheme::Symbol         # the scheme of Flows.jl
    template::A              # an empty array of the type of the batches
       cache::Vector{Dict{Int, Any}} # one entry per task: the storage of its batches of B fields,
                             # under the key B, built at the first call with B fields and looked
                             # up by the later ones, shared with the propagators at other
                             # Reynolds numbers
    tangents::Vector{Dict{Tuple{Int, Int}, Any}} # one entry per task: what the tangent equation
                             # adds to it for K perturbations of B fields, under the key (B, K),
                             # shared likewise
    adjoints::Dict{Int, Any} # what adjoint! adds to the storage of the first task, under the key
                             # B, built at its first call with B fields, shared in the same way

    function KolmogorovPropagator(n::Int,
                                  m::Int,
                                 Re::Real,
                                 kf::Int;
                                 dt::Real=0.01,
                             scheme::Symbol=:CB3R2R3e,
                              array::Type=Array,
                              tasks::Int=1)
        # ---- input checks ----
        2m + 2 >= 3n + 1 ||
            throw(ArgumentError("the grid must dealias the products, 2m + 2 at least 3n + 1"))
        1 <= kf <= n ||
            throw(ArgumentError("the forcing wavenumber must be in 1:n"))
        scheme in (:CNRK2, :CB3R2R2, :CB3R2R3e, :CB3R2R3c, :CB4R3R4) ||
            throw(ArgumentError("scheme must be :CNRK2, :CB3R2R2, :CB3R2R3e, :CB3R2R3c or :CB4R3R4"))

        # ---- an empty array of the type of the batches, from which their storage is made ----
        template = array{ComplexF64}(undef, 0, 0, 0)

        return new{typeof(template)}(n,
                                     m,
                                     Float64(Re),
                                     kf,
                                     Float64(dt),
                                     scheme,
                                     template,
                                     [Dict{Int, Any}() for _ in 1:tasks],
                                     [Dict{Tuple{Int, Int}, Any}() for _ in 1:tasks],
                                     Dict{Int, Any}())
    end

    # the propagator of P at another Reynolds number, with its terms and its storage, shared
    KolmogorovPropagator(P::KolmogorovPropagator{A}, Re::Real) where {A} =
        new{A}(P.n,
               P.m,
               Float64(Re),
               P.kf,
               P.dt,
               P.scheme,
               P.template,
               P.cache,
               P.tangents,
               P.adjoints)
end


# ---------------------------------------------------------------------------- #
# the interface of MultipleShooting.jl                                         #
# ---------------------------------------------------------------------------- #

"""
    propagate!(P::KolmogorovPropagator, X, τ) -> X

Advance in place every field of the batch `X` by the time `τ`, all together, or chunk by chunk
on the tasks of the propagator. The batch is first given the symmetry of fields, by
`apply_symmetry!`. `X` may be a view of a batch.
"""
function propagate!(P::KolmogorovPropagator, X::FTFields, τ::Real)
    # ---- every task advances its chunk of the fields: with one task, the whole batch ----
    _foreach_chunk(P, size(X, 3)) do c, fields
        storage = _storage(P, length(fields), c)
        chunk   = view(X, :, :, fields)

        # ---- the flow of the equation, at the Reynolds number of P ----
        ϕ = Flows.flow(storage.explicit,
                       ImplicitTerm(P.Re),
                       storage.single,
                       Flows.TimeStepConstant(P.dt))

        # ---- the fields of the chunk in the storage of the flow, with the symmetry of fields ----
        apply_symmetry!(copyto!(storage.X, chunk))

        # ---- advanced, and back ----
        ϕ(storage.X, (0, τ))
        copyto!(chunk, storage.X)
    end

    return X
end

"""
    tangent!(P::KolmogorovPropagator, X, V, τ) -> V

Advance in place by `τ` the fields `X` and the perturbations `V` together, each perturbation by
the tangent equation about its field, with the scheme and the steps of [`propagate!`](@ref), so
that `V` becomes the derivative of the discrete flow applied to `V`.

`V` is a batch like `X`, one perturbation of every field, for a product of the Jacobian of
MultipleShooting.jl, or a vector of k′ such batches, the directions of its subspace step. Every
field goes with its k′ perturbations in one batch, and is advanced once for all of them, so that
the k′ directions cost one integration of the equation and k′ of its tangent. Batch by batch the
result is that of `tangent!` on the batch alone, to rounding.
"""
tangent!(P::KolmogorovPropagator, X::FTFields, V::FTFields, τ::Real) =
    (_advance!(P, X, (V,), τ); V)

tangent!(P::KolmogorovPropagator, X::FTFields, V::AbstractVector{<:FTFields}, τ::Real) =
    _advance!(P, X, V, τ)

"""
    forced!(P::KolmogorovPropagator, X, V, τ) -> V

As [`tangent!`](@ref), with the tangent equation forced by the derivative of the right-hand side
in the Reynolds number, `-Δω/Re²`. From `V = 0` it gives the derivative of the end points in the
Reynolds number, the column of the parameter of MultipleShooting.jl. The scheme treats the
forcing as an explicit term, so that this is the derivative of the flow up to the error of the
scheme, not that of the discrete flow to rounding.
"""
forced!(P::KolmogorovPropagator, X::FTFields, V::FTFields, τ::Real) =
    (_advance!(P, X, (V,), τ, term -> ForcedTangentTerm(term, P.Re)); V)

"""
    stages(P::KolmogorovPropagator, X, τ) -> S

The stages of every step of the flow over `τ` from the fields `X`, kept for [`adjoint!`](@ref),
which goes back over them. `X` holds the fields at the start of the interval and is left as it
is. The steps are those of [`propagate!`](@ref). The stages are made once and serve any number of
calls of `adjoint!`, with the propagator that made them.

`S` holds the stages of each of the `N` steps over `τ`, four with the default scheme, `4N`
batches like `X`, in the stage cache of Flows.jl. For one field with `n = 33` and steps of 0.01
this is 14.6 GB over 1000 time units, whatever the number of segments the orbit is cut into.
"""
function stages(P::KolmogorovPropagator, X::FTFields, τ::Real)
    storage = _storage(P, size(X, 3))

    # ---- the flow of the equation, at the Reynolds number of P ----
    ϕ = Flows.flow(storage.explicit,
                   ImplicitTerm(P.Re),
                   storage.single,
                   Flows.TimeStepConstant(P.dt))

    # ---- the fields in the storage of the flow, with the symmetry of fields ----
    apply_symmetry!(copyto!(storage.X, X))

    # ---- advanced, the stages of every step kept, in copies that Flows.jl makes ----
    S = Flows.RAMStageCache(Flows.nstages(storage.single), storage.X)
    ϕ(storage.X, (0, τ), S)

    return S
end

"""
    adjoint!(P::KolmogorovPropagator, S, W) -> W

The transpose of [`tangent!`](@ref) in the inner product of the fields, [`realdots`](@ref), over
the stages `S` of [`stages`](@ref): `W` becomes the transpose of the derivative of the discrete
flow, over the interval and from the fields of `S`, applied to `W`. For any batches `V` and `W`,
field by field and to rounding,

    realdots(tangent!(P, copy(X), V, τ), W) == realdots(V, adjoint!(P, stages(P, X, τ), W))

The adjoint scheme runs backwards over `S`, from the last step to the first, and leaves `S` as
it is, so that one forward pass serves every `W`: a call costs one backward pass. `W` has as many
fields as the batch that made `S`, each about the field in the same place, and is first given the
symmetry of fields, as `tangent!` gives it to its input, so that the transpose includes that
projection. `W` may be a view of a batch. This is a method of `LinearAlgebra.adjoint!`.
"""
function LinearAlgebra.adjoint!(P::KolmogorovPropagator, S::Flows.RAMStageCache, W::FTFields)
    # ---- input checks ----
    P.scheme === :CB4R3R4 &&
        throw(ArgumentError("Flows.jl has no adjoint of the scheme CB4R3R4"))

    storage = _storage(P, size(W, 3))
    back    = _adjoint_storage(P, size(W, 3))

    # ---- the flow of the adjoint equation back over the stages, at the Reynolds number of P ----
    ψ = Flows.flow(back.term,
                   ImplicitTerm(P.Re),
                   back.scheme,
                   Flows.TimeStepFromCache())

    # ---- the adjoint fields in the storage of the flow, with the symmetry of fields ----
    apply_symmetry!(copyto!(storage.V, W))

    # ---- back over the stages, from the last step to the first ----
    ψ(storage.V, S)

    return copyto!(W, storage.V)
end

"""
    rhs!(P::KolmogorovPropagator, F, X) -> F

The time derivative of every field of the batch `X`, advection, forcing and viscous term, written
into `F`.
"""
function rhs!(P::KolmogorovPropagator, F::FTFields, X::FTFields)
    storage = _storage(P, size(X, 3))

    # ---- the fields in the storage of the propagator, with the symmetry of fields ----
    Ω = apply_symmetry!(copyto!(storage.X, X))

    # ---- advection and forcing ----
    storage.explicit(0.0, Ω, storage.V)

    # ---- plus the viscous term, Δω/Re ----
    storage.V .-= (Ω.j.^2 .+ Ω.k.^2) .* Ω ./ P.Re

    return copyto!(F, storage.V)
end

"""
    shift!(P::KolmogorovPropagator, X, s) -> X

Shift in place every field of the batch `X` along x, `ω(x, y) → ω(x + s, y)`, the coefficients
times `cis(j s)`. `s` is the shift, or the vector of the shifts of MultipleShooting.jl, which
holds this one alone.
"""
function shift!(P::KolmogorovPropagator, X::FTFields, s)
    # ---- e^{i j (x + s)} is e^{i j s} times e^{i j x} ----
    X .*= cis.(X.j .* first(s))

    return X
end

"""
    dshift!(P::KolmogorovPropagator, G, X, i=1) -> G

The generator of the shift along x at every field of the batch `X`, `∂ω/∂x`, written into `G`.
`i` is the index of the shift, as MultipleShooting.jl passes it.
"""
dshift!(P::KolmogorovPropagator, G::FTFields, X::FTFields, i::Int=1) = ddx!(G, X)


# ---------------------------------------------------------------------------- #
# the tasks of a propagator, on the CPU                                        #
# ---------------------------------------------------------------------------- #

# On a GPU one call advances a whole batch, and the device works on all its fields at once. On a
# CPU an operation on an array runs on one core, so that a batch, however large, uses one core.
# To use the others, a propagator made with `tasks` greater than one cuts a batch into as many
# chunks, ranges of whole fields of sizes as equal as can be, and gives one to each task. A batch
# of 10 fields with 4 tasks is cut as
#
#     task 1   fields 1:2
#     task 2   fields 3:5
#     task 3   fields 6:7
#     task 4   fields 8:10
#
# Every task advances its chunk as a batch of its own, on its own thread, at the same time as the
# others. The fields of a batch do not act on one another, a field with its perturbations being
# one segment of an orbit, so that the chunks are separate integrations and the result is that of
# one task. The tasks run at the same time and cannot share arrays or plans: each has its own
# storage, `P.cache[c]` and `P.tangents[c]` for task `c`.

# f(c, fields) for every task c of the propagator and the fields of its chunk of a batch of B
# fields. A batch of fewer fields than tasks gives one field to each of as many tasks. Task c
# always runs on thread c, so that its storage is used by the thread that wrote it first and
# stays in the memory next to that core.
function _foreach_chunk(f, P::KolmogorovPropagator, B::Int)
    tasks  = min(length(P.cache), B)
    chunks = [(B*(c - 1) ÷ tasks + 1):(B*c ÷ tasks) for c in 1:tasks]

    # ---- one task: the whole batch, with no threads ----
    tasks == 1 &&
        return f(1, chunks[1])

    # ---- every task on its own thread ----
    Threads.@threads :static for c in 1:tasks
        f(c, chunks[c])
    end

    return nothing
end


# ---------------------------------------------------------------------------- #
# the storage                                                                  #
# ---------------------------------------------------------------------------- #

# the registers of the scheme of the propagator for batches like X, in a mode of Flows.jl
_scheme(P::KolmogorovPropagator, X::FTFields, mode...) = getfield(Flows, P.scheme)(X, mode...)

# The storage of the batches of B fields of task c. It lives in the propagator, in P.cache[c]
# under the key B: it is built once, at the first call with B fields, and every later call with B
# fields only looks it up. It is kept by batch size because MultipleShooting calls the propagator
# with batches of different sizes, the seeds and the directions, not known when the propagator is
# made, and by task because the tasks run at the same time.
function _storage(P::KolmogorovPropagator, B::Int, c::Int=1)
    return get!(P.cache[c], B) do
        # ---- the batches that the flows advance, on the device of the propagator ----
        X = FTFields(similar(P.template, ComplexF64, (P.n + 1, 2P.n + 1, B)))
        V = similar(X)

        (X        = X,                          # the fields
         V        = V,                          # a second batch: the adjoint fields, or a term
         explicit = ExplicitTerm(X, P.m, P.kf), # the explicit term of the equation
         single   = _scheme(P, X))              # the registers of the scheme, for the equation
    end
end

# The storage of the tangent equation of task c for B fields with K perturbations, k′ batches of
# as many as the fields: in P.tangents[c] under the key (B, K), built once, at the first
# call of tangent! with these sizes. The fields and their perturbations are advanced
# as one batch, the bundle of linearised.jl, in which every field is followed by its k′
# perturbations.
function _tangent_storage(P::KolmogorovPropagator, B::Int, K::Int, c::Int=1)
    return get!(P.tangents[c], (B, K)) do
        # ---- the bundle that the flow advances, on the device of the propagator ----
        Z = FTFields(similar(P.template, ComplexF64, (P.n + 1, 2P.n + 1, B + K)))

        (Z      = Z,                                   # the fields, each with its perturbations
         term   = TangentTerm(Z, K ÷ B, P.m, P.kf),    # the explicit term of the equation and of
                                                       # its tangent
         scheme = _scheme(P, Z))                       # the registers of the scheme
    end
end

# What adjoint! adds to the storage of the batches of B fields, in P.adjoints under the key B. It
# is built once, at the first call of adjoint! with B fields, and not with the rest, so that a
# search that never calls adjoint! holds none of it.
function _adjoint_storage(P::KolmogorovPropagator, B::Int)
    return get!(P.adjoints, B) do
        # ---- a batch of the storage, for the size and the device of the others ----
        X = _storage(P, B).X

        (term   = AdjointTerm(X, P.m),                         # the explicit term of the adjoint
         scheme = _scheme(P, X, Flows.DiscreteMode(true)))     # the registers of the scheme, for the adjoint
    end
end


# ---------------------------------------------------------------------------- #
# fields advanced with their perturbations                                     #
# ---------------------------------------------------------------------------- #

# the fields X advanced by τ, and the perturbations V with them, k′ batches of as many as the
# fields, a vector or a tuple, by the tangent equation, with the explicit term `wrap(term)` of its
# storage: what tangent! and forced! do
function _advance!(P::KolmogorovPropagator, X::FTFields, V, τ::Real, wrap=identity)
    # ---- input checks ----
    all(v -> size(v, 3) == size(X, 3), V) ||
        throw(ArgumentError("every batch of perturbations must have as many as the fields"))

    B  = size(X, 3)
    k′ = length(V)

    # ---- every task advances its chunk of the fields, each with its perturbations: with one
    #      task, the whole batch ----
    _foreach_chunk(P, B) do c, fields
        b       = length(fields)
        tangent = _tangent_storage(P, b, k′*b, c)
        Z       = tangent.Z

        # ---- the flow of the equation with its tangent, at the Reynolds number of P ----
        ϕ = Flows.flow(wrap(tangent.term),
                       ImplicitTerm(P.Re),
                       tangent.scheme,
                       Flows.TimeStepConstant(P.dt))

        # ---- every field of the chunk with its perturbations in the bundle of the flow, with
        #      the symmetry of fields ----
        apply_symmetry!(_bundle!(Z, X, V, fields))

        # ---- advanced together, the fields once, and back ----
        ϕ(Z, (0, τ))
        _unbundle!(X, V, Z, fields)
    end

    return V
end
