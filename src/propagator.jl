# The propagator of Kolmogorov flow for MultipleShooting.jl: the methods of its interface on
# batches of fields, all the fields of a batch advanced together by an implicit–explicit scheme
# of Flows.jl with constant steps, CB3R2R3e unless another is asked for, the viscous term
# implicit, and the transpose of its tangent, the adjoint of the discrete flow.
#
# A propagator is of one size: batches of B fields with k′ perturbations each, k′ = 0 for none.
# It holds what its flows advance and work on, its workspace, made with it, so that a call
# allocates no batch. A search calls a few sizes, all known before it starts: B fields alone, B
# fields with one perturbation each, for a product of the Jacobian, and B fields with k′
# perturbations each, for a subspace step. `propagators` makes a propagator for each, and returns
# them in a dictionary by size, which is itself a propagator: it hands every call to the one of
# its size.
#
# The file has four parts: the propagator and the dictionary; the methods of the interface, which
# are what a search calls; the same methods on the dictionary; then what serves them, the tasks
# of a propagator on the CPU and the advance of fields with their perturbations.

"""
    KolmogorovPropagator(n, m, Re, kf; B, k′=0, dt=0.01, scheme=:CB3R2R3e, device=:cpu, tasks=1,
                         adjoint=false)
    KolmogorovPropagator(P, Re)

The flow of two-dimensional Kolmogorov flow on batches of `B` vorticity fields with `k′`
perturbations each, for the searches of MultipleShooting.jl: [`propagate!`](@ref),
[`tangent!`](@ref), [`forced!`](@ref), [`rhs!`](@ref), [`shift!`](@ref) and [`dshift!`](@ref),
and the transpose of the tangent, [`adjoint!`](@ref), over the stages kept by
[`stages`](@ref). A batch is an [`FTFields`](@ref), the Fourier coefficients of `B` fields, on
the CPU or on the GPU. Field by field the results are those of OpenKolmogorovFlow with the same
scheme and steps, to rounding.

The propagator is of one size, and holds the workspace of that size, made with it, so that a
call allocates no batch, but for the stages that [`stages`](@ref) returns. It takes batches of
`B` fields, and [`tangent!`](@ref) a vector of `k′` batches of perturbations, or one batch when
`k′ = 1`, as [`forced!`](@ref). Another size is an error. A search calls a few sizes, which
[`propagators`](@ref) makes together.

The second form is the propagator of `P` at another Reynolds number. It shares the workspace of
`P` and costs nothing. Two propagators that share their workspace must not run at the same time.

# Arguments
- `n::Int`: the largest wavenumber of the fields.
- `m::Int`: the grid of the products has `2m+2` points along each direction, at least `3n + 1`,
  which keeps them free of aliasing.
- `Re::Real`: the Reynolds number.
- `kf::Int`: the wavenumber of the forcing, from 1 to `n`.
- `B::Int`: the fields of a batch, required.
- `k′::Int=0`: the perturbations of every field, `0` for the equation alone.
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
  so that the result is that of one task, to rounding. This is how [`propagate!`](@ref),
  [`tangent!`](@ref) and [`forced!`](@ref) use the cores of a CPU. The other methods run on one
  task. Start Julia with at least as many threads, `julia -t 16` for `tasks=16`.
- `adjoint::Bool=false`: whether the propagator has the workspace of [`adjoint!`](@ref), which a
  search does not call.
"""
struct KolmogorovPropagator
          n::Int        # the largest wavenumber of the fields
          m::Int        # the grid has 2m+2 points along each direction
         Re::Float64    # the Reynolds number
         kf::Int        # the wavenumber of the forcing
         dt::Float64    # the time step
     scheme::Symbol     # the scheme of Flows.jl
          B::Int        # the fields of a batch
         k′::Int        # the perturbations of every field, 0 for none
      tasks::Vector     # every task, its chunk of the fields with the workspaces of the chunk
      whole::Any        # the workspace of the equation for the whole batch, on one task
    adjoint::Any        # that of the adjoint for the whole batch, or nothing

    # A workspace is a named tuple,
    #
    #     of the equation   (X, V, explicit, scheme)    the fields, a second batch, the explicit
    #                                                   term, the registers of the scheme
    #     of the tangent    (Z, term, scheme)           the bundle of the fields with their k′
    #                                                   perturbations (linearised.jl), its term,
    #                                                   the registers
    #     of the adjoint    (term, scheme)              its term, the registers in the mode of
    #                                                   the discrete adjoint
    #
    # and a task (see the tasks below) is a named tuple (fields, equation, tangent): the fields of
    # its chunk, a range, and the workspaces of the chunk, of the equation and of the tangent
    # equation, nothing if k′ = 0. The methods that run on one task use the workspaces of the whole
    # batch, which are those of the task when there is one.
    function KolmogorovPropagator(n::Int,
                                  m::Int,
                                 Re::Real,
                                 kf::Int;
                                  B::Int,
                                 k′::Int=0,
                                 dt::Real=0.01,
                             scheme::Symbol=:CB3R2R3e,
                             device::Symbol=:cpu,
                              tasks::Int=1,
                            adjoint::Bool=false)
        # ---- input checks ----
        2m + 2 >= 3n + 1 ||
            throw(ArgumentError("the grid must dealias the products, 2m + 2 at least 3n + 1"))
        1 <= kf <= n ||
            throw(ArgumentError("the forcing wavenumber must be in 1:n"))
        device in (:cpu, :gpu) ||
            throw(ArgumentError("device must be :cpu or :gpu"))
        B >= 1 && k′ >= 0 ||
            throw(ArgumentError("a batch has B ≥ 1 fields with k′ ≥ 0 perturbations each"))
        haskey(Flows.imex_schemes, scheme) ||
            throw(ArgumentError("scheme must be one of Flows.imex_schemes, " *
                                "$(sort(collect(keys(Flows.imex_schemes))))"))

        # ---- a batch of b fields on the device, and the registers of the scheme for batches
        #      like X, in a mode of Flows.jl ----
        template              = _array(Val(device)){ComplexF64}(undef, 0, 0, 0)
        batch(b)              = FTFields(similar(template, ComplexF64, (n + 1, 2n + 1, b)))
        registers(X, mode...) = Flows.imex_schemes[scheme](X, mode...)

        # ---- the workspaces of b fields: of the equation, of the tangent equation, and of the
        #      adjoint ----
        function equation(b)
            X = batch(b)
            return (X        = X,
                    V        = similar(X),
                    explicit = ExplicitTerm(X, m, kf),
                    scheme   = registers(X))
        end
        function tangent(b)
            Z = batch((1 + k′)*b)
            return (Z      = Z,
                    term   = TangentTerm(Z, k′, m, kf),
                    scheme = registers(Z))
        end
        function backward(b)
            X = batch(b)
            return (term   = AdjointTerm(X, m),
                    scheme = registers(X, Flows.DiscreteMode(true)))
        end

        # ---- every task: its chunk, t ranges of fields of sizes as equal as can be, and the
        #      workspaces of the chunk ----
        t    = min(tasks, B)
        work = map(1:t) do c
            fields = (B*(c - 1) ÷ t + 1):(B*c ÷ t)
            (fields   = fields,
             equation = equation(length(fields)),
             tangent  = k′ >= 1 ? tangent(length(fields)) : nothing)
        end

        # ---- the workspaces of the whole batch, for the methods that run on one task ----
        whole = t == 1 ? work[1].equation : equation(B)
        back  = adjoint ? backward(B) : nothing

        return new(n,
                   m,
                   Float64(Re),
                   kf,
                   Float64(dt),
                   scheme,
                   B,
                   k′,
                   work,
                   whole,
                   back)
    end

    # the propagator of P at another Reynolds number, with its workspaces, shared
    KolmogorovPropagator(P::KolmogorovPropagator, Re::Real) =
        new(P.n,
            P.m,
            Float64(Re),
            P.kf,
            P.dt,
            P.scheme,
            P.B,
            P.k′,
            P.tasks,
            P.whole,
            P.adjoint)
end

# the arrays of the batches on a device: Array on the CPU, CuArray on the GPU, a method that the
# extension of the package for CUDA adds once CUDA is loaded
_array(::Val{:cpu}) = Array
_array(::Val)       = throw(ArgumentError("device=:gpu needs CUDA: `using CUDA` first"))

# the propagators of several sizes, by size (B, k′)
const Propagators = Dict{Tuple{Int, Int}, KolmogorovPropagator}

"""
    propagators(n, m, Re, kf; sizes, kwargs...) -> D
    propagators(D, Re) -> D

The propagators of the sizes `sizes`, a vector of pairs `(B, k′)`, each a
[`KolmogorovPropagator`](@ref) of `B` fields with `k′` perturbations each, with the options
`kwargs` of that constructor, in a dictionary by size: `D[B, k′]` is the propagator of that size.
The dictionary is a propagator of MultipleShooting.jl that hands every call to the one of its
size: [`propagate!`](@ref), [`rhs!`](@ref), [`stages`](@ref) and [`adjoint!`](@ref) of `B`
fields to that of `B` fields with the fewest perturbations, [`tangent!`](@ref) with one batch of
perturbations and [`forced!`](@ref) to `(B, 1)`, and [`tangent!`](@ref) with a vector of `k′`
of them to `(B, k′)`. A size that it does not hold is an error.

A search of MultipleShooting.jl with `m` seeds and a subspace of `k′` directions takes
`sizes = [(m, 1), (m, k′)]`; the sweeps of `N` segments with `k` vectors, `[(1, k), (N, k)]`.

The second form is the dictionary `D` at another Reynolds number, every propagator sharing its
workspace, so that `Re -> propagators(D, Re)` is the family of propagators that the searches with
a free parameter of MultipleShooting.jl take.
"""
function propagators(n::Int, m::Int, Re::Real, kf::Int; sizes::AbstractVector, kwargs...)
    # ---- input checks ----
    isempty(sizes) &&
        throw(ArgumentError("sizes must hold at least one pair (B, k′)"))

    return Propagators((B, k′) => KolmogorovPropagator(n, m, Re, kf; B=B, k′=k′, kwargs...)
                       for (B, k′) in sizes)
end

propagators(D::Propagators, Re::Real) = Propagators(s => KolmogorovPropagator(P, Re) for (s, P) in D)

# the propagator of D of B fields with k′ perturbations each
function _sized(D::Propagators, B::Int, k′::Int)
    # ---- input checks ----
    haskey(D, (B, k′)) ||
        throw(ArgumentError("no propagator of $B fields with $k′ perturbations each among " *
                            "the sizes $(sort(collect(keys(D))))"))

    return D[B, k′]
end

# the propagator of D of B fields with the fewest perturbations, for the methods of the equation
function _fields(D::Propagators, B::Int)
    k′ = minimum((k for (b, k) in keys(D) if b == B); init=typemax(Int))

    # ---- input checks ----
    k′ < typemax(Int) ||
        throw(ArgumentError("no propagator of $B fields among the sizes $(sort(collect(keys(D))))"))

    return D[B, k′]
end

# the check that a call of B fields with k′ perturbations each is of the size of P
function _check(P::KolmogorovPropagator, B::Int, k′::Int=P.k′)
    (B, k′) == (P.B, P.k′) ||
        throw(ArgumentError("a propagator of $(P.B) fields with $(P.k′) perturbations each, " *
                            "called with $B fields with $k′"))

    return nothing
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
    # ---- input checks ----
    _check(P, size(X, 3))

    # ---- every task advances its chunk of the fields: with one task, the whole batch ----
    _foreach_task(P) do task
        w     = task.equation
        chunk = view(X, :, :, task.fields)

        # ---- the flow of the equation, at the Reynolds number of P ----
        ϕ = Flows.flow(w.explicit,
                       ImplicitTerm(P.Re),
                       w.scheme,
                       Flows.TimeStepConstant(P.dt))

        # ---- the fields of the chunk in the workspace of the flow, with the symmetry of
        #      fields ----
        apply_symmetry!(copyto!(w.X, chunk))

        # ---- advanced, and back ----
        ϕ(w.X, (0, τ))
        copyto!(chunk, w.X)
    end

    return X
end

"""
    tangent!(P::KolmogorovPropagator, X, V, τ) -> V

Advance in place by `τ` the fields `X` and the perturbations `V` together, each perturbation by
the tangent equation about its field, with the scheme and the steps of [`propagate!`](@ref), so
that `V` becomes the derivative of the discrete flow applied to `V`.

`V` is a batch like `X`, one perturbation of every field, for a product of the Jacobian of
MultipleShooting.jl, with a propagator of `k′ = 1`, or a vector of `k′` such batches, the
directions of its subspace step. Every field goes with its `k′` perturbations in one batch, and is
advanced once for all of them, so that the `k′` directions cost one integration of the equation
and `k′` of its tangent. Batch by batch the result is that of `tangent!` on the batch alone, to
rounding.
"""
tangent!(P::KolmogorovPropagator, X::FTFields, V::FTFields, τ::Real) =
    (_advance!(P, X, (V,), τ); V)

tangent!(P::KolmogorovPropagator, X::FTFields, V::AbstractVector{<:FTFields}, τ::Real) =
    _advance!(P, X, V, τ)

"""
    forced!(P::KolmogorovPropagator, X, V, τ) -> V

As [`tangent!`](@ref) with one batch of perturbations, with the tangent equation forced by the
derivative of the right-hand side in the Reynolds number, `-Δω/Re²`. From `V = 0` it gives the
derivative of the end points in the Reynolds number, the column of the parameter of
MultipleShooting.jl. The scheme treats the forcing as an explicit term, so that this is the
derivative of the flow up to the error of the scheme, not that of the discrete flow to rounding.
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
    # ---- input checks ----
    _check(P, size(X, 3))

    w = P.whole

    # ---- the flow of the equation, at the Reynolds number of P ----
    ϕ = Flows.flow(w.explicit,
                   ImplicitTerm(P.Re),
                   w.scheme,
                   Flows.TimeStepConstant(P.dt))

    # ---- the fields in the workspace of the flow, with the symmetry of fields ----
    apply_symmetry!(copyto!(w.X, X))

    # ---- advanced, the stages of every step kept, in copies that Flows.jl makes ----
    S = Flows.RAMStageCache(Flows.nstages(w.scheme), w.X)
    ϕ(w.X, (0, τ), S)

    return S
end

"""
    adjoint!(P::KolmogorovPropagator, S, W) -> W

The transpose of [`tangent!`](@ref) in the inner product of the fields, [`dots`](@ref), over
the stages `S` of [`stages`](@ref): `W` becomes the transpose of the derivative of the discrete
flow, over the interval and from the fields of `S`, applied to `W`. For any batches `V` and `W`,
field by field and to rounding,

    dots(tangent!(P, copy(X), V, τ), W) == dots(V, adjoint!(P, stages(P, X, τ), W))

The adjoint scheme runs backwards over `S`, from the last step to the first, and leaves `S` as
it is, so that one forward pass serves every `W`: a call costs one backward pass. `W` has as many
fields as the batch that made `S`, each about the field in the same place, and is first given the
symmetry of fields, as `tangent!` gives it to its input, so that the transpose includes that
projection. `W` may be a view of a batch. The propagator must be made with `adjoint=true`. This
is a method of `LinearAlgebra.adjoint!`.
"""
function LinearAlgebra.adjoint!(P::KolmogorovPropagator, S::Flows.RAMStageCache, W::FTFields)
    # ---- input checks ----
    _check(P, size(W, 3))
    P.adjoint === nothing &&
        throw(ArgumentError("the propagator was made without the adjoint: adjoint=true"))

    w    = P.whole
    back = P.adjoint

    # ---- the flow of the adjoint equation back over the stages, at the Reynolds number of P ----
    ψ = Flows.flow(back.term,
                   ImplicitTerm(P.Re),
                   back.scheme,
                   Flows.TimeStepFromCache())

    # ---- the adjoint fields in the workspace of the flow, with the symmetry of fields ----
    apply_symmetry!(copyto!(w.V, W))

    # ---- back over the stages, from the last step to the first ----
    ψ(w.V, S)

    return copyto!(W, w.V)
end

"""
    rhs!(P::KolmogorovPropagator, F, X) -> F

The time derivative of every field of the batch `X`, advection, forcing and viscous term, written
into `F`.
"""
function rhs!(P::KolmogorovPropagator, F::FTFields, X::FTFields)
    # ---- input checks ----
    _check(P, size(X, 3))

    w = P.whole

    # ---- the fields in the workspace of the propagator, with the symmetry of fields ----
    Ω = apply_symmetry!(copyto!(w.X, X))

    # ---- advection and forcing ----
    w.explicit(0.0, Ω, w.V)

    # ---- plus the viscous term, Δω/Re ----
    w.V .-= (Ω.j.^2 .+ Ω.k.^2) .* Ω ./ P.Re

    return copyto!(F, w.V)
end

"""
    shift!(P::KolmogorovPropagator, X, s) -> X

Shift in place every field of the batch `X` along x, `ω(x, y) → ω(x + s, y)`, the coefficients
times `cis(j s)`. `s` is the shift, or the vector of the shifts of MultipleShooting.jl, which
holds this one alone. Any number of fields, with no workspace, and so for a dictionary of
[`propagators`](@ref).
"""
function shift!(P::Union{KolmogorovPropagator, Propagators}, X::FTFields, s)
    # ---- e^{i j (x + s)} is e^{i j s} times e^{i j x} ----
    X .*= cis.(X.j .* first(s))

    return X
end

"""
    dshift!(P::KolmogorovPropagator, G, X, i=1) -> G

The generator of the shift along x at every field of the batch `X`, `∂ω/∂x`, written into `G`.
`i` is the index of the shift, as MultipleShooting.jl passes it. Any number of fields, with no
workspace, and so for a dictionary of [`propagators`](@ref).
"""
dshift!(P::Union{KolmogorovPropagator, Propagators}, G::FTFields, X::FTFields, i::Int=1) = ddx!(G, X)


# ---------------------------------------------------------------------------- #
# the interface on a dictionary of propagators, by size                        #
# ---------------------------------------------------------------------------- #

propagate!(D::Propagators, X::FTFields, τ::Real) = propagate!(_fields(D, size(X, 3)), X, τ)

tangent!(D::Propagators, X::FTFields, V::FTFields, τ::Real) =
    tangent!(_sized(D, size(X, 3), 1), X, V, τ)

tangent!(D::Propagators, X::FTFields, V::AbstractVector{<:FTFields}, τ::Real) =
    tangent!(_sized(D, size(X, 3), length(V)), X, V, τ)

forced!(D::Propagators, X::FTFields, V::FTFields, τ::Real) = forced!(_sized(D, size(X, 3), 1), X, V, τ)

stages(D::Propagators, X::FTFields, τ::Real) = stages(_fields(D, size(X, 3)), X, τ)

LinearAlgebra.adjoint!(D::Propagators, S::Flows.RAMStageCache, W::FTFields) =
    adjoint!(_fields(D, size(W, 3)), S, W)

rhs!(D::Propagators, F::FTFields, X::FTFields) = rhs!(_fields(D, size(X, 3)), F, X)


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
# workspaces, made with the propagator.

# f(task) for every task of the propagator P, each with its chunk of the fields and its
# workspaces. A batch of fewer fields than tasks gives one field to each of as many tasks. Task c
# always runs on thread c, so that its workspaces are used by the thread that wrote them first and
# stay in the memory next to that core.
function _foreach_task(f, P::KolmogorovPropagator)
    # ---- one task: the whole batch, with no threads ----
    length(P.tasks) == 1 &&
        return f(P.tasks[1])

    # ---- every task on its own thread ----
    Threads.@threads :static for task in P.tasks
        f(task)
    end

    return nothing
end


# ---------------------------------------------------------------------------- #
# fields advanced with their perturbations                                     #
# ---------------------------------------------------------------------------- #

# the fields X advanced by τ, and the perturbations V with them, k′ batches of as many as the
# fields, a vector or a tuple, by the tangent equation, with the explicit term `wrap(term)` of
# the workspace: what tangent! and forced! do
function _advance!(P::KolmogorovPropagator, X::FTFields, V, τ::Real, wrap=identity)
    # ---- input checks ----
    _check(P, size(X, 3), length(V))
    all(v -> size(v, 3) == size(X, 3), V) ||
        throw(ArgumentError("every batch of perturbations must have as many as the fields"))

    # ---- every task advances its chunk of the fields, each with its perturbations: with one
    #      task, the whole batch ----
    _foreach_task(P) do task
        w = task.tangent

        # ---- the flow of the equation with its tangent, at the Reynolds number of P ----
        ϕ = Flows.flow(wrap(w.term),
                       ImplicitTerm(P.Re),
                       w.scheme,
                       Flows.TimeStepConstant(P.dt))

        # ---- every field of the chunk with its perturbations in the bundle of the flow, with
        #      the symmetry of fields ----
        apply_symmetry!(_bundle!(w.Z, X, V, task.fields))

        # ---- advanced together, the fields once, and back ----
        ϕ(w.Z, (0, τ))
        _unbundle!(X, V, w.Z, task.fields)
    end

    return V
end
