# The propagator of Kolmogorov flow for MultipleShooting.jl: the methods of its interface on
# batches of fields, all the fields of a batch advanced together by the IMEX Runge–Kutta scheme
# CB3R2R3e of Flows.jl with constant steps, the viscous term implicit, and the transpose of its
# tangent, the adjoint of the discrete flow.

"""
    KolmogorovPropagator(n, m, Re, kf; dt=0.01, array=Array)
    KolmogorovPropagator(P, Re)

The flow of two-dimensional Kolmogorov flow on batches of vorticity fields, for the searches of
MultipleShooting.jl: [`propagate!`](@ref), [`tangent!`](@ref), [`forced!`](@ref),
[`rhs!`](@ref), [`shift!`](@ref) and [`dshift!`](@ref), and the transpose of the tangent,
[`adjoint!`](@ref), over the stages kept by [`stages`](@ref). A batch is an [`FTFields`](@ref) on an array of type `array`, the Fourier
coefficients of `B` fields. Field by field the results are those of OpenKolmogorovFlow with the
same scheme and steps, to rounding.

The storage of the batches of `B` fields, the terms with their plans and arrays and the registers
of the scheme, lives in the propagator. It is built once, at the first call with `B` fields, and
every later call with `B` fields only looks it up, so that a call allocates no batch, but for the
stages that [`stages`](@ref) returns. What `adjoint!` adds, its term and its registers, is built
at its own first call, so that a search that never calls it holds none of it.

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
- `array::Type=Array`: the type of the batches, `Array`, or `CuArray` once CUDA is loaded.
"""
struct KolmogorovPropagator{A<:AbstractArray}
           n::Int            # the largest wavenumber of the fields
           m::Int            # the grid has 2m+2 points along each direction
          Re::Float64        # the Reynolds number
          kf::Int            # the wavenumber of the forcing
          dt::Float64        # the time step
    template::A              # an empty array of the type of the batches
       cache::Dict{Int, Any} # the storage of the batches of B fields, under the key B, built at
                             # the first call with B fields and looked up by the later ones,
                             # shared with the propagators at other Reynolds numbers
    adjoints::Dict{Int, Any} # what adjoint! adds to it, under the key B, built at its first call
                             # with B fields, shared in the same way

    function KolmogorovPropagator(n::Int,
                                  m::Int,
                                 Re::Real,
                                 kf::Int;
                                 dt::Real=0.01,
                              array::Type=Array)
        # ---- input checks ----
        2m + 2 >= 3n + 1 ||
            throw(ArgumentError("the grid must dealias the products, 2m + 2 at least 3n + 1"))
        1 <= kf <= n ||
            throw(ArgumentError("the forcing wavenumber must be in 1:n"))

        # ---- an empty array of the type of the batches, from which their storage is made ----
        template = array{ComplexF64}(undef, 0, 0, 0)

        return new{typeof(template)}(n,
                                     m,
                                     Float64(Re),
                                     kf,
                                     Float64(dt),
                                     template,
                                     Dict{Int, Any}(),
                                     Dict{Int, Any}())
    end

    # the propagator of P at another Reynolds number, with its terms and its storage, shared
    KolmogorovPropagator(P::KolmogorovPropagator{A}, Re::Real) where {A} =
        new{A}(P.n,
               P.m,
               Float64(Re),
               P.kf,
               P.dt,
               P.template,
               P.cache,
               P.adjoints)
end

# The storage of the batches of B fields. It lives in the propagator, in P.cache under the key B:
# it is built once, at the first call with B fields, and every later call with B fields only
# looks it up. It is kept by batch size because MultipleShooting calls the propagator with
# batches of different sizes, the seeds and the directions, not known when the propagator is
# made.
function _storage(P::KolmogorovPropagator, B::Int)
    return get!(P.cache, B) do
        # ---- the batches that the flows advance, on the device of the propagator ----
        X = FTFields(similar(P.template, ComplexF64, (P.n + 1, 2P.n + 1, B)))
        V = similar(X)

        (X        = X,                                  # the fields
         V        = V,                                  # the perturbations, or the adjoint fields
         explicit = ExplicitTerm(X, P.m, P.kf),         # the explicit term of the equation
         tangent  = TangentTerm(X, P.m),                # and of its tangent
         single   = Flows.CB3R2R3e(X),                  # the registers of the scheme, for the equation
         coupled  = Flows.CB3R2R3e(Flows.couple(X, V))) # and for the equation with its tangent
    end
end

# What adjoint! adds to the storage of the batches of B fields, in P.adjoints under the key B. It
# is built once, at the first call of adjoint! with B fields, and not with the rest, so that a
# search that never calls adjoint! holds none of it.
function _adjoint_storage(P::KolmogorovPropagator, B::Int)
    return get!(P.adjoints, B) do
        # ---- a batch of the storage, for the size and the device of the others ----
        X = _storage(P, B).X

        (term   = AdjointTerm(X, P.m),                          # the explicit term of the adjoint
         scheme = Flows.CB3R2R3e(X, Flows.DiscreteMode(true)))  # the registers of the scheme, for the adjoint
    end
end

# the fields X and the vectors V advanced by τ, the vectors by the tangent equation with the
# explicit term `linearised`
function _advance!(P::KolmogorovPropagator, X::FTFields, V::FTFields, τ::Real, linearised)
    storage = _storage(P, size(X, 3))
    L       = ImplicitTerm(P.Re)

    # ---- the flow of the equation coupled to its tangent, at the Reynolds number of P ----
    ϕ = Flows.flow(Flows.couple(storage.explicit, linearised),
                   Flows.couple(L, L),
                   storage.coupled,
                   Flows.TimeStepConstant(P.dt))

    # ---- the fields and the vectors in the storage of the flow, with the symmetry of fields ----
    apply_symmetry!(copyto!(storage.X, X))
    apply_symmetry!(copyto!(storage.V, V))

    # ---- advanced together, and back ----
    ϕ(Flows.couple(storage.X, storage.V), (0, τ))
    copyto!(X, storage.X)

    return copyto!(V, storage.V)
end

"""
    propagate!(P::KolmogorovPropagator, X, τ) -> X

Advance in place every field of the batch `X` by the time `τ`, all together. The batch is first
given the symmetry of fields, by `apply_symmetry!`. `X` may be a view of a batch.
"""
function propagate!(P::KolmogorovPropagator, X::FTFields, τ::Real)
    storage = _storage(P, size(X, 3))

    # ---- the flow of the equation, at the Reynolds number of P ----
    ϕ = Flows.flow(storage.explicit,
                   ImplicitTerm(P.Re),
                   storage.single,
                   Flows.TimeStepConstant(P.dt))

    # ---- the fields in the storage of the flow, with the symmetry of fields ----
    apply_symmetry!(copyto!(storage.X, X))

    # ---- advanced, and back ----
    ϕ(storage.X, (0, τ))

    return copyto!(X, storage.X)
end

"""
    tangent!(P::KolmogorovPropagator, X, V, τ) -> V

Advance in place by `τ` the fields `X` and the perturbations `V` together, each perturbation by
the tangent equation about the field in the same place, with the scheme and the steps of
[`propagate!`](@ref), so that `V` becomes the derivative of the discrete flow applied to `V`.
"""
tangent!(P::KolmogorovPropagator, X::FTFields, V::FTFields, τ::Real) =
    _advance!(P, X, V, τ, _storage(P, size(X, 3)).tangent)

"""
    forced!(P::KolmogorovPropagator, X, V, τ) -> V

As [`tangent!`](@ref), with the tangent equation forced by the derivative of the right-hand side
in the Reynolds number, `-Δω/Re²`. From `V = 0` it gives the derivative of the end points in the
Reynolds number, the column of the parameter of MultipleShooting.jl. The scheme treats the
forcing as an explicit term, so that this is the derivative of the flow up to the error of the
scheme, not that of the discrete flow to rounding.
"""
forced!(P::KolmogorovPropagator, X::FTFields, V::FTFields, τ::Real) =
    _advance!(P, X, V, τ, ForcedTangentTerm(_storage(P, size(X, 3)).tangent, P.Re))

"""
    stages(P::KolmogorovPropagator, X, τ) -> S

The stages of every step of the flow over `τ` from the fields `X`, kept for [`adjoint!`](@ref),
which goes back over them. `X` holds the fields at the start of the interval and is left as it
is. The steps are those of [`propagate!`](@ref). The stages are made once and serve any number of
calls of `adjoint!`, with the propagator that made them.

`S` holds the four stages of each of the `N` steps over `τ`, `4N` batches like `X`, in the
stage cache of Flows.jl. For one field with `n = 33` and steps of 0.01 this is 14.6 GB over 1000
time units, whatever the number of segments the orbit is cut into.
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
    S = Flows.RAMStageCache(4, storage.X)
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
