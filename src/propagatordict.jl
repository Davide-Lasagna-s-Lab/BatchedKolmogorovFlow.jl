# The propagators of the few sizes that a search calls, in a dictionary by size, which is itself a
# propagator of MultipleShooting.jl. A search calls a few sizes, all known before it starts: B
# fields alone, B fields with one perturbation each, for a product of the Jacobian, and B fields
# with k′ perturbations each, for a subspace step. The dictionary holds a propagator of each size,
# a KolmogorovPropagator (propagator.jl), and hands every call to the one of its size.
#
# The file has two parts: the dictionary, indexed by size, and the methods of the interface of
# MultipleShooting.jl on it.

"""
    KolmogorovPropagatorDict(n, m, Re, kf; sizes, kwargs...) -> D
    KolmogorovPropagatorDict(D, Re) -> D

The propagators of the sizes `sizes`, a vector of pairs `(B, k′)`, each a
[`KolmogorovPropagator`](@ref) of `B` fields with `k′` perturbations each, with the options
`kwargs` of that constructor, in a dictionary by size: `D[B, k′]` is the propagator of that size.
`KolmogorovPropagatorDict` is the type of that dictionary, `Dict{Tuple{Int, Int},
KolmogorovPropagator}`.

The dictionary is a propagator of MultipleShooting.jl that hands every call to the one of its
size: [`tangent!`](@ref) with one batch of perturbations and [`forced!`](@ref) to `(B, 1)`,
[`tangent!`](@ref) with a vector of `k′` of them to `(B, k′)`, and [`propagate!`](@ref),
[`rhs!`](@ref), [`stages`](@ref) and [`adjoint!`](@ref) of `B` fields to any of `B` fields, as
every one has the workspaces of the equation. A size that it does not hold is an error.

A search of MultipleShooting.jl with `m` seeds and a subspace of `k′` directions takes
`sizes = [(m, 1), (m, k′)]`; the sweeps of `N` segments with `k` vectors, `[(1, k), (N, k)]`.

The second form is the dictionary `D` at another Reynolds number, every propagator sharing its
workspaces, so that `Re -> KolmogorovPropagatorDict(D, Re)` is the family of propagators that
the searches with a free parameter of MultipleShooting.jl take.
"""
const KolmogorovPropagatorDict = Dict{Tuple{Int, Int}, KolmogorovPropagator}

function KolmogorovPropagatorDict(n::Int,
                                  m::Int,
                                 Re::Real,
                                 kf::Int;
                              sizes::AbstractVector,
                                    kwargs...)
    # ---- input checks ----
    isempty(sizes) &&
        throw(ArgumentError("sizes must hold at least one pair (B, k′)"))

    # ---- a propagator for every size, with the same options ----
    return KolmogorovPropagatorDict((B, k′) => KolmogorovPropagator(n, m, Re, kf;
                                                                     B=B,
                                                                     k′=k′,
                                                                     kwargs...)
                                    for (B, k′) in sizes)
end

KolmogorovPropagatorDict(D::KolmogorovPropagatorDict, Re::Real) =
    KolmogorovPropagatorDict(s => KolmogorovPropagator(P, Re) for (s, P) in D)

# D[B, k′], the propagator of B fields with k′ perturbations each, for tangent! and forced!, and an
# error that lists the sizes of D if it has none
function Base.getindex(D::KolmogorovPropagatorDict, B::Int, k′::Int)
    # ---- input checks ----
    haskey(D, (B, k′)) ||
        throw(ArgumentError("no propagator of $B fields with $k′ perturbations each among " *
                            "the sizes $(sort(collect(keys(D))))"))

    return D[(B, k′)]
end

# D[B], a propagator of B fields, for the methods of the equation: any, since every one has the
# workspaces of the equation, and the same, as they are made with the same options
function Base.getindex(D::KolmogorovPropagatorDict, B::Int)
    # ---- the first of B fields ----
    for ((b, k′), P) in D
        b == B && return P
    end

    throw(ArgumentError("no propagator of $B fields among the sizes $(sort(collect(keys(D))))"))
end


# ---------------------------------------------------------------------------- #
# the interface of MultipleShooting.jl, by size                                #
# ---------------------------------------------------------------------------- #

propagate!(D::KolmogorovPropagatorDict, X::FTFields, τ::Real) =
    propagate!(D[size(X, 3)], X, τ)

tangent!(D::KolmogorovPropagatorDict, X::FTFields, V::FTFields, τ::Real) =
    tangent!(D[size(X, 3), 1], X, V, τ)

tangent!(D::KolmogorovPropagatorDict, X::FTFields, V::AbstractVector{<:FTFields}, τ::Real) =
    tangent!(D[size(X, 3), length(V)], X, V, τ)

forced!(D::KolmogorovPropagatorDict, X::FTFields, V::FTFields, τ::Real) =
    forced!(D[size(X, 3), 1], X, V, τ)

stages(D::KolmogorovPropagatorDict, X::FTFields, τ::Real) =
    stages(D[size(X, 3)], X, τ)

LinearAlgebra.adjoint!(D::KolmogorovPropagatorDict, S::Flows.RAMStageCache, W::FTFields) =
    adjoint!(D[size(W, 3)], S, W)

rhs!(D::KolmogorovPropagatorDict, F::FTFields, X::FTFields) =
    rhs!(D[size(X, 3)], F, X)

# the shift and its generator need no workspace: any propagator of the dictionary
shift!(D::KolmogorovPropagatorDict, X::FTFields, s) = shift!(first(values(D)), X, s)

dshift!(D::KolmogorovPropagatorDict, G::FTFields, X::FTFields, i::Int=1) =
    dshift!(first(values(D)), G, X, i)
