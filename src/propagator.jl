"""
    KolmogorovPropagator(n, m, Re, kf; dt=0.01, batch=1, array=Array)

Flow map of batches of vorticity fields of Kolmogorov flow, all the fields of
a batch advanced together by the IMEX Runge–Kutta scheme `CB3R2R3e` of
Flows.jl with constant steps `dt`.

A batch is a complex array `X` of size `(2m+2, m+2, B)` and of type `array`
(`Array`, or `CuArray` once CUDA is loaded). Slice `X[:, :, i]` is the data of
an `FTField{n, m}` of OpenKolmogorovFlow: the signed wavenumber `k` along `y`
down the rows, in FFT order, and `j ≥ 0` along `x` across the columns. Field by
field, the results are those of `ForwardEquation(n, m, Re, kf)` with the same
scheme and steps, to round-off. The propagator acts on the fields of
OpenKolmogorovFlow alone: every batch it advances is first projected onto them,
with zero inactive modes and zero mean and the column `j = 0` Hermitian, so that
the flow map and its tangent vanish on the rest of the stored entries. Without
the projection those entries would be carried unchanged, a neutral subspace of
the tangent map that random perturbations fill.

`n` is the largest active wavenumber and `2m+2` the number of grid points along
each direction (`m = 3n/2` dealiases the products). The work arrays and the
FFT plans of batches of `batch` fields are built at once; those of any other
batch size at its first use, and kept.

See [`propagate!`](@ref), [`tangent!`](@ref), [`rhs!`](@ref),
[`shift!`](@ref), [`dshift!`](@ref) and [`realdot`](@ref).
"""
struct KolmogorovPropagator{C<:AbstractArray, R<:AbstractMatrix{Float64}, L}
           n::Int              # largest active wavenumber
           m::Int              # storage cutoff, 2m+2 grid points
          kf::Int              # forcing wavenumber
          dt::Float64          # constant time step
    template::C                # empty array of the device and element type of the batches
    implicit::L                # viscous term, shared by all batch sizes
          jx::R                # wavenumbers j along x, as a row (1, m+2)
          ky::R                # signed wavenumbers k along y, as a column (2m+2, 1)
       cache::Dict{Any, Any}   # explicit terms and flows of each batch size

    function KolmogorovPropagator(n::Int,
                                  m::Int,
                                 Re::Real,
                                 kf::Int;
                                 dt::Real=0.01,
                              batch::Int=1,
                              array::Type=Array)
        # ---- input checks ----
        m >= n ||
            throw(ArgumentError("m must be at least n"))
        1 <= kf <= n ||
            throw(ArgumentError("the forcing wavenumber must be in 1:n"))

        # ---- template of the batches and wavenumbers for the shift ----
        template = array{ComplexF64}(undef, 0, 0, 0)
        jx       = _device(template, reshape(Float64.(_js(m)), 1, :))
        ky       = _device(template, reshape(Float64.(_ks(m)), :, 1))

        # ---- assemble, with the flow of the first batch size ----
        implicit = ImplicitTerm(template, n, m, Re)
        P = new{typeof(template), typeof(jx), typeof(implicit)}(n,
                                                                m,
                                                                kf,
                                                                Float64(dt),
                                                                template,
                                                                implicit,
                                                                jx,
                                                                ky,
                                                                Dict{Any, Any}())
        _forward(P, batch)
        return P
    end
end

# the explicit term and the flow of the batches of size B, built at their first use
function _forward(P::KolmogorovPropagator, B::Int)
    return get!(P.cache, B) do
        explicit = ExplicitTerm(P.template, P.n, P.m, P.kf, B, :forward)
        state    = similar(P.template, ComplexF64, (2P.m + 2, P.m + 2, B))
        ϕ        = Flows.flow(explicit,
                              P.implicit,
                              Flows.CB3R2R3e(state),
                              Flows.TimeStepConstant(P.dt))
        (explicit, ϕ)
    end
end

# the projection of a batch onto the fields of OKF, in place: zero inactive modes and mean, and the
# Hermitian part of the column j = 0, k ↔ -k; the identity on fields of OKF
function _project!(P::KolmogorovPropagator, X::AbstractArray)
    m    = P.m
    X  .*= P.implicit.mask
    pos  = view(X, 2:(m + 1), 1, :)
    neg  = view(X, (2m + 2):-1:(m + 3), 1, :)
    pos .= 0.5 .* (pos .+ conj.(neg))
    neg .= conj.(pos)
    return X
end

# the flow of the equation coupled to its tangent, for batches of size B
function _tangent(P::KolmogorovPropagator, B::Int)
    return get!(P.cache, (:tangent, B)) do
        forward = ExplicitTerm(P.template, P.n, P.m, P.kf, B, :forward)
        tangent = ExplicitTerm(P.template, P.n, P.m, P.kf, B, :tangent)
        state   = similar(P.template, ComplexF64, (2P.m + 2, P.m + 2, B))
        Flows.flow(Flows.couple(forward, tangent),
                   Flows.couple(P.implicit, P.implicit),
                   Flows.CB3R2R3e(Flows.couple(state, similar(state))),
                   Flows.TimeStepConstant(P.dt))
    end
end

"""
    propagate!(P::KolmogorovPropagator, X, τ) -> X

Advance in place every field of the batch `X` by the time `τ`, all together:
`floor(τ/dt)` steps of `dt` and a last shorter step that ends on `τ`, as
`Flows.TimeStepConstant` (a last step shorter than `10⁻⁶ dt` is merged into
the one before). `X` must be an array of the type of the propagator, not a
view.
"""
function propagate!(P::KolmogorovPropagator, X::AbstractArray{ComplexF64, 3}, τ::Real)
    _, ϕ = _forward(P, size(X, 3))
    ϕ(_project!(P, X), (0, τ))
    return X
end

"""
    tangent!(P::KolmogorovPropagator, X, V, τ) -> V

Advance in place by `τ` the fields `X` and the perturbations `V` together,
`V[:, :, i]` by the tangent equation about `X[:, :, i]`: OKF's
`LinearisedEquation(TangentMode())` coupled to the equation, with the same
scheme and steps, so that `V` becomes the derivative of the discrete flow map
applied to `V`. `X` and `V` have the same size, the batch of the bases repeated
when several perturbations share a base.
"""
function tangent!(P::KolmogorovPropagator, X::AbstractArray{ComplexF64, 3},
                                           V::AbstractArray{ComplexF64, 3}, τ::Real)
    ϕ = _tangent(P, size(X, 3))
    ϕ(Flows.couple(_project!(P, X), _project!(P, V)), (0, τ))
    return V
end

"""
    rhs!(P::KolmogorovPropagator, out, X) -> out

Time derivative of every field of the batch `X`: viscous term, advection and
forcing, as the call of OKF's `ForwardEquation`. `out` is overwritten.
"""
function rhs!(P::KolmogorovPropagator, out::AbstractArray, X::AbstractArray)
    explicit, _ = _forward(P, size(X, 3))
    explicit(0.0, X, out)
    out .+= X .* P.implicit.lap .* (1/P.implicit.Re)
    return out
end

"""
    shift!(P::KolmogorovPropagator, X, s, ℓy=0) -> X

Shift every field of `X` in place, `ω(x, y) → ω(x + s, y + ℓy)`: the
coefficients are multiplied by `cis(j s + k ℓy)`. With `ℓy = 0` this is OKF's
`xshift!(Ω, s)`. `X` may be a view, of one field or of a batch, with any number
of trailing batch dimensions. A method of OKF's `shift!`.
"""
shift!(P::KolmogorovPropagator, X::AbstractArray, s::Real, ℓy::Real=0) =
    (X .*= cis.(s .* P.jx .+ ℓy .* P.ky); X)

"""
    dshift!(P::KolmogorovPropagator, out, X) -> out

The generator of the shift along `x`, `∂ω/∂x = im j X`, for every field of `X`,
as OKF's `ddx!`. `X` may have any number of trailing batch dimensions.
"""
dshift!(P::KolmogorovPropagator, out::AbstractArray, X::AbstractArray) =
    (out .= (im .* P.jx) .* X; out)

"""
    realdot(X, Y) -> Float64

The inner product of OKF, `dot` of two `FTField`s, summed over the fields of
the batches `X` and `Y`: twice the modes `j ≥ 1`, which stand for their
conjugates, once those of the column `j = 0`, and half the mean. It is the
spatial mean of `ω₁ ω₂` (`realdot(X, X)` is the mean of `ω²` summed over the
fields, `Re` times the dissipation rate of OKF's `dissrate`). It counts all the
stored modes, equal to OKF's `dot` for fields with zero inactive modes, as
those of the propagator. One reduction on the device of `X`.
"""
realdot(X::AbstractArray, Y::AbstractArray) =
    sum(Broadcast.instantiate(Broadcast.broadcasted(_realdot,
                                                    X,
                                                    Y,
                                                    reshape(1:size(X, 1), :, 1),
                                                    reshape(1:size(X, 2), 1, :))))

# the term of row r and column c of realdot
_realdot(x, y, r::Int, c::Int) = ifelse(c == 1, ifelse(r == 1, 0.5, 1.0), 2.0) * real(conj(x) * y)
