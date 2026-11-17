# The explicit term of the adjoint equation, the transpose of the linearised advection of
# linearised.jl, in the call interface of Flows.jl for the adjoint of a discrete flow. In the inner
# product of the fields, the mean of their product, the adjoint of the linearised advection about
# a field ω is
#
#     μ → u ∂μ/∂x + v ∂μ/∂y + Δ⁻¹(∂μ/∂x ∂ω/∂y - ∂μ/∂y ∂ω/∂x),
#
# with (u, v) the velocity of ω, and the adjoint equation, solved backwards in time, is
#
#     -∂μ/∂t = u ∂μ/∂x + v ∂μ/∂y + Δ⁻¹(∂μ/∂x ∂ω/∂y - ∂μ/∂y ∂ω/∂x) + Δμ/Re,
#
# as the LinearisedExTerm of OpenKolmogorovFlow in its adjoint mode.
#
# The term is TangentTerm read backwards, each step replaced by its transpose in realdots and in
# the mean over the grid: a multiplier in Fourier space by its conjugate, a product with u or v by
# itself, and each transform by the other, since the mean over the grid of the values of a field
# times any values is the inner product of the field with the modes of those values. It is then
# the exact transpose of TangentTerm, to rounding, and, as the products have no aliasing, the
# operator above.

# the transpose of advection!: the transforms of the two products, P of u v′ + u′ v and Q of
# 2 (v v′ - u u′), from the adjoint fields Λ, each with the multiplier it has in advection!
function advection_transpose!(P::FTFields, Q::FTFields, Λ::FTFields)
    # ---- -(∂²/∂x² - ∂²/∂y²) is j² - k² and -∂²/∂x∂y is j k on every mode, real multipliers that
    #      are their own transposes ----
    P .= (Λ.j.^2 .- Λ.k.^2) .* Λ
    Q .= Λ.j .* Λ.k .* Λ

    return P, Q
end

# the transpose of the velocity of a perturbation, u′ = -∂ψ′/∂y and v′ = ∂ψ′/∂x with Δψ′ = λ,
# from the transforms U′ and V′ of what stands in its place: Δ⁻¹(∂U′/∂y - ∂V′/∂x)
function velocity_transpose!(dΛdt::FTFields, U′::FTFields, V′::FTFields)
    # ---- ∂/∂x and ∂/∂y are i j and i k, whose conjugates are -i j and -i k, and Δ⁻¹ is real: the
    #      transpose of -∂/∂y Δ⁻¹ is ∂/∂y Δ⁻¹, and that of ∂/∂x Δ⁻¹ is -∂/∂x Δ⁻¹ ----
    ddy!(U′, U′)
    ddx!(V′, V′)
    dΛdt .= U′ .- V′

    return invlaplacian!(dΛdt, dΛdt)
end

"""
    AdjointTerm(X, m)

The transpose of the linearised advection of [`TangentTerm`](@ref) in the inner product of the
fields, [`realdots`](@ref), for batches of fields like `X`, with its transforms and its storage.
Called as `eq(t, Ω, Λ, dΛdt)`, as Flows.jl calls the linearised term of an adjoint flow on its
cached stages, it writes into `dΛdt`

    u ∂λ/∂x + v ∂λ/∂y + Δ⁻¹(∂λ/∂x ∂ω/∂y - ∂λ/∂y ∂ω/∂x)

for every adjoint field of the batch `Λ` about the field of the batch `Ω` in the same place, and
returns it. Field by field, `realdots(Et(t, Ω, dΩdt, V, similar(V)), W)` equals
`realdots(V, Ea(t, Ω, W, similar(W)))` to rounding, `Et` the tangent term and `Ea` this one.

# Arguments
- `X::FTFields`: a batch of fields, for its size and its device.
- `m::Int`: the grid of the products has `2m+2` points along each direction, at least `3n + 1`.
"""
struct AdjointTerm{FT<:FTFields, F<:Fields, IT, FFT}
        U::FT  # a velocity along x in Fourier space, then the transform of u′
        V::FT  # a velocity along y, then the transform of v′
        P::FT  # the transpose of the advection, in the place of the transform of u v′ + u′ v
        Q::FT  # and in the place of the transform of 2 (v v′ - u u′)
        u::F   # the velocity of the fields along x, on the grid
        v::F   # the velocity of the fields along y
       u′::F   # the transpose of the products, in the place of the velocity of the
               # perturbations along x
       v′::F   # and in the place of the velocity along y
        p::F   # P on the grid
        q::F   # Q on the grid
    ifft!::IT  # from the fields to the grid
     fft!::FFT # from the grid to the fields

    function AdjointTerm(X::FTFields, m::Int)
        # ---- a batch on the grid, for the type of the others ----
        u = Fields(X, m)

        # ---- the transforms ----
        ifft! = InverseFFT!(X, m)
        fft!  = ForwardFFT!(X, m)

        return new{typeof(X), typeof(u), typeof(ifft!), typeof(fft!)}(similar(X),
                                                                     similar(X),
                                                                     similar(X),
                                                                     similar(X),
                                                                     u,
                                                                     similar(u),
                                                                     similar(u),
                                                                     similar(u),
                                                                     similar(u),
                                                                     similar(u),
                                                                     ifft!,
                                                                     fft!)
    end
end

function (eq::AdjointTerm)(t::Real, Ω::FTFields, Λ::FTFields, dΛdt::FTFields)
    u, v   = eq.u, eq.v
    u′, v′ = eq.u′, eq.v′
    p, q   = eq.p, eq.q

    # ---- the velocity of the fields, on the grid, as in the tangent term ----
    velocity!(u, v, Ω, eq.U, eq.V, eq.ifft!)

    # ---- the transpose of the advection ----
    advection_transpose!(eq.P, eq.Q, Λ)

    # ---- the transpose of the forward transforms of the products: the inverse ones ----
    eq.ifft!(p, eq.P)
    eq.ifft!(q, eq.Q)

    # ---- the transpose of the products u v′ + u′ v and 2 (v v′ - u u′) in (u′, v′), point by point ----
    u′ .= v .* p .- 2 .* u .* q
    v′ .= u .* p .+ 2 .* v .* q

    # ---- the transpose of the inverse transforms of the velocity: the forward ones ----
    eq.fft!(eq.U, u′)
    eq.fft!(eq.V, v′)

    # ---- the transpose of the velocity of the perturbations ----
    return velocity_transpose!(dΛdt, eq.U, eq.V)
end
