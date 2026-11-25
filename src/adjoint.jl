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
    U, V   = eq.U, eq.V
    P, Q   = eq.P, eq.Q
    u, v   = eq.u, eq.v
    u′, v′ = eq.u′, eq.v′
    p, q   = eq.p, eq.q

    # ---- the velocity of the fields, in Fourier space and then on the grid, as in the tangent
    #      term ----
    ddy!(U, Ω)
    invlaplacian!(U, U)
    U .*= -1
    ddx!(V, Ω)
    invlaplacian!(V, V)
    eq.ifft!(u, U)
    eq.ifft!(v, V)

    # ---- the transpose of the advection: the adjoint fields in the places of the transforms of
    #      the two products, each with the multiplier it has there, j² - k² and j k, which are
    #      real and their own transposes ----
    P .= (Λ.j.^2 .- Λ.k.^2) .* Λ
    Q .= Λ.j .* Λ.k .* Λ

    # ---- the transpose of the forward transforms of the products: the inverse ones ----
    eq.ifft!(p, P)
    eq.ifft!(q, Q)

    # ---- the transpose of the products u v′ + u′ v and 2 (v v′ - u u′) in (u′, v′), point by point ----
    u′ .= v .* p .- 2 .* u .* q
    v′ .= u .* p .+ 2 .* v .* q

    # ---- the transpose of the inverse transforms of the velocity: the forward ones ----
    eq.fft!(U, u′)
    eq.fft!(V, v′)

    # ---- the transpose of the velocity of a perturbation, u′ = -∂ψ′/∂y and v′ = ∂ψ′/∂x with
    #      Δψ′ = λ: Δ⁻¹(∂U/∂y - ∂V/∂x). The conjugates of i j and i k are -i j and -i k and Δ⁻¹
    #      is real, so that the transpose of -∂/∂y Δ⁻¹ is ∂/∂y Δ⁻¹ and that of ∂/∂x Δ⁻¹ is
    #      -∂/∂x Δ⁻¹ ----
    ddy!(U, U)
    ddx!(V, V)
    dΛdt .= U .- V

    return invlaplacian!(dΛdt, dΛdt)
end
