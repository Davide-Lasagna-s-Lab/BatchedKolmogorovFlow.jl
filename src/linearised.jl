# The linearised equations, about a field ω: the tangent equation, forward in time, and its
# adjoint, backward, its transpose in the inner product of the fields.
#
# The explicit term of the vorticity equation together with that of its tangent equation, the
# equation linearised about a field ω,
#
#     ∂λ/∂t = -(u ∂λ/∂x + v ∂λ/∂y + u′ ∂ω/∂x + v′ ∂ω/∂y) + Δλ/Re,
#
# with (u, v) the velocity of ω and (u′, v′) that of the perturbation λ, in the call interface of
# Flows.jl, as the ForwardExplicitTerm of OpenKolmogorovFlow with its LinearisedExTerm in the
# tangent mode. The advection of a perturbation is the derivative of the form of Basdevant in
# nonlinear.jl: the products u v and v² - u² become u v′ + u′ v and 2 (v v′ - u u′).
#
# The bundle. A field and its perturbations are advanced as one batch, in which every field is
# followed by its k′ perturbations,
#
#     Z = (ω₁, λ₁¹, …, λ₁ᵏ′,   ω₂, λ₂¹, …, λ₂ᵏ′,   …,   ω_B, λ_B¹, …, λ_Bᵏ′),
#
# an FTFields of (1 + k′) B fields. The steps of a stage that are the same for a field and for a
# perturbation, the velocity, the transforms, the derivatives and the implicit solve, act on the
# whole batch at once. The fields are advanced once whatever k′ is, and only the products differ
# between the fields and their perturbations.
#
# Where it comes from. tangent! takes the B fields in a batch X, and their perturbations in V,
# a batch like X for one perturbation of every field, or a vector of k′ of them, the directions
# of a subspace step, V[k₁] holding perturbation k₁ of every field. With B = 3 fields and k′ = 2,
# the slices along the last dimension are
#
#     X       = (ω₁,  ω₂,  ω₃)
#     V[1]    = (λ₁¹, λ₂¹, λ₃¹)
#     V[2]    = (λ₁², λ₂², λ₃²)
#
# and the bundle holds them field by field, every field followed by its own perturbations,
#
#     Z       = (ω₁, λ₁¹, λ₁²,   ω₂, λ₂¹, λ₂²,   ω₃, λ₃¹, λ₃²),
#
# 9 fields. Seen as an array (n+1, 2n+1, 1 + k′, B) = (n+1, 2n+1, 3, 3), the third index is the
# place in the group of a field, 1 for the field and 1 + k₁ for its perturbation k₁, and the
# fourth the field: field i is [:, :, 1, i], and its perturbation k₁ is [:, :, 1 + k₁, i]. So
#
#     X[:, :, i]       goes to   Z[:, :, 1, i]         the fields
#     V[k₁][:, :, i]   goes to   Z[:, :, 1 + k₁, i]    the perturbation k₁ of every field
#
# _bundle! makes this copy, one for the fields and one for every k₁, each of whole batches, and
# _unbundle! the inverse. With the fields first in their groups, the fields of a stage are
# [:, :, 1:1, :], which broadcasts against the k′ perturbations [:, :, 2:end, :]: the products of
# the tangent term are written once for all the directions.

# The two parts of a bundle of fields with their k′ perturbations, on its entries, seen as an
# array (·, ·, 1 + k′, B): the fields, (·, ·, 1, B), and the perturbations, (·, ·, k′, B), so that
# the fields broadcast against their perturbations, and place k₁ of the second is perturbation
# k₁. Views, with no copy. For a bundle in Fourier space or on the grid.
function _parts(Z::AbstractArray, k′::Int)
    A = reshape(parent(Z), size(Z, 1), size(Z, 2), 1 + k′, :)

    return view(A, :, :, 1:1, :), view(A, :, :, 2:(1 + k′), :)
end

# The fields `fields` of X, all of them or those of the chunk of a task, and their perturbations
# in V written into the bundle Z, as in the example at the top: X[:, :, i] into Z[:, :, 1, i], and
# V[k₁][:, :, i] into Z[:, :, 1 + k₁, i]. V holds k′ batches like X, a vector or a tuple of them:
# V[k₁] is perturbation k₁ of every field, as MultipleShooting.jl hands over the directions of a
# subspace step. Z holds as many groups as the fields copied.
function _bundle!(Z::FTFields, X::FTFields, V, fields::UnitRange{Int}=1:size(X, 3))
    k′   = length(V)
    Ω, Λ = _parts(Z, k′)

    # ---- the fields, first in their groups ----
    #      Z[:, :, 1, i] = X[:, :, i]
    view(Ω, :, :, 1, :) .= view(parent(X), :, :, fields)

    # ---- the perturbation k₁ of every field, in place 1 + k₁ of its group, one batch at a
    #      time ----
    #      Z[:, :, 1 + k₁, i] = V[k₁][:, :, i]
    for k₁ in 1:k′
        view(Λ, :, :, k₁, :) .= view(parent(V[k₁]), :, :, fields)
    end

    return Z
end

# The inverse of _bundle!: the fields of the bundle Z written into the fields `fields` of X, and
# their perturbations into the same places of the k′ batches of V, Z[:, :, 1, i] into
# X[:, :, i] and Z[:, :, 1 + k₁, i] into V[k₁][:, :, i]
function _unbundle!(X::FTFields, V, Z::FTFields, fields::UnitRange{Int}=1:size(X, 3))
    k′   = length(V)
    Ω, Λ = _parts(Z, k′)

    # ---- the fields, from the first place of their groups ----
    #      X[:, :, i] = Z[:, :, 1, i]
    view(parent(X), :, :, fields) .= view(Ω, :, :, 1, :)

    # ---- the perturbation k₁ of every field, from place 1 + k₁ of its group ----
    #      V[k₁][:, :, i] = Z[:, :, 1 + k₁, i]
    for k₁ in 1:k′
        view(parent(V[k₁]), :, :, fields) .= view(Λ, :, :, k₁, :)
    end

    return V
end

"""
    TangentTerm(Z, k′, m, kf)

The explicit term of the vorticity equation and of its tangent equation, for bundles like `Z`, in
which every field is followed by its `k′` perturbations, with its transforms and its storage.
Called as `eq(t, Z, dZdt)`, as Flows.jl calls an explicit term, it writes into `dZdt`, in the
place of every field,

    -(u ∂ω/∂x + v ∂ω/∂y) - kf cos(kf y),

as [`ExplicitTerm`](@ref), and in the place of every perturbation of that field

    -(u ∂λ/∂x + v ∂λ/∂y + u′ ∂ω/∂x + v′ ∂ω/∂y),

and returns it. The velocity of a field is computed once, whatever the number of its
perturbations.

# Arguments
- `Z::FTFields`: a bundle, for its size and its device: `(1 + k′) B` fields.
- `k′::Int`: the perturbations of every field.
- `m::Int`: the grid of the products has `2m+2` points along each direction, at least `3n + 1`.
- `kf::Int`: the wavenumber of the forcing.
"""
struct TangentTerm{FT<:FTFields, F<:Fields, IT, FFT}
        U::FT  # the velocity along x, in Fourier space
        V::FT  # the velocity along y
        P::FT  # the transform of the first product
        Q::FT  # the transform of the second product
        u::F   # the velocity along x, on the grid
        v::F   # the velocity along y
        p::F   # the first product: u v for a field, u v′ + u′ v for a perturbation
        q::F   # the second product: v² - u² for a field, 2 (v v′ - u u′) for a perturbation
    ifft!::IT  # from the bundle to the grid
     fft!::FFT # from the grid to the bundle
       k′::Int # the perturbations of every field
       kf::Int # the wavenumber of the forcing

    function TangentTerm(Z::FTFields, k′::Int, m::Int, kf::Int)
        # ---- input checks ----
        size(Z, 3) % (1 + k′) == 0 ||
            throw(ArgumentError("a bundle holds 1 + k′ fields for every field"))

        # ---- a batch on the grid, for the type of the others ----
        u = Fields(Z, m)

        # ---- the transforms ----
        ifft! = InverseFFT!(Z, m)
        fft!  = ForwardFFT!(Z, m)

        return new{typeof(similar(Z)), typeof(u), typeof(ifft!), typeof(fft!)}(similar(Z),
                                                                              similar(Z),
                                                                              similar(Z),
                                                                              similar(Z),
                                                                              u,
                                                                              similar(u),
                                                                              similar(u),
                                                                              similar(u),
                                                                              ifft!,
                                                                              fft!,
                                                                              k′,
                                                                              kf)
    end
end

function (eq::TangentTerm)(t::Real, Z::FTFields{n}, dZdt::FTFields{n}) where {n}
    U, V = eq.U, eq.V
    P, Q = eq.P, eq.Q

    # ---- in Fourier space: the velocity of the fields and of their perturbations, U = -∂ψ/∂y
    #      and V = ∂ψ/∂x, with the stream function ψ = Δ⁻¹ω of each ----
    ddy!(U, Z)
    invlaplacian!(U, U)
    U .*= -1
    ddx!(V, Z)
    invlaplacian!(V, V)

    # ---- to the grid ----
    eq.ifft!(eq.u, U)
    eq.ifft!(eq.v, V)

    # ---- on the grid: the fields, and their perturbations ----
    u, u′ = _parts(eq.u, eq.k′)
    v, v′ = _parts(eq.v, eq.k′)
    p, p′ = _parts(eq.p, eq.k′)
    q, q′ = _parts(eq.q, eq.k′)

    # ---- the two products of Basdevant for the fields, point by point ----
    p .= u .* v
    q .= v.^2 .- u.^2

    # ---- and linearised for the perturbations, each with the velocity of its field ----
    p′ .= u .* v′ .+ u′ .* v
    q′ .= 2 .* (v .* v′ .- u .* u′)

    # ---- back to Fourier space ----
    eq.fft!(P, eq.p)
    eq.fft!(Q, eq.q)

    # ---- in Fourier space: the advection of the fields and of their perturbations, from the
    #      derivatives of the two products, as in the explicit term of the equation ----
    dZdt .= (P.j.^2 .- P.k.^2) .* P .+ P.j .* P.k .* Q

    # ---- the forcing -kf cos(kf y) of the fields: -kf/2 at the two modes j = 0, k = ±kf ----
    F, _ = _parts(dZdt, eq.k′)
    view(F, 1, _column(eq.kf, n), 1, :)  .-= eq.kf/2
    view(F, 1, _column(-eq.kf, n), 1, :) .-= eq.kf/2

    return dZdt
end

"""
    ForcedTangentTerm(tangent, Re)

The explicit term of the equation and of its tangent equation forced by the derivative of the
right-hand side of the vorticity equation in the Reynolds number,

    ∂f/∂Re = -Δω/Re²,

called as a [`TangentTerm`](@ref). From a zero perturbation the tangent equation with this term
gives the derivative of a field in the Reynolds number.

# Arguments
- `tangent::TangentTerm`: the term of the equation and of its tangent.
- `Re::Real`: the Reynolds number.
"""
struct ForcedTangentTerm{T<:TangentTerm}
    tangent::T       # the term of the equation and of its tangent
         Re::Float64 # the Reynolds number
end

function (eq::ForcedTangentTerm)(t::Real, Z::FTFields, dZdt::FTFields)
    # ---- the advection, of the fields and linearised ----
    eq.tangent(t, Z, dZdt)

    # ---- the fields, and the term of their perturbations ----
    Ω, _   = _parts(Z, eq.tangent.k′)
    _, dΛ  = _parts(dZdt, eq.tangent.k′)

    # ---- the forcing -Δω/Re²: the Laplacian is -(j² + k²) on every mode ----
    dΛ .+= (Z.j.^2 .+ Z.k.^2) .* Ω ./ eq.Re^2

    return dZdt
end


# ---------------------------------------------------------------------------- #
# the adjoint, the transpose of the linearised advection                       #
# ---------------------------------------------------------------------------- #

# The explicit term of the adjoint equation, the transpose of the linearised advection above, in
# the call interface of Flows.jl for the adjoint of a discrete flow. In the inner
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
# The term is TangentTerm read backwards, each step replaced by its transpose in dots and in
# the mean over the grid: a multiplier in Fourier space by its conjugate, a product with u or v by
# itself, and each transform by the other, since the mean over the grid of the values of a field
# times any values is the inner product of the field with the modes of those values. It is then
# the exact transpose of TangentTerm, to rounding, and, as the products have no aliasing, the
# operator above.

"""
    AdjointTerm(X, m)

The transpose of the linearised advection of [`TangentTerm`](@ref) in the inner product of the
fields, [`dots`](@ref), for batches of fields like `X`, with its transforms and its storage.
Called as `eq(t, Ω, Λ, dΛdt)`, as Flows.jl calls the linearised term of an adjoint flow on its
cached stages, it writes into `dΛdt`

    u ∂λ/∂x + v ∂λ/∂y + Δ⁻¹(∂λ/∂x ∂ω/∂y - ∂λ/∂y ∂ω/∂x)

for every adjoint field of the batch `Λ` about the field of the batch `Ω` in the same place, and
returns it. Field by field, `dots(Et(t, Ω, dΩdt, V, similar(V)), W)` equals
`dots(V, Ea(t, Ω, W, similar(W)))` to rounding, `Et` the tangent term and `Ea` this one.

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
