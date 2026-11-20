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
# an FTFields of (1 + k′) B fields. Seen as an array (n+1, 2n+1, 1 + k′, B), field i is
# [:, :, 1, i] and its perturbations are [:, :, 2:end, i]. The steps of a stage that are the same
# for a field and for a perturbation, the velocity, the transforms, the derivatives and the
# implicit solve, act on the whole batch at once. The fields are advanced once whatever k′ is,
# and only the products differ between the fields and their perturbations.

# The two parts of a bundle of fields with their k′ perturbations, on its entries: the fields, an
# array (·, ·, 1, B), and the perturbations, (·, ·, k′, B), so that the fields broadcast against
# their perturbations. For a bundle in Fourier space or on the grid.
function _parts(Z::AbstractArray, k′::Int)
    A = reshape(parent(Z), size(Z, 1), size(Z, 2), 1 + k′, :)

    return view(A, :, :, 1:1, :), view(A, :, :, 2:(1 + k′), :)
end

# The fields X and their perturbations V written into the bundle Z. V holds k′ blocks, one after
# the other, each of as many perturbations as X has fields: block a is perturbation a of every
# field, as MultipleShooting.jl packs the directions of a subspace step.
function _bundle!(Z::FTFields, X::FTFields, V::FTFields)
    B      = size(X, 3)
    k′     = size(V, 3) ÷ B
    Ω, Λ   = _parts(Z, k′)
    blocks = reshape(parent(V), size(V, 1), size(V, 2), B, k′)

    # ---- every field, and block a of the perturbations in place a after the fields ----
    view(Ω, :, :, 1, :) .= parent(X)
    for a in 1:k′
        view(Λ, :, :, a, :) .= view(blocks, :, :, :, a)
    end

    return Z
end

# The fields of the bundle Z written into X, and their perturbations into the k′ blocks of V
function _unbundle!(X::FTFields, V::FTFields, Z::FTFields)
    B      = size(X, 3)
    k′     = size(V, 3) ÷ B
    Ω, Λ   = _parts(Z, k′)
    blocks = reshape(parent(V), size(V, 1), size(V, 2), B, k′)

    # ---- every field, and place a after the fields into block a of the perturbations ----
    parent(X) .= view(Ω, :, :, 1, :)
    for a in 1:k′
        view(blocks, :, :, :, a) .= view(Λ, :, :, a, :)
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
    # ---- the velocity of the fields and of their perturbations, on the grid ----
    velocity!(eq.u, eq.v, Z, eq.U, eq.V, eq.ifft!)

    # ---- the fields, and their perturbations ----
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

    # ---- their transforms, and the advection ----
    eq.fft!(eq.P, eq.p)
    eq.fft!(eq.Q, eq.q)
    advection!(dZdt, eq.P, eq.Q)

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
