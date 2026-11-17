# The explicit term of the tangent equation, the vorticity equation linearised about a field ω,
#
#     ∂λ/∂t = -(u ∂λ/∂x + v ∂λ/∂y + u′ ∂ω/∂x + v′ ∂ω/∂y) + Δλ/Re,
#
# with (u, v) the velocity of ω and (u′, v′) that of the perturbation λ, in the call interface of
# Flows.jl for an equation coupled to its tangent, as the LinearisedExTerm of OpenKolmogorovFlow
# in its tangent mode. The advection is the derivative of the form of Basdevant in nonlinear.jl:
# the products u v and v² - u² become u v′ + u′ v and 2 (v v′ - u u′).

"""
    TangentTerm(X, m)

The linearised advection of the tangent equation, for batches of fields like `X`, with its
transforms and its storage. Called as `eq(t, Ω, dΩdt, Λ, dΛdt)`, as Flows.jl calls the second
term of a coupled system, it writes into `dΛdt`

    -(u ∂λ/∂x + v ∂λ/∂y + u′ ∂ω/∂x + v′ ∂ω/∂y)

for every perturbation of the batch `Λ` about the field of the batch `Ω` in the same place, and
returns it. It takes nothing from the term of the equation: the velocity of `Ω` is computed here.

# Arguments
- `X::FTFields`: a batch of fields, for its size and its device.
- `m::Int`: the grid of the products has `2m+2` points along each direction, at least `3n + 1`.
"""
struct TangentTerm{FT<:FTFields, F<:Fields, IT, FFT}
        U::FT  # a velocity along x, in Fourier space
        V::FT  # a velocity along y
        P::FT  # the transform of u v′ + u′ v
        Q::FT  # the transform of 2 (v v′ - u u′)
        u::F   # the velocity of the fields along x, on the grid
        v::F   # the velocity of the fields along y
       u′::F   # the velocity of the perturbations along x
       v′::F   # the velocity of the perturbations along y
        p::F   # the product u v′ + u′ v
        q::F   # the product 2 (v v′ - u u′)
    ifft!::IT  # from the fields to the grid
     fft!::FFT # from the grid to the fields

    function TangentTerm(X::FTFields, m::Int)
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

function (eq::TangentTerm)(t::Real, Ω::FTFields, dΩdt::FTFields,
                                    Λ::FTFields, dΛdt::FTFields)
    u, v   = eq.u, eq.v
    u′, v′ = eq.u′, eq.v′
    p, q   = eq.p, eq.q

    # ---- the velocity of the fields, on the grid ----
    velocity!(u, v, Ω, eq.U, eq.V, eq.ifft!)

    # ---- the velocity of the perturbations ----
    velocity!(u′, v′, Λ, eq.U, eq.V, eq.ifft!)

    # ---- the two products of Basdevant linearised, point by point ----
    p .= u .* v′ .+ u′ .* v
    q .= 2 .* (v .* v′ .- u .* u′)

    # ---- and their transforms ----
    eq.fft!(eq.P, p)
    eq.fft!(eq.Q, q)

    # ---- the advection ----
    return advection!(dΛdt, eq.P, eq.Q)
end

"""
    ForcedTangentTerm(tangent, Re)

The explicit term of the tangent equation forced by the derivative of the right-hand side of the
vorticity equation in the Reynolds number,

    ∂f/∂Re = -Δω/Re²,

called as a [`TangentTerm`](@ref). From a zero perturbation the tangent equation with this term
gives the derivative of a field in the Reynolds number.

# Arguments
- `tangent::TangentTerm`: the term of the tangent equation.
- `Re::Real`: the Reynolds number.
"""
struct ForcedTangentTerm{T<:TangentTerm}
    tangent::T       # the term of the tangent equation
         Re::Float64 # the Reynolds number
end

function (eq::ForcedTangentTerm)(t::Real, Ω::FTFields, dΩdt::FTFields,
                                          Λ::FTFields, dΛdt::FTFields)
    # ---- the linearised advection ----
    eq.tangent(t, Ω, dΩdt, Λ, dΛdt)

    # ---- the forcing -Δω/Re²: the Laplacian is -(j² + k²) on every mode ----
    dΛdt .+= (Ω.j.^2 .+ Ω.k.^2) .* Ω ./ eq.Re^2

    return dΛdt
end
