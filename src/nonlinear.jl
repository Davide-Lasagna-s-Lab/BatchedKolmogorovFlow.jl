# The explicit term of the vorticity equation, advection and forcing, in the call interface of
# Flows.jl, as the ForwardExplicitTerm of OpenKolmogorovFlow. The equation is
#
#     ∂ω/∂t = -(u ∂ω/∂x + v ∂ω/∂y) - kf cos(kf y) + Δω/Re,
#
# with the velocity (u, v) = (-∂ψ/∂y, ∂ψ/∂x) of the stream function, Δψ = ω. The advection is
# written as in Basdevant (1983),
#
#     u ∂ω/∂x + v ∂ω/∂y = (∂²/∂x² - ∂²/∂y²)(u v) + ∂²/∂x∂y (v² - u²),
#
# an identity for a velocity without divergence. It needs the velocity alone on the grid, two
# inverse transforms and two forward ones, where the left side needs four and one.

# the velocity (u, v) of the fields Ω on the grid, with U and V its storage in Fourier space
function velocity!(u::Fields, v::Fields, Ω::FTFields, U::FTFields, V::FTFields, ifft!::InverseFFT!)
    # ---- u = -∂ψ/∂y, with the stream function ψ = Δ⁻¹ω ----
    ddy!(U, Ω)
    invlaplacian!(U, U)
    U .*= -1

    # ---- v = ∂ψ/∂x ----
    ddx!(V, Ω)
    invlaplacian!(V, V)

    # ---- on the grid ----
    ifft!(u, U)
    ifft!(v, V)

    return u, v
end

# the advection of the fields from the transforms P of u v and Q of v² - u²,
# -(∂²/∂x² - ∂²/∂y²)(u v) - ∂²/∂x∂y (v² - u²)
function advection!(dΩdt::FTFields, P::FTFields, Q::FTFields)
    # ---- ∂²/∂x² is -j², ∂²/∂y² is -k² and ∂²/∂x∂y is -j k on every mode ----
    dΩdt .= (P.j.^2 .- P.k.^2) .* P .+ P.j .* P.k .* Q

    return dΩdt
end

"""
    ExplicitTerm(X, m, kf)

The advection and the forcing of the vorticity equation, for batches of fields like `X`, with
its transforms and its storage. Called as `eq(t, Ω, dΩdt)`, as Flows.jl calls an explicit term,
it writes into `dΩdt`

    -(u ∂ω/∂x + v ∂ω/∂y) - kf cos(kf y)

for every field of the batch `Ω`, and returns it.

# Arguments
- `X::FTFields`: a batch of fields, for its size and its device.
- `m::Int`: the grid of the products has `2m+2` points along each direction, at least `3n + 1`.
- `kf::Int`: the wavenumber of the forcing.
"""
struct ExplicitTerm{FT<:FTFields, F<:Fields, IT, FFT}
        U::FT  # the velocity along x, in Fourier space
        V::FT  # the velocity along y
        P::FT  # the transform of u v
        Q::FT  # the transform of v² - u²
        u::F   # the velocity along x, on the grid
        v::F   # the velocity along y
        p::F   # the product u v
        q::F   # the product v² - u²
    ifft!::IT  # from the fields to the grid
     fft!::FFT # from the grid to the fields
       kf::Int # the wavenumber of the forcing

    function ExplicitTerm(X::FTFields, m::Int, kf::Int)
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
                                                                     ifft!,
                                                                     fft!,
                                                                     kf)
    end
end

function (eq::ExplicitTerm)(t::Real, Ω::FTFields{n}, dΩdt::FTFields{n}) where {n}
    u, v = eq.u, eq.v
    p, q = eq.p, eq.q

    # ---- the velocity on the grid ----
    velocity!(u, v, Ω, eq.U, eq.V, eq.ifft!)

    # ---- the two products, point by point ----
    p .= u .* v
    q .= v.^2 .- u.^2

    # ---- and their transforms ----
    eq.fft!(eq.P, p)
    eq.fft!(eq.Q, q)

    # ---- the advection ----
    advection!(dΩdt, eq.P, eq.Q)

    # ---- the forcing -kf cos(kf y): -kf/2 at the two modes j = 0, k = ±kf ----
    view(parent(dΩdt), 1, _column(eq.kf, n), :)  .-= eq.kf/2
    view(parent(dΩdt), 1, _column(-eq.kf, n), :) .-= eq.kf/2

    return dΩdt
end
