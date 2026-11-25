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

"""
    ExplicitTerm(X, m, kf)

The advection and the forcing of the vorticity equation, for batches of fields like `X`, with
its transforms and its arrays. Called as `eq(t, Ω, dΩdt)`, as Flows.jl calls an explicit term,
it writes into `dΩdt`

    -(u ∂ω/∂x + v ∂ω/∂y) - kf cos(kf y)

for every field of the batch `Ω`, and returns it.

# Arguments
- `X::FTFields`: a batch of fields, for its size and its device.
- `m::Int`: the grid of the products has `2m+2` points along each direction, at least `3n + 1`.
- `kf::Int`: the wavenumber of the forcing.
"""
struct ExplicitTerm{FT<:FTFields, F<:Fields, IT, FFT}
       UV::FT  # the velocity in Fourier space, along x and along y in the two halves, 2B fields
       PQ::FT  # the transforms of the two products, u v and v² - u², in the two halves
       uv::F   # the velocity on the grid, u and v
       pq::F   # the two products, u v and v² - u²
    ifft!::IT  # from the velocity to the grid, both halves in one transform
     fft!::FFT # from the products to Fourier space, both halves in one transform
       kf::Int # the wavenumber of the forcing

    function ExplicitTerm(X::FTFields, m::Int, kf::Int)
        # ---- the batches of the two components and of the two products, twice the fields of X,
        #      in Fourier space and on the grid ----
        UV = _twice(X)
        uv = Fields(UV, m)

        # ---- the transforms of such batches ----
        ifft! = InverseFFT!(UV, m)
        fft!  = ForwardFFT!(UV, m)

        return new{typeof(UV), typeof(uv), typeof(ifft!), typeof(fft!)}(UV,
                                                                       similar(UV),
                                                                       uv,
                                                                       similar(uv),
                                                                       ifft!,
                                                                       fft!,
                                                                       kf)
    end
end

function (eq::ExplicitTerm)(t::Real, Ω::FTFields, dΩdt::FTFields)
    # ---- the velocity, and on the grid, its two components in one transform ----
    _velocity!(eq.UV, Ω)
    eq.ifft!(eq.uv, eq.UV)

    # ---- on the grid: the two products, point by point ----
    u, v = _halves(eq.uv)
    p, q = _halves(eq.pq)
    p .= u .* v
    q .= v.^2 .- u.^2

    # ---- back to Fourier space, the two products in one transform ----
    eq.fft!(eq.PQ, eq.pq)
    P, Q = _halves(eq.PQ)

    # ---- in Fourier space: the advection, -(∂²/∂x² - ∂²/∂y²)(u v) - ∂²/∂x∂y (v² - u²), where
    #      ∂²/∂x² is -j², ∂²/∂y² is -k² and ∂²/∂x∂y is -j k on every mode, and the forcing
    #      -kf cos(kf y), -kf/2 at the two modes j = 0, k = ±kf, in one broadcast ----
    dΩdt .= (P.j.^2 .- P.k.^2) .* P .+ P.j .* P.k .* Q .-
            (eq.kf/2) .* (iszero.(P.j) .& (abs.(P.k) .== eq.kf))

    return dΩdt
end

# The velocity of the fields Ω in Fourier space into the two halves of UV, U = -∂ψ/∂y and
# V = ∂ψ/∂x with the stream function ψ = Δ⁻¹ω: i k ω/(j² + k²) and -i j ω/(j² + k²), zero at the
# mean, one broadcast each
function _velocity!(UV::FTFields, Ω::FTFields)
    U, V = _halves(UV)

    U .= im .* Ω.k .* Ω .* _over.(Ω.j.^2 .+ Ω.k.^2)
    V .= .-im .* Ω.j .* Ω .* _over.(Ω.j.^2 .+ Ω.k.^2)

    return UV
end
