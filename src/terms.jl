# The explicit and implicit terms of the vorticity equation, and the explicit
# term of its tangent equation, for a batch of fields, in the call interface of
# Flows.jl. Field by field, they equal ForwardEquation and
# LinearisedEquation(TangentMode()) of OpenKolmogorovFlow to round-off.
#
# The nonlinear term is written as Basdevant (1983), with the velocity (u, v),
#
#     -(u ∂ω/∂x + v ∂ω/∂y) = -(∂²/∂x² - ∂²/∂y²)(u v) - ∂²/∂x∂y (v² - u²),
#
# an identity for divergence-free fields. OKF evaluates the left side, with four
# inverse transforms and one forward. The right side needs two and two. Both are
# the same trigonometric polynomial before the truncation, and the grid dealiases
# quadratic products exactly, so the two discrete terms agree to round-off. The
# tangent is the derivative of the right side. It takes the velocity of the base
# from the term of the equation, which Flows calls first at the same state, and
# adds two and two transforms, against eight and one in OKF.
#
# The terms act on compact batches, the active modes alone. A compact batch is a complex array
# (n+1, 2n+1, B), j = 0, …, n down the rows and k = 0, …, n, -n, …, -1 across the columns: 45% of the
# stored entries of a field of OKF, so that every kernel of the time stepper moves less memory. The
# real-to-complex transforms of cuFFT halve the first dimension of the region, while the fields of
# OKF halve the second (x). The transforms therefore work with x along the first dimension, as the
# compact batches do, and only the conversions to and from the fields transpose.

# a host array copied to the device of `template`
_device(template::AbstractArray, h::AbstractArray) =
    copyto!(similar(template, eltype(h), size(h)), h)

# signed wavenumbers k down the rows (FFT order) and j ≥ 0 across the columns
_ks(m::Int) = [r <= m ? r : r - (2m + 2) for r in 0:(2m + 1)]
_js(m::Int) = collect(0:(m + 1))

# the signed wavenumbers k along the columns of a compact batch
_kc(n::Int) = [0:n; -n:-1]


# ---------------------------------------------------------------------------- #
# The implicit term, Δ/Re, diagonal in Fourier space
# ---------------------------------------------------------------------------- #
struct ImplicitTerm{R<:AbstractMatrix{Float64}}
     Re::Float64 # Reynolds number
    lap::R       # -(j² + k²), (n+1, 2n+1)

    function ImplicitTerm(template::AbstractArray, n::Int, Re::Real)
        lap = [-Float64(j^2 + k^2) for j in 0:n, k in _kc(n)]
        R   = typeof(_device(template, lap))
        return new{R}(Float64(Re), _device(template, lap))
    end
end

# the Laplacian over Re, as `mul!` of OKF's ImplicitTerm
mul!(out::AbstractArray, L::ImplicitTerm, X::AbstractArray) =
    (out .= X .* L.lap .* (1/L.Re); out)

# Z = (I - c Δ/Re)⁻¹ Y, as `Flows.ImcA!` of OKF's ImplicitTerm
Flows.ImcA!(L::ImplicitTerm, c::Real, Y::AbstractArray, Z::AbstractArray) =
    (Z .= Y ./ (1 .- (c/L.Re) .* L.lap); Z)

# Z = (I - c Δ/Re)⁻¹ (Δ/Re) Y: the two calls above in one kernel
_implicit!(Z::AbstractArray, L::ImplicitTerm, c::Real, Y::AbstractArray) =
    (Z .= Y .* (1/L.Re) .* L.lap ./ (1 .- (c/L.Re) .* L.lap); Z)


# ---------------------------------------------------------------------------- #
# The explicit terms for a batch of B fields, advection and forcing of the
# equation, or the linearised advection of the tangent equation
# ---------------------------------------------------------------------------- #
struct ExplicitTerm{C4, R4, PB, PF, R, U}
       F̂::C4               # velocity to transform, (m+2, 2m+2, B, 2)
       f::R4               # the same in physical space, (2m+2, 2m+2, B, 2)
       w::R4               # the two products in physical space, (2m+2, 2m+2, B, 2)
       ŵ::C4               # their transforms, (m+2, 2m+2, B, 2)
    ifft::PB               # inverse transform of F̂ into f, unnormalised as OKF
     fft::PF               # forward transform of w into ŵ, unnormalised
       a::R                # û = i a Ω̂: k/(j² + k²), compact, zero at the mean
       b::R                # v̂ = i b Ω̂: -j/(j² + k²), compact, zero at the mean
       c::R                # weight of the transform of u v: (j² - k²)/(2m+2)², compact
       d::R                # weight of that of v² - u²: j k/(2m+2)², compact
       J::UnitRange{Int}   # the rows j = 0, …, n of F̂ and ŵ
      K⁺::UnitRange{Int}   # the columns k = 0, …, n of F̂ and ŵ
      K⁻::UnitRange{Int}   # the columns k = -n, …, -1 of F̂ and ŵ
      c⁺::UnitRange{Int}   # the columns k = 0, …, n of a compact batch
      c⁻::UnitRange{Int}   # the columns k = -n, …, -1 of a compact batch
       n::Int              # largest active wavenumber
      kf::Int              # forcing wavenumber
    base::U                # tangent: the velocity f of the term of the equation; equation: nothing

    function ExplicitTerm(template::AbstractArray,
                                 n::Int,
                                 m::Int,
                                kf::Int,
                                 B::Int;
                             base::Union{Nothing, ExplicitTerm}=nothing)
        # ---- the tangent shares the scratch of the term of the equation ----
        if base !== nothing
            f = similar(base.f)
            return new{typeof(base.F̂), typeof(f), typeof(base.ifft), typeof(base.fft),
                       typeof(base.a), typeof(f)}(base.F̂,
                                                  f,
                                                  base.w,
                                                  base.ŵ,
                                                  base.ifft,
                                                  base.fft,
                                                  base.a,
                                                  base.b,
                                                  base.c,
                                                  base.d,
                                                  base.J,
                                                  base.K⁺,
                                                  base.K⁻,
                                                  base.c⁺,
                                                  base.c⁻,
                                                  n,
                                                  kf,
                                                  base.f)
        end

        # ---- wavenumbers, one at the mean to avoid 0/0 ----
        js = reshape(Float64.(0:n), :, 1)
        ks = reshape(Float64.(_kc(n)), 1, :)
        d2 = js.^2 .+ ks.^2
        d2[1, 1] = 1

        # ---- the weights of the inverse and of the forward transforms, zero at the mean ----
        a = ks ./ d2
        b = .-js ./ d2
        c = (js.^2 .- ks.^2) ./ (2m + 2)^2
        d = (js .* ks) ./ (2m + 2)^2

        # ---- work arrays, the two fields stacked along the last dimension ----
        F̂ = similar(template, ComplexF64, (m + 2, 2m + 2, B, 2))
        f = similar(template, Float64,    (2m + 2, 2m + 2, B, 2))
        w = similar(template, Float64,    (2m + 2, 2m + 2, B, 2))
        ŵ = similar(template, ComplexF64, (m + 2, 2m + 2, B, 2))

        # ---- transforms over the first two dimensions, all the fields in one call ----
        ifft = plan_brfft(F̂, 2m + 2, (1, 2))
        fft  = plan_rfft(w, (1, 2))

        # ---- assemble ----
        R = typeof(_device(template, a))
        return new{typeof(F̂), typeof(f), typeof(ifft), typeof(fft), R, Nothing}(F̂,
                                                                                f,
                                                                                w,
                                                                                ŵ,
                                                                                ifft,
                                                                                fft,
                                                                                _device(template, a),
                                                                                _device(template, b),
                                                                                _device(template, c),
                                                                                _device(template, d),
                                                                                1:(n + 1),
                                                                                1:(n + 1),
                                                                                (2m + 3 - n):(2m + 2),
                                                                                1:(n + 1),
                                                                                (n + 2):(2n + 1),
                                                                                n,
                                                                                kf,
                                                                                nothing)
    end
end

# the velocity (u, v) of the compact fields Ω, in physical space into E.f, with u = -∂ψ/∂y,
# v = ∂ψ/∂x and Δψ = ω. The transform destroys F̂, so that its zeros are written again
function _velocity!(E::ExplicitTerm, Ω::AbstractArray)
    Ω⁺ = view(Ω, :, E.c⁺, :)
    Ω⁻ = view(Ω, :, E.c⁻, :)
    E.F̂ .= 0
    view(E.F̂, E.J, E.K⁺, :, 1) .= (im .* view(E.a, :, E.c⁺)) .* Ω⁺
    view(E.F̂, E.J, E.K⁻, :, 1) .= (im .* view(E.a, :, E.c⁻)) .* Ω⁻
    view(E.F̂, E.J, E.K⁺, :, 2) .= (im .* view(E.b, :, E.c⁺)) .* Ω⁺
    view(E.F̂, E.J, E.K⁻, :, 2) .= (im .* view(E.b, :, E.c⁻)) .* Ω⁻
    mul!(E.f, E.ifft, E.F̂)

    return E
end

# dΩdt, compact, from the two products in E.w, u v and v² - u² or their derivatives: the transforms
# with the second derivatives of the identity, and the Hermitian symmetry of the row j = 0, as OKF's
# ForwardFFT!
function _forward!(E::ExplicitTerm, dΩdt::AbstractArray)
    # ---- forward transforms, differentiated, normalised ----
    mul!(E.ŵ, E.fft, E.w)
    view(dΩdt, :, E.c⁺, :) .= view(E.c, :, E.c⁺) .* view(E.ŵ, E.J, E.K⁺, :, 1) .+
                              view(E.d, :, E.c⁺) .* view(E.ŵ, E.J, E.K⁺, :, 2)
    view(dΩdt, :, E.c⁻, :) .= view(E.c, :, E.c⁻) .* view(E.ŵ, E.J, E.K⁻, :, 1) .+
                              view(E.d, :, E.c⁻) .* view(E.ŵ, E.J, E.K⁻, :, 2)

    # ---- Hermitian symmetry of the row j = 0, k ↔ -k for k = 1, …, n ----
    return _hermitian!(dΩdt, E.n)
end

# the row j = 0 of a compact batch Hermitian: its entries at k and -k conjugate
function _hermitian!(Ω::AbstractArray, n::Int)
    pos  = view(Ω, 1, 2:(n + 1), :)
    neg  = view(Ω, 1, (2n + 1):-1:(n + 2), :)
    pos .= 0.5 .* (pos .+ conj.(neg))
    neg .= conj.(pos)

    return Ω
end

# the equation: dΩdt = -(u ∂ω/∂x + v ∂ω/∂y) + forcing, overwritten, for every field of Ω
function (E::ExplicitTerm)(t::Real, Ω::AbstractArray, dΩdt::AbstractArray)
    # ---- the velocity, two inverse transforms in one call ----
    _velocity!(E, Ω)

    # ---- the two products in physical space ----
    u = view(E.f, :, :, :, 1)
    v = view(E.f, :, :, :, 2)
    view(E.w, :, :, :, 1) .= u .* v
    view(E.w, :, :, :, 2) .= v .* v .- u .* u

    # ---- back to Fourier space ----
    _forward!(E, dΩdt)

    # ---- Kolmogorov forcing at k = ±kf, j = 0 ----
    view(dΩdt, 1, E.kf + 1,        :) .-= E.kf/2
    view(dΩdt, 1, 2E.n + 2 - E.kf, :) .-= E.kf/2

    return dΩdt
end

# the tangent equation about the fields Ω: dΛdt = -(u ∂λ/∂x + v ∂λ/∂y + u′ ∂ω/∂x + v′ ∂ω/∂y),
# overwritten, for every perturbation of Λ, the call of Flows for a coupled system. The velocity of Ω
# is that of the term of the equation, called before this one
function (E::ExplicitTerm)(t::Real, Ω::AbstractArray, dΩdt::AbstractArray,
                                    Λ::AbstractArray, dΛdt::AbstractArray)
    # ---- the velocity of the perturbation, two inverse transforms in one call ----
    _velocity!(E, Λ)

    # ---- the derivatives of the two products in physical space ----
    u  = view(E.base, :, :, :, 1)
    v  = view(E.base, :, :, :, 2)
    u′ = view(E.f, :, :, :, 1)
    v′ = view(E.f, :, :, :, 2)
    view(E.w, :, :, :, 1) .= u .* v′ .+ u′ .* v
    view(E.w, :, :, :, 2) .= 2 .* (v .* v′ .- u .* u′)

    # ---- back to Fourier space, with zero mean ----
    return _forward!(E, dΛdt)
end
