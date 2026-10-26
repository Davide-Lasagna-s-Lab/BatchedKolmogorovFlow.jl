# The explicit and implicit terms of the vorticity equation, and the explicit
# term of its tangent equation, for a batch of fields, in the call interface of
# Flows.jl. Field by field, they repeat the operations of ForwardEquation and of
# LinearisedEquation(TangentMode()) of OpenKolmogorovFlow: the same masks, the
# same order of the arithmetic, the same normalisation of the transforms. Only
# the transforms differ, done in batch over the first two dimensions.
#
# The real-to-complex transforms of cuFFT halve the first dimension of the
# region, while the fields of OKF halve the second (x). The transforms work on
# transposed copies, physical space as (x, y, batch) and Fourier space as
# (j, k, batch), written and read through a PermutedDimsArray in the same
# broadcasts that apply the derivatives, the mask and the normalisation.

# a host array copied to the device of `template`
_device(template::AbstractArray, h::AbstractArray) =
    copyto!(similar(template, eltype(h), size(h)), h)

# signed wavenumbers k down the rows (FFT order) and j ≥ 0 across the columns
_ks(m::Int) = [r <= m ? r : r - (2m + 2) for r in 0:(2m + 1)]
_js(m::Int) = collect(0:(m + 1))

# the active modes of OKF, |k| ≤ n and j ≤ n, without the mean
_mask(n::Int, m::Int) = [Float64(abs(k) <= n && j <= n && !(k == 0 && j == 0))
                         for k in _ks(m), j in _js(m)]


# ---------------------------------------------------------------------------- #
# The implicit term, Δ/Re, diagonal in Fourier space
# ---------------------------------------------------------------------------- #
struct ImplicitTerm{R<:AbstractMatrix{Float64}}
      Re::Float64 # Reynolds number
     lap::R       # -(j² + k²) on the active modes, zero elsewhere, (2m+2, m+2)
      k2::R       # j² + k², (2m+2, m+2)
    mask::R       # one on the active modes, zero elsewhere and on the mean

    function ImplicitTerm(template::AbstractArray, n::Int, m::Int, Re::Real)
        # ---- wavenumbers in the layout of the fields ----
        k2   = [Float64(j^2 + k^2) for k in _ks(m), j in _js(m)]
        mask = _mask(n, m)
        lap  = .-k2 .* mask

        # ---- on the device ----
        R = typeof(_device(template, k2))
        return new{R}(Float64(Re),
                      _device(template, lap),
                      _device(template, k2),
                      _device(template, mask))
    end
end

# the Laplacian over Re, as `mul!` of OKF's ImplicitTerm
mul!(out::AbstractArray, L::ImplicitTerm, X::AbstractArray) =
    (out .= X .* L.lap .* (1/L.Re); out)

# Z = (I - c Δ/Re)⁻¹ Y, as `Flows.ImcA!` of OKF's ImplicitTerm
Flows.ImcA!(L::ImplicitTerm, c::Real, Y::AbstractArray, Z::AbstractArray) =
    (Z .= Y ./ (1 .+ (c/L.Re) .* L.k2) .* L.mask; Z)


# ---------------------------------------------------------------------------- #
# The explicit terms for a batch of B fields: advection and forcing of the
# equation, with 4 fields in physical space (∂ω/∂x, ∂ω/∂y, u, v), or the
# linearised advection of the tangent equation, with 8 (the same for the
# perturbation λ, then u′ and v′)
# ---------------------------------------------------------------------------- #
struct ExplicitTerm{C4, R4, R3, C3, PB, PF, R}
        F̂::C4  # fields to transform in Fourier space, transposed, (m+2, 2m+2, B, 4 or 8)
        f::R4  # the same in physical space, transposed, (2m+2, 2m+2, B, 4 or 8)
        w::R3  # nonlinear term in physical space, transposed, (2m+2, 2m+2, B)
        ŵ::C3  # its transform, (m+2, 2m+2, B)
     ifft::PB  # inverse transform of F̂ into f, unnormalised as OKF
      fft::PF  # forward transform of w into ŵ, unnormalised
       jt::R   # j as a column, (m+2, 1)
       kt::R   # signed k as a row, (1, 2m+2)
      d2t::R   # j² + k², one at the mean, (m+2, 2m+2)
    maskt::R   # active modes, transposed, (m+2, 2m+2)
     mask::R   # active modes in the layout of the fields, (2m+2, m+2)
        m::Int # storage cutoff
       kf::Int # forcing wavenumber

    function ExplicitTerm(template::AbstractArray,
                                 n::Int,
                                 m::Int,
                                kf::Int,
                                 B::Int,
                              mode::Symbol)
        # ---- wavenumbers, transposed ----
        jt    = reshape(Float64.(_js(m)), :, 1)
        kt    = reshape(Float64.(_ks(m)), 1, :)
        d2t   = jt.^2 .+ kt.^2
        maskt = permutedims(_mask(n, m))

        # avoid 0/0 at the mean, which the mask then sets to zero
        d2t[1, 1] = 1

        # ---- work arrays, the fields to transform stacked along the last dimension ----
        nf = mode === :tangent ? 8 : 4
        F̂  = similar(template, ComplexF64, (m + 2, 2m + 2, B, nf))
        f  = similar(template, Float64,    (2m + 2, 2m + 2, B, nf))
        w  = similar(template, Float64,    (2m + 2, 2m + 2, B))
        ŵ  = similar(template, ComplexF64, (m + 2, 2m + 2, B))

        # ---- transforms over the first two dimensions, all the fields in one call ----
        ifft = plan_brfft(F̂, 2m + 2, (1, 2))
        fft  = plan_rfft(w, (1, 2))

        # ---- assemble ----
        R = typeof(_device(template, d2t))
        return new{typeof(F̂), typeof(f), typeof(w), typeof(ŵ),
                   typeof(ifft), typeof(fft), R}(F̂,
                                                 f,
                                                 w,
                                                 ŵ,
                                                 ifft,
                                                 fft,
                                                 _device(template, jt),
                                                 _device(template, kt),
                                                 _device(template, d2t),
                                                 _device(template, maskt),
                                                 _device(template, _mask(n, m)),
                                                 m,
                                                 kf)
    end
end

# ∂ω/∂x, ∂ω/∂y, u and v of the fields Ω into the slots a, …, a+3 of F̂, masked as
# OKF before its inverse transform: u = -∂ψ/∂y and v = ∂ψ/∂x, with Δψ = ω
function _derivatives!(E::ExplicitTerm, Ω::AbstractArray, a::Int)
    Ωt  = PermutedDimsArray(Ω, (2, 1, 3))     # Ωt[j, k, b] = Ω[k, j, b]
    view(E.F̂, :, :, :, a)     .= (im .* E.jt) .* Ωt .* E.maskt
    view(E.F̂, :, :, :, a + 1) .= (im .* E.kt) .* Ωt .* E.maskt
    view(E.F̂, :, :, :, a + 2) .= ((im .* E.kt) .* Ωt) ./ E.d2t .* E.maskt
    view(E.F̂, :, :, :, a + 3) .= .-(((im .* E.jt) .* Ωt) ./ E.d2t) .* E.maskt
    return E
end

# the transform of the product w back to the layout of the fields, normalised,
# masked and with the Hermitian symmetry of the column j = 0, as OKF's ForwardFFT!
function _forward!(E::ExplicitTerm, dΩdt::AbstractArray)
    # ---- forward transform, normalised and masked ----
    m = E.m
    mul!(E.ŵ, E.fft, E.w)
    dΩdt .= PermutedDimsArray(E.ŵ, (2, 1, 3)) .* (1/(2m + 2)^2) .* E.mask

    # ---- Hermitian symmetry of the j = 0 column, k ↔ -k for k = 1, …, m ----
    pos  = view(dΩdt, 2:(m + 1), 1, :)
    neg  = view(dΩdt, (2m + 2):-1:(m + 3), 1, :)
    pos .= 0.5 .* (pos .+ conj.(neg))
    neg .= conj.(pos)

    return dΩdt
end

# the equation: dΩdt = -(u ∂ω/∂x + v ∂ω/∂y) + forcing, overwritten, for every field of Ω
function (E::ExplicitTerm)(t::Real, Ω::AbstractArray, dΩdt::AbstractArray)
    # ---- derivatives and velocity, the four inverse transforms in one call ----
    _derivatives!(E, Ω, 1)
    mul!(E.f, E.ifft, E.F̂)

    # ---- the product in physical space, -(u ∂ω/∂x + v ∂ω/∂y) ----
    ωx   = view(E.f, :, :, :, 1)
    ωy   = view(E.f, :, :, :, 2)
    u    = view(E.f, :, :, :, 3)
    v    = view(E.f, :, :, :, 4)
    E.w .= .- u.*ωx .- v.*ωy

    # ---- back to Fourier space ----
    _forward!(E, dΩdt)

    # ---- Kolmogorov forcing at k = ±kf, j = 0 ----
    view(dΩdt, E.kf + 1,        1, :) .-= E.kf/2
    view(dΩdt, 2E.m + 3 - E.kf, 1, :) .-= E.kf/2

    return dΩdt
end

# the tangent equation about the fields Ω: dΛdt = -(u ∂λ/∂x + v ∂λ/∂y + u′ ∂ω/∂x +
# v′ ∂ω/∂y), overwritten, for every perturbation of Λ, the call of Flows for a
# coupled system
function (E::ExplicitTerm)(t::Real, Ω::AbstractArray, dΩdt::AbstractArray,
                                    Λ::AbstractArray, dΛdt::AbstractArray)
    # ---- derivatives and velocities of the base and the perturbation, eight inverse transforms in one call ----
    _derivatives!(E, Ω, 1)
    _derivatives!(E, Λ, 5)
    mul!(E.f, E.ifft, E.F̂)

    # ---- the products in physical space ----
    dωdx = view(E.f, :, :, :, 1)
    dωdy = view(E.f, :, :, :, 2)
    u    = view(E.f, :, :, :, 3)
    v    = view(E.f, :, :, :, 4)
    dλdx = view(E.f, :, :, :, 5)
    dλdy = view(E.f, :, :, :, 6)
    u′   = view(E.f, :, :, :, 7)
    v′   = view(E.f, :, :, :, 8)
    E.w .= .-u.*dλdx .- v.*dλdy .- u′.*dωdx .- v′.*dωdy

    # ---- back to Fourier space, with zero mean ----
    return _forward!(E, dΛdt)
end
