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
#
# The operations are kernels, not broadcasts, because the time of a batch of a few fields is the time
# to launch them, and of a large batch the memory they read and write. One kernel writes the two
# entries of each velocity and of each pair of products, and `step!` in propagator.jl goes further.

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


# ---------------------------------------------------------------------------- #
# The explicit terms for a batch of B fields, advection and forcing of the
# equation, or the linearised advection of the tangent equation
# ---------------------------------------------------------------------------- #
struct ExplicitTerm{C4, R4, PB, PF, R, U}
       F̂::C4  # velocity to transform, (m+2, 2m+2, B, 2)
       f::R4  # the same in physical space, (2m+2, 2m+2, B, 2)
       w::R4  # the two products in physical space, (2m+2, 2m+2, B, 2)
       ŵ::C4  # their transforms, (m+2, 2m+2, B, 2)
    ifft::PB  # inverse transform of F̂ into f, unnormalised as OKF
     fft::PF  # forward transform of w into ŵ, unnormalised
       a::R   # û = i a Ω̂: k/(j² + k²), compact, zero at the mean
       b::R   # v̂ = i b Ω̂: -j/(j² + k²), compact, zero at the mean
       c::R   # weight of the transform of u v: (j² - k²)/(2m+2)², compact
       d::R   # weight of that of v² - u²: j k/(2m+2)², compact
       n::Int # largest active wavenumber
     gap::Int # from the column of k < 0 of a compact batch to its column in F̂ and ŵ
      kf::Int # forcing wavenumber
    base::U   # tangent: the velocity f of the term of the equation; equation: nothing

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
                                                  n,
                                                  base.gap,
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
        F̂    = _input(ifft, F̂)

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
                                                                                n,
                                                                                2m + 1 - 2n,
                                                                                kf,
                                                                                nothing)
    end
end

# the array that the inverse transform reads, its zeros, and the transform; methods for the GPU in
# an extension
_input(plan, F̂)        = F̂
_zero!(F̂)              = fill!(F̂, 0)
_inverse!(f, plan, F̂)  = mul!(f, plan, F̂)

# the column of F̂ and ŵ of the column k of a compact batch, and the compact column of -k
@inline _column(k::Int, n::Int, gap::Int) = k <= n + 1 ? k : k + gap
@inline _minus(k::Int, n::Int)            = k == 1 ? 1 : 2n + 3 - k

# the derivatives of the identity at row j, column k of field i, from the transforms ŵ of the two
# products, normalised
@inline function _derivatives(ŵ, c, d, j, k, i, n, gap)
    K = _column(k, n, gap)
    return @inbounds c[j, k]*ŵ[j, K, i, 1] + d[j, k]*ŵ[j, K, i, 2]
end

# the explicit term there, with the Hermitian symmetry of the row j = 0, as OKF's ForwardFFT!, and
# the force f at k = ±kf, j = 0
@inline function _nonlinear(ŵ, c, d, j, k, i, n, gap, kf, f)
    N = _derivatives(ŵ, c, d, j, k, i, n, gap)
    if j == 1
        N = 0.5*(N + conj(_derivatives(ŵ, c, d, j, _minus(k, n), i, n, gap)))
        (k == kf + 1 || k == 2n + 2 - kf) && (N -= f)
    end

    return N
end

# the velocity of the fields Ω into F̂, one write of each entry of û and v̂ per read of Ω̂
@kernel function _velocity_kernel!(F̂, Ω, a, b, n, gap)
    j, k, i = @index(Global, NTuple)
    K = _column(k, n, gap)
    @inbounds begin
        F̂[j, K, i, 1] = im*a[j, k]*Ω[j, k, i]
        F̂[j, K, i, 2] = im*b[j, k]*Ω[j, k, i]
    end
end

# the products u v and v² - u² at a point of the grid
@kernel function _products_kernel!(w, f)
    p, q, i = @index(Global, NTuple)
    @inbounds begin
        u = f[p, q, i, 1]
        v = f[p, q, i, 2]
        w[p, q, i, 1] = u*v
        w[p, q, i, 2] = v*v - u*u
    end
end

# their derivatives along the velocity (u′, v′) in f, about the velocity (u, v) in g
@kernel function _tangent_products_kernel!(w, f, g)
    p, q, i = @index(Global, NTuple)
    @inbounds begin
        u  = g[p, q, i, 1]
        v  = g[p, q, i, 2]
        u′ = f[p, q, i, 1]
        v′ = f[p, q, i, 2]
        w[p, q, i, 1] = u*v′ + u′*v
        w[p, q, i, 2] = 2*(v*v′ - u*u′)
    end
end

# the explicit term into the compact batch D
@kernel function _nonlinear_kernel!(D, ŵ, c, d, n, gap, kf, f)
    j, k, i = @index(Global, NTuple)
    @inbounds D[j, k, i] = _nonlinear(ŵ, c, d, j, k, i, n, gap, kf, f)
end

# the velocity (u, v) of the compact fields Ω, in physical space into E.f, with u = -∂ψ/∂y,
# v = ∂ψ/∂x and Δψ = ω. The transform destroys F̂, so that its zeros are written again
function _velocity!(E::ExplicitTerm, Ω::AbstractArray)
    _zero!(E.F̂)
    _velocity_kernel!(get_backend(Ω))(E.F̂, Ω, E.a, E.b, E.n, E.gap; ndrange=size(Ω))
    _inverse!(E.f, E.ifft, E.F̂)

    return E
end

# the two products from the velocity in E.f, and their transforms
function _products!(E::ExplicitTerm)
    _products_kernel!(get_backend(E.f))(E.w, E.f; ndrange=size(E.f)[1:3])
    mul!(E.ŵ, E.fft, E.w)

    return E
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
    _velocity!(E, Ω)
    _products!(E)
    _nonlinear_kernel!(get_backend(Ω))(dΩdt, E.ŵ, E.c, E.d, E.n, E.gap, E.kf, E.kf/2; ndrange=size(Ω))

    return dΩdt
end

# the tangent equation about the fields Ω: dΛdt = -(u ∂λ/∂x + v ∂λ/∂y + u′ ∂ω/∂x + v′ ∂ω/∂y),
# overwritten, for every perturbation of Λ, the call of Flows for a coupled system. The velocity of Ω
# is that of the term of the equation, called before this one
function (E::ExplicitTerm)(t::Real, Ω::AbstractArray, dΩdt::AbstractArray,
                                    Λ::AbstractArray, dΛdt::AbstractArray)
    _velocity!(E, Λ)
    _tangent_products_kernel!(get_backend(Λ))(E.w, E.f, E.base; ndrange=size(E.f)[1:3])
    mul!(E.ŵ, E.fft, E.w)
    _nonlinear_kernel!(get_backend(Λ))(dΛdt, E.ŵ, E.c, E.d, E.n, E.gap, E.kf, 0.0; ndrange=size(Λ))

    return dΛdt
end
