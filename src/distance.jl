# The distance of two states of Kolmogorov flow up to its symmetries, for the searches of near recurrences: the
# least of ‖S(σ, sy) b - a‖/‖a‖ over the shift σ along x and the translations sy along y by whole wavelengths of
# the forcing, with S the shift of XShift, and the moduli of the coefficients, a bound of it from below that
# costs no search (notes/recurrence-search.tex of the paper on long orbits, and kf_recurrences_tight.jl, whose
# distance this is).
#
# The shift is an isometry, so that for a start a and an end b
#
#     ‖S(σ, sy) b - a‖² = ‖a‖² + ‖b‖² - 2 f(σ, sy),   f(σ, sy) = Re Σⱼ e^{i j σ} Σ_q e^{2πi q sy/kf} G_jq,
#
# with G_jq = Σ_{k ≡ q mod kf} w_j conj(a_jk) b_jk, the sum over the wavenumbers k along y of one class modulo kf,
# and w_j the weights of the inner product of the fields, 1 for j = 0 and 2 for j ≥ 1. For each sy, f is a
# trigonometric polynomial of degree n in σ: one inverse FFT of length 256 gives it at 256 shifts, and Newton's
# method refines the largest over the shifts and the translations.

"""
    Distance(n, kf; sy=(1 - kf÷2):(kf÷2))

The distance of two fields up to the symmetries of Kolmogorov flow, called as `D(a, b)`: the least over the
shift `σ` along x and the translations `sy` along y of

    ‖S(σ, sy) b - a‖ / ‖a‖,   S(σ, sy) ω(x, y) = ω(x + σ, y + 2π sy/kf),

returned as `(d, σ, sy)`. The shift is that of [`XShift`](@ref), so that `XShift(; sy=sy, kf=kf)(b, σ)` is the
state of the orbit of `b` closest to `a`, at the distance `d ‖a‖`, and the norm is that of the inner product of
the fields, [`dots`](@ref). `σ` lies in (-π, π] and `sy` is one of the translations given. For a near recurrence
from `a = u(t)` to `b = u(t + T)` this is `u(t + T)(x, y) = u(t)(x - σ, y - 2π sy/kf)` to within `d`, the
convention of the seeds of the campaigns, `σ` their `d2` and `sy` their `other_sy`.

The object holds its work arrays and its FFT plan, made with it, so that a call allocates nothing. One object
serves one task at a time.

# Arguments
- `n::Int`: the largest wavenumber of the fields.
- `kf::Int`: the wavenumber of the forcing, whose wavelength the translations take.
- `sy=(1 - kf÷2):(kf÷2)`: the translations along y compared, in wavelengths of the forcing, `-1:2` for
  `kf = 4`, as the seeds give them; `0:0` for the shift along x alone.

Called as `D(a, b)`, with
- `a::FTFields`, `b::FTFields`: one field each, on the CPU, or a view of one field of a batch.
"""
struct Distance{P}
       n::Int                   # the largest wavenumber of the fields
      kf::Int                   # the wavenumber of the forcing
      sy::Vector{Int}           # the translations along y compared
   class::Vector{Int}           # 1 + (k mod kf) for every column k of a field
   phase::Matrix{ComplexF64}    # e^{2πi q sy/kf}, q = 0, …, kf - 1 down, one translation across
       G::Matrix{ComplexF64}    # the sums G_jq, one class of k across
      Gs::Vector{ComplexF64}    # the coefficients of f at the best translation, Σ_q e^{2πi q sy/kf} G_jq
       C::Matrix{ComplexF64}    # f at the 256 shifts, one translation across
    plan::P                     # the inverse FFT of the columns of C, in place

    function Distance(n::Int, kf::Int; sy=(1 - kf÷2):(kf÷2))
        # ---- the class of every column, from the wavenumbers along y of a field, and the phases of the
        #      translations, by class ----
        class = mod.(Int.(vec(_wavenumbers_y(n, :cpu))), kf) .+ 1
        phase = [cis(2π*mod(q*s, kf)/kf) for q in 0:(kf - 1), s in sy]

        # ---- the work arrays, and the plan of the FFT along the shifts ----
        C    = zeros(ComplexF64, _SHIFTS, length(sy))
        plan = FFTW.plan_bfft!(C, 1)

        return new{typeof(plan)}(n,
                                 kf,
                                 collect(Int, sy),
                                 class,
                                 phase,
                                 zeros(ComplexF64, n + 1, kf),
                                 zeros(ComplexF64, n + 1),
                                 C,
                                 plan)
    end
end

"""
    moduli(a) -> M

The moduli of the coefficients of the field `a`, each times the square root of the weight of its row in the
inner product of the fields, an array `(n + 1, 2n + 1)` whose 2-norm is the norm of `a`. They do not change
under the shift along x and the translation along y, which change the phases alone, so that for two fields

    ‖moduli(b) - moduli(a)‖ ≤ ‖S(σ, sy) b - a‖   for every σ and sy,

and `‖moduli(b) - moduli(a)‖/‖moduli(a)‖` bounds the distance of [`Distance`](@ref) from below, with no search: a
pair whose bound is above a threshold is above it.

# Arguments
- `a::FTFields`: one field, or a view of one field of a batch.
"""
moduli(a::FTFields) = sqrt.(a.w) .* abs.(view(parent(a), :, :, 1))

function (D::Distance)(a::FTFields, b::FTFields)
    A = parent(a)
    B = parent(b)
    n = D.n

    # ---- the sums over k by class of k mod kf, and the norms, with the weights of the rows ----
    #      G_jq = Σ_{k ≡ q} w_j conj(a_jk) b_jk,   ‖a‖² = Σ w_j |a_jk|²
    fill!(D.G, 0)
    na = 0.0
    nb = 0.0
    @inbounds for k in 1:(2n + 1), j in 1:(n + 1)
        w = j == 1 ? 1.0 : 2.0
        D.G[j, D.class[k]] += w*conj(A[j, k, 1])*B[j, k, 1]
        na += w*abs2(A[j, k, 1])
        nb += w*abs2(B[j, k, 1])
    end

    # ---- f at the 256 shifts σₗ = 2π (l - 1)/256 for every translation, by one inverse FFT of its coefficients
    #      in σ, and the largest over the shifts and the translations ----
    fill!(D.C, 0)
    @inbounds for s in eachindex(D.sy), q in 1:D.kf, j in 1:(n + 1)
        D.C[j, s] += D.phase[q, s]*D.G[j, q]
    end
    D.plan * D.C
    c, at = findmax(real, D.C)
    s     = at[2]
    σ     = 2π*(at[1] - 1)/_SHIFTS

    # ---- the coefficients of f in σ at that translation ----
    fill!(D.Gs, 0)
    @inbounds for q in 1:D.kf, j in 1:(n + 1)
        D.Gs[j] += D.phase[q, s]*D.G[j, q]
    end

    # ---- Newton's method on f from the largest, four steps at most, while f is concave: with p = e^{i j σ},
    #      f′ = -Σ j Im(Gⱼ p) and f″ = -Σ j² Re(Gⱼ p), the powers of e^{iσ} built one from the other ----
    for _ in 1:4
        e  = cis(σ)
        p  = one(ComplexF64)
        f′ = 0.0
        f″ = 0.0
        for j in 1:n
            p  *= e
            g   = D.Gs[j + 1]*p
            f′ -= j*imag(g)
            f″ -= j^2*real(g)
        end
        f″ < 0 ||
            break
        σ -= f′/f″
    end

    # ---- f at the refined shift, kept if it is larger than on the grid ----
    e = cis(σ)
    p = one(ComplexF64)
    f = real(D.Gs[1])
    for j in 1:n
        p *= e
        f += real(D.Gs[j + 1]*p)
    end
    if f < c
        f = c
        σ = 2π*(at[1] - 1)/_SHIFTS
    end

    # ---- the distance, from ‖S b - a‖² = ‖a‖² + ‖b‖² - 2f ----
    return (sqrt(max(na + nb - 2f, 0.0)/na), rem(σ, 2π, RoundNearest), D.sy[s])
end

# the shifts along x at which f is first evaluated, by one FFT, before Newton's method
const _SHIFTS = 256
