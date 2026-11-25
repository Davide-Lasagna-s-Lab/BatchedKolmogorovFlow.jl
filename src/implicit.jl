# The viscous term of the vorticity equation, Δω/Re, the linear part of the time stepping, in the
# call interface of Flows.jl: implicit for the implicit–explicit schemes, as the ImplicitTerm of
# OpenKolmogorovFlow, and exact for the exponential ones, by the coefficients of its diagonal.

"""
    ImplicitTerm(X, Re)
    ImplicitTerm(L, Re)

The viscous term `Δω/Re` of the vorticity equation on batches of fields like `X`: `mul!(out, L, X)`
applies it, `Flows.ImcA!(L, c, Y, Z)` solves `(I - c Δ/Re) Z = Y` for the implicit–explicit
schemes of Flows.jl, and `Flows.etdrk2_coeffs(L, Δt)` and `Flows.etdrk4_coeffs(L, Δt)` give the
coefficients of the exponential ones. The term is diagonal in Fourier space, `-(j² + k²)/Re` on
every mode, and holds that diagonal and the arrays of the coefficients, made once. The second
form is `L` at another Reynolds number, which shares them and costs nothing.

# Arguments
- `X::FTFields`: a batch, for the wavenumbers and the device of the diagonal.
- `Re::Real`: the Reynolds number.
"""
struct ImplicitTerm{D<:AbstractMatrix{Float64}}
               Δ::D             # the Laplacian on every mode, -(j² + k²), (n+1, 2n+1) on the device
              Re::Float64       # the Reynolds number
    coefficients::NTuple{6, D}  # the coefficients of the exponential schemes, mode by mode, like Δ

    function ImplicitTerm(X::FTFields, Re::Real)
        Δ = .-(X.j.^2 .+ X.k.^2)

        return new{typeof(Δ)}(Δ,
                              Float64(Re),
                              ntuple(_ -> similar(Δ), 6))
    end

    ImplicitTerm(L::ImplicitTerm{D}, Re::Real) where {D} =
        new{D}(L.Δ,
               Float64(Re),
               L.coefficients)
end

# out = Δ X/Re, in one broadcast
function LinearAlgebra.mul!(out::FTFields, L::ImplicitTerm, X::FTFields)
    out .= L.Δ .* X ./ L.Re

    return out
end

# Z = (I - c Δ/Re)⁻¹ Y, the solve of an implicit stage, in one broadcast
function Flows.ImcA!(L::ImplicitTerm, c::Real, Y::FTFields, Z::FTFields)
    Z .= Y ./ (1 .- (c/L.Re) .* L.Δ)

    return Z
end


# ---------------------------------------------------------------------------- #
# the coefficients of the exponential schemes                                  #
# ---------------------------------------------------------------------------- #

# The exponential schemes of Flows.jl integrate the linear part exactly, through the coefficients
# exp(L) and the φ functions of L = Δt Δ/Re, here a diagonal, mode by mode. They are those of
# Flows.jl for a Diagonal, written into the arrays of the term, (n+1, 2n+1) on the device, made
# once: evaluated once per mode, and not per field, since Flows.jl then writes each into a batch
# of its registers, `c .= v`, which broadcasts it against a batch of any number of fields or a
# bundle of fields with their perturbations. The coefficients change with the step, at the
# shorter last step of a call, and this makes no array.

function Flows.etdrk2_coeffs(A::ImplicitTerm, Δt::Real)
    E, f1, f2 = A.coefficients
    c         = Δt/A.Re

    # ---- exp(L), Δt φ₁(L) and Δt φ₂(L), L = c Δ ----
    E  .= exp.(c .* A.Δ)
    f1 .= Δt .* _contour.(z -> (exp(z) - 1) / z,       c .* A.Δ)
    f2 .= Δt .* _contour.(z -> (exp(z) - 1 - z) / z^2, c .* A.Δ)

    return (E, f1, f2)
end

function Flows.etdrk4_coeffs(A::ImplicitTerm, Δt::Real)
    E, E2, Q, f1, f2, f3 = A.coefficients
    c                    = Δt/A.Re

    # ---- the coefficients of Cox & Matthews (2002), L = c Δ ----
    E  .= exp.(c .* A.Δ)
    E2 .= exp.(c .* A.Δ ./ 2)
    Q  .= Δt .* _contour.(z -> (exp(z/2) - 1) / z,                     c .* A.Δ)
    f1 .= Δt .* _contour.(z -> (-4 - z + exp(z)*(4 - 3z + z^2)) / z^3, c .* A.Δ)
    f2 .= Δt .* _contour.(z -> ( 2 + z + exp(z)*(-2 + z)) / z^3,       c .* A.Δ)
    f3 .= Δt .* _contour.(z -> (-4 - 3z - z^2 + exp(z)*(4 - z)) / z^3, c .* A.Δ)

    return (E, E2, Q, f1, f2, f3)
end

# the mean of the function f on the circle of radius 1 about the real number l, by 32 points on
# its upper half, as Flows.jl takes it (Kassam & Trefethen, 2005): it removes the cancellation of
# the φ functions near l = 0. A loop on one number, so that it broadcasts on the device
function _contour(f, l::Real)
    s = 0.0
    for p in 1:32
        s += real(f(l + cis(π*(p - 0.5)/32)))
    end

    return s/32
end
