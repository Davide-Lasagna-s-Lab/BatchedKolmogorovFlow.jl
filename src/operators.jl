# The derivatives of a batch of fields, with the names of OpenKolmogorovFlow. Each is diagonal in
# Fourier space, a product of every coefficient with the wavenumbers of its mode, j along x and
# k along y, which the batch holds, and acts on all the fields of the batch at once. `out` may be
# `X` itself.

# one over a squared wavenumber, zero at the mean, where it would be 1/0
_over(q) = iszero(q) ? 0.0 : 1/q

"""
    ddx!(out, X) -> out

The derivative along x of the fields of the batch `X`, their coefficients times `im*j`.
"""
function ddx!(out::FTFields, X::FTFields)
    # ---- ∂/∂x of e^{i (j x + k y)} is i j times it ----
    out .= im .* X.j .* X

    return out
end

"""
    ddy!(out, X) -> out

The derivative along y of the fields of the batch `X`, their coefficients times `im*k`.
"""
function ddy!(out::FTFields, X::FTFields)
    # ---- ∂/∂y of e^{i (j x + k y)} is i k times it ----
    out .= im .* X.k .* X

    return out
end

"""
    laplacian!(out, X) -> out

The Laplacian of the fields of the batch `X`, their coefficients times `-(j² + k²)`.
"""
function laplacian!(out::FTFields, X::FTFields)
    # ---- Δ of e^{i (j x + k y)} is -(j² + k²) times it ----
    out .= .-(X.j.^2 .+ X.k.^2) .* X

    return out
end

"""
    invlaplacian!(out, X) -> out
    invlaplacian!(out, X, c) -> out

The inverse of an elliptic operator on the fields of the batch `X`. Without `c`, `Δ⁻¹ X`, the
coefficients over `-(j² + k²)`, with zero mean. With `c`, `(I - cΔ)⁻¹ X`, the solve of an
implicit stage of the time stepping, as `Flows.ImcA!` asks for it.
"""
function invlaplacian!(out::FTFields, X::FTFields)
    # ---- the coefficients over -(j² + k²), and zero at the mean, which Δ⁻¹ leaves free ----
    out .= .-X .* _over.(X.j.^2 .+ X.k.^2)

    return out
end

function invlaplacian!(out::FTFields, X::FTFields, c::Real)
    # ---- I - cΔ is 1 + c (j² + k²) on every mode ----
    out .= X ./ (1 .+ c .* (X.j.^2 .+ X.k.^2))

    return out
end
