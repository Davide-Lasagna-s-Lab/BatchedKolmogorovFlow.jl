# The viscous term of the vorticity equation, Δω/Re, the implicit term of the time stepping, in
# the call interface of Flows.jl, as the ImplicitTerm of OpenKolmogorovFlow.

"""
    ImplicitTerm(Re)

The viscous term `Δω/Re` of the vorticity equation on batches of fields: `mul!(out, L, X)`
applies it and `Flows.ImcA!(L, c, Y, Z)` solves `(I - c Δ/Re) Z = Y`. It holds the Reynolds
number alone, so that one at another Reynolds number costs nothing.

# Arguments
- `Re::Real`: the Reynolds number.
"""
struct ImplicitTerm
    Re::Float64 # the Reynolds number
end

function LinearAlgebra.mul!(out::FTFields, L::ImplicitTerm, X::FTFields)
    # ---- the Laplacian, over the Reynolds number ----
    laplacian!(out, X)
    out .*= 1/L.Re

    return out
end

# Z = (I - c Δ/Re)⁻¹ Y, the solve of an implicit stage
Flows.ImcA!(L::ImplicitTerm, c::Real, Y::FTFields, Z::FTFields) = invlaplacian!(Z, Y, c/L.Re)
