# The shift along x, the continuous symmetry of Kolmogorov flow, for relative periodic orbits: the
# shift S of MultipleShooting.jl, and its generator dS, both one object.

"""
    XShift()
    XShift(S, Re)

The shift of the fields along x, called as `S(X, s)`: every field of the batch `X` shifted in
place by the number `s`, `ω(x, y) → ω(x + s, y)`, the coefficients times `cis(j s)`, and `X`
returned. Called as `S(dX, X, j)` it is the generator of the shift, `∂ω/∂x` at every field of
`X`, written into `dX`, with `j`, the index of the shift, 1, the only one. Any number of fields,
with no workspace.

The second form is `S` itself, which does not depend on the Reynolds number, for the families of
the searches with a parameter, as the other objects of the dynamics.
"""
struct XShift end

XShift(S::XShift, Re::Real) = S

function (S::XShift)(X::FTFields, s::Real)
    # ---- e^{i j (x + s)} is e^{i j s} times e^{i j x} ----
    X .*= cis.(X.j .* s)

    return X
end

function (S::XShift)(dX::FTFields, X::FTFields, j::Int)
    # ---- input checks ----
    j == 1 ||
        throw(ArgumentError("the shift along x is the one shift, j = 1"))

    return ddx!(dX, X)
end
