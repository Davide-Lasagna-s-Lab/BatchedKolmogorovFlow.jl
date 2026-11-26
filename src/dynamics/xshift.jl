# The shift along x, the continuous symmetry of Kolmogorov flow, for relative periodic orbits: the
# shift S of MultipleShooting.jl, and its generator dS, both one object. An orbit may also close
# after a translation along y by whole wavelengths 2π/kf of the forcing, a discrete symmetry of
# the flow, fixed for the orbit. The shift then translates too, so that every part of a search
# that shifts the end of the last segment translates it, with no change of its own.

"""
    XShift(; sy=0, kf=0)
    XShift(S, Re)

The shift of the fields along x, called as `S(X, s)`: every field of the batch `X` shifted in
place by the number `s`, `ω(x, y) → ω(x + s, y)`, the coefficients times `cis(j s)`, and `X`
returned. With `sy ≠ 0` the fields are also translated along y by `sy` wavelengths of the
forcing, `ω(x, y) → ω(x + s, y + 2π sy/kf)`, the coefficients times `cis(j s + 2π k sy/kf)`, as
`shift!(Ω, s, sy)` of OpenKolmogorovFlow for `kf = 4`: the shift of an orbit that closes after
this translation. Called as `S(dX, X, j)` it is the generator of the shift, `∂ω/∂x` at every field
of `X`, written into `dX`, with `j`, the index of the shift, 1, the only one: the translation is
discrete, with no generator and no unknown. Any number of fields, with no workspace.

The second form is `S` itself, which does not depend on the Reynolds number, for the families of
the searches with a parameter, as the other objects of the dynamics.

# Arguments
- `sy::Int=0`: the translation along y, in wavelengths of the forcing. With `sy = 0` the shift is
  along x alone, as with no keyword.
- `kf::Int=0`: the wavenumber of the forcing, that of the propagators, needed when `sy ≠ 0`.
"""
struct XShift
    sy::Int    # the translation along y, in wavelengths 2π/kf of the forcing
    kf::Int    # the wavenumber of the forcing, 0 when sy = 0 and none is given

    function XShift(; sy::Int=0, kf::Int=0)
        # ---- input checks ----
        sy == 0 || kf >= 1 ||
            throw(ArgumentError("a translation along y, sy ≠ 0, needs the forcing wavenumber kf"))

        return new(sy,
                   kf)
    end
end

XShift(S::XShift, Re::Real) = S

function (S::XShift)(X::FTFields, s::Real)
    # ---- e^{i j (x + s)} is e^{i j s} times e^{i j x} ----
    X .*= cis.(X.j .* s)

    # ---- and e^{i k (y + 2π sy/kf)} is e^{i 2π k sy/kf} times e^{i k y}; nothing when sy = 0 ----
    S.sy == 0 ||
        (X .*= cis.(X.k .* (2π*S.sy/S.kf)))

    return X
end

function (S::XShift)(dX::FTFields, X::FTFields, j::Int)
    # ---- input checks ----
    j == 1 ||
        throw(ArgumentError("the shift along x is the one shift, j = 1"))

    return ddx!(dX, X)
end
