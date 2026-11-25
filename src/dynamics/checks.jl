# The checks of the arguments that the objects of the dynamics share.

# a grid that dealiases the products, a forcing among the modes, a batch of at least one field, a
# device, and a scheme of Flows.jl for a split system; kf and scheme are checked when given
function _check(n::Int, m::Int, B::Int, device::Symbol; kf::Int=1, scheme::Symbol=:CB3R2R3e)
    2m + 2 >= 3n + 1 ||
        throw(ArgumentError("the grid must dealias the products, 2m + 2 at least 3n + 1"))
    1 <= kf <= n ||
        throw(ArgumentError("the forcing wavenumber must be in 1:n"))
    B >= 1 ||
        throw(ArgumentError("a batch has B ≥ 1 fields"))
    device in (:cpu, :gpu) ||
        throw(ArgumentError("device must be :cpu or :gpu"))
    haskey(Flows.imex_schemes, scheme) ||
        throw(ArgumentError("scheme must be one of Flows.imex_schemes, " *
                            "$(sort(collect(keys(Flows.imex_schemes))))"))

    return nothing
end
