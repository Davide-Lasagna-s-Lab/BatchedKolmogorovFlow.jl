# The flows of Flows.jl of the propagators, each made once with its propagator, and made again at
# another Reynolds number, with the same terms and registers.

# the flow ϕ at the Reynolds number Re, with its terms and the registers of its scheme, shared: the
# implicit term is that of Re, and the explicit term too when it depends on Re, as the forced one
_reynolds(ϕ::Flows.Flow, Re::Real) =
    Flows.flow(_reynolds(ϕ.sys.g, Re),
               ImplicitTerm(Re),
               ϕ.meth,
               ϕ.tstep)

_reynolds(term, Re::Real)                    = term
_reynolds(term::ForcedTangentTerm, Re::Real) = ForcedTangentTerm(term.tangent, Re)
