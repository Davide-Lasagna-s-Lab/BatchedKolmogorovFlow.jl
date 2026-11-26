# Examples of use of BatchedKolmogorovFlow.jl, those of the README, each with a check: a batch,
# the flow, the tangent of one perturbation and of several, the tangent forced by the Reynolds
# number, the right-hand side and the shift, the adjoint, and the same on a GPU.
#
#     julia --project=examples examples/usage.jl
#     julia --project=<an environment with CUDA> examples/usage.jl gpu
using BatchedKolmogorovFlow
using LinearAlgebra
using Printf
using Random

const DEVICE = Symbol(get(ARGS, 1, "cpu"))
if DEVICE === :gpu
    using CUDA
end

# the largest wavenumber, the grid of 2m + 2 = 64 points, the Reynolds number, the wavenumber of
# the forcing, and the fields of a batch
n, m, Re, kf = 21, 31, 40.0, 4
B            = 3

# the largest difference of two batches, relative to the second
reldiff(X, Y) = maximum(abs, Array(parent(X)) .- Array(parent(Y)))/maximum(abs, Array(parent(Y)))


# ---------------------------------------------------------------------------- #
# a batch                                                                      #
# ---------------------------------------------------------------------------- #

# B smooth random fields on the device: random coefficients that fall with the wavenumber, X.j and
# X.k, made fields by the symmetry of the row j = 0 and a zero mean
function fields(B; seed=1)
    Random.seed!(seed)
    X  = randn!(FTFields(n, B))
    X .= X ./ (1 .+ X.j.^2 .+ X.k.^2)
    BatchedKolmogorovFlow.apply_symmetry!(X)
    Y  = FTFields(n, B; device=DEVICE)
    copyto!(parent(Y), parent(X))

    return Y
end

X = fields(B)
V = fields(B; seed=2)
W = fields(B; seed=3)

println("a batch of ", size(X, 3), " fields, an array ", size(X), " on the ", DEVICE)
println("the inner products of the fields with themselves ", round.(vec(Array(dots(X, X))), digits=4),
        ", and their sum ", round(dot(X, X), digits=4))

# the batch X + a V
plus(X, a, V) = FTFields(parent(X) .+ a .* parent(V))


# ---------------------------------------------------------------------------- #
# the flow                                                                     #
# ---------------------------------------------------------------------------- #

# the flow of batches of B fields: all advanced together, in place
G = NonlinearPropagator(n, m, Re, kf; B=B, device=DEVICE)
Y = G(copy(X), 1.0)

@printf("the flow: G(X, 1) against G(G(X, 0.5), 0.5), %.1e\n", reldiff(G(G(copy(X), 0.5), 0.5), Y))


# ---------------------------------------------------------------------------- #
# the tangent                                                                  #
# ---------------------------------------------------------------------------- #

# one perturbation of every field: X and V advanced together, V by the tangent equation along X
L  = TangentPropagator(n, m, Re, kf; B=B, device=DEVICE)
LV = L(copy(X), copy(V), 1.0)
D  = plus(G(plus(X, 1e-6, V), 1.0), -1.0, G(plus(X, -1e-6, V), 1.0))
D .= D ./ 2e-6

@printf("the tangent: L(X, V, 1) against a centred difference of the flow, %.1e\n", reldiff(LV, D))

# three perturbations of every field, a vector of three batches: the fields are advanced once
L₃ = TangentPropagator(n, m, Re, kf; B=B, k′=3, device=DEVICE)
Vs = [fields(B; seed=10 + a) for a in 1:3]
V₂ = L(copy(X), copy(Vs[2]), 1.0)
L₃(copy(X), Vs, 1.0)

@printf("three perturbations at once: the second against the tangent of it alone, %.1e\n",
        reldiff(Vs[2], V₂))


# ---------------------------------------------------------------------------- #
# the Reynolds number                                                          #
# ---------------------------------------------------------------------------- #

# the tangent forced by the derivative of the equation in the Reynolds number: from V = 0, the
# derivative of the flow in it. The flow at another Reynolds number shares the arrays of G
Lp = TangentPropagator(n, m, Re, kf; B=B, forced=true, device=DEVICE)
dR = Lp(copy(X), fill!(similar(X), 0), 1.0)
G⁺ = NonlinearPropagator(G, Re + 1e-3)
G⁻ = NonlinearPropagator(G, Re - 1e-3)
D  = plus(G⁺(copy(X), 1.0), -1.0, G⁻(copy(X), 1.0))
D .= D ./ 2e-3

@printf("the forced tangent: against a centred difference in the Reynolds number, %.1e\n",
        reldiff(dR, D))


# ---------------------------------------------------------------------------- #
# the right-hand side and the shift                                            #
# ---------------------------------------------------------------------------- #

# the time derivative of every field, and the shift along x with its generator, ∂ω/∂x
F  = RightHandSide(n, m, Re, kf; B=B, device=DEVICE)
S  = XShift()
dX = F(similar(X), X)
gX = S(similar(X), X, 1)

D  = plus(G(copy(X), 1e-4), -1.0, X)
D .= D ./ 1e-4
@printf("the right-hand side: F(dX, X) against (G(X, h) - X)/h, h = 1e-4, %.1e\n", reldiff(dX, D))

D  = plus(S(copy(X), 1e-5), -1.0, S(copy(X), -1e-5))
D .= D ./ 2e-5
@printf("the generator of the shift: S(dX, X, 1) against a centred difference of S, %.1e\n",
        reldiff(gX, D))
@printf("the shift and the flow: G(S(X, 0.7), 1) against S(G(X, 1), 0.7), %.1e\n",
        reldiff(G(S(copy(X), 0.7), 1.0), S(copy(Y), 0.7)))


# ---------------------------------------------------------------------------- #
# the adjoint                                                                  #
# ---------------------------------------------------------------------------- #

# the transpose of the tangent, back over the stages of the flow, kept once for any W
A  = AdjointPropagator(n, m, Re; B=B, device=DEVICE)
St = stages(G, X, 1.0)
AW = A(copy(W), St)

println("the adjoint: <L V, W> and <V, A W>, field by field,")
println("    ", vec(Array(dots(LV, W))))
println("    ", vec(Array(dots(V, AW))))
