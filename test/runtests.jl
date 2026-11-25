# The tests of the package, by topic, on the CPU or on another array type:
#
#     julia --project=test test/runtests.jl
#     DEVICE=gpu julia --project=test test/runtests.jl
#
# With DEVICE=gpu the batches are CuArrays. CUDA is not a dependency of the package nor of this
# environment: it is loaded from another environment of the load path, the global one by default.
#
# Every function is compared with OpenKolmogorovFlow (OKF) and Flows.jl on a small grid, field by
# field, and checked against finite differences and the properties that it must have. The fields
# are made on the CPU, from fixed seeds, and moved to the device; the results are compared on the
# CPU.

using Test
using LinearAlgebra
using Random
using FFTW
using Flows
using MultipleShooting
using BatchedKolmogorovFlow
import OpenKolmogorovFlow

const KS  = BatchedKolmogorovFlow
const OKF = OpenKolmogorovFlow

# ---- the device ----
const DEVICE = get(ENV, "DEVICE", "cpu")
if DEVICE == "gpu"
    using CUDA
    CUDA.allowscalar(false)
end
const ARRAY = DEVICE == "gpu" ? CuArray : Array

include(joinpath(@__DIR__, "..", "examples", "orbits.jl"))


# ---------------------------------------------------------------------------- #
# a small grid, and fields                                                     #
# ---------------------------------------------------------------------------- #

const n  = 8
const m  = 12
const Re = 40.0
const kf = 4
const dt = 0.01

# a batch on the device, and the coefficients or the values of a batch, or an array, on the CPU
dev(X::FTFields)      = FTFields(ARRAY(parent(X)))
cpu(X::FTFields)      = Array(parent(X))
cpu(u::Fields)        = Array(parent(u))
cpu(A::AbstractArray) = Array(A)

# the batch of the fields Ωs of OKF, on the device
batch(Ωs) = dev(FTFields(Ωs))

# the coefficients of a field of OKF in the layout of a batch
coefficients(Ω) = cpu(FTFields([Ω]))[:, :, 1]

# smooth fields of OKF, near the laminar flow or not
function okf_fields(B; seed=1, laminar=true)
    Random.seed!(seed)
    return [begin
                a = rand(6)
                u = OKF.Field(m, (x, y) -> sin(a[1] + x)*cos(2y + a[2]) + 0.5*sin(3x + 2y + a[3]) +
                                           0.3*cos(x - 4y + a[4]) + 0.2*sin(5x + a[5])*cos(y + a[6]))
                Ω = OKF.FFT(u, n)
                laminar && (Ω .+= OKF.laminarflow(n, m, Re, kf))
                Ω
            end for i in 1:B]
end

# a batch of fields with every mode active, the coefficients falling as 1/(1 + j² + k²), for the
# identities of the transposes
function random_fields(B; seed=7)
    Random.seed!(seed)
    j    = 0:n
    k    = [0:n; -n:-1]
    data = randn(ComplexF64, n + 1, 2n + 1, B) ./ (1 .+ j.^2 .+ k'.^2)

    return dev(KS.apply_symmetry!(FTFields(data)))
end

# a batch with what is not a field: a row j = 0 that is not Hermitian, and a mean
function junk(B; seed=3)
    Random.seed!(seed)
    J = zeros(ComplexF64, n + 1, 2n + 1, B)
    a = randn(ComplexF64, n, B)
    J[1, 2:(n + 1), :]           .= a
    J[1, (2n + 1):-1:(n + 2), :] .= .-conj.(a)
    J[1, 1, :]                   .= randn(B)

    return dev(FTFields(J))
end

# the batch X + a Y
plus(X, a, Y) = FTFields(X .+ a .* Y)

# the batch V as k blocks of perturbations, for tangents!
blocks(V, k) = FTFields(parent(V); L=k)

# the bundle of the fields X with their perturbations V, k′ blocks of as many, and the fields and
# the perturbations of a bundle Z of B fields, as two batches
bundle(X, V) = KS._bundle!(dev(FTFields(zeros(ComplexF64, n + 1, 2n + 1, size(X, 3) + size(V, 3)))), X, V)

function unbundle(Z, B)
    X = dev(FTFields(zeros(ComplexF64, n + 1, 2n + 1, B)))
    V = dev(FTFields(zeros(ComplexF64, n + 1, 2n + 1, size(Z, 3) - B)))
    KS._unbundle!(X, V, Z)

    return X, V
end

# the largest difference relative to the largest coefficient, on the CPU
reldiff(A, B) = maximum(abs, cpu(A) .- cpu(B)) / maximum(abs, cpu(B))

# the inner products of two batches, field by field, as a vector on the CPU
products(X, Y) = vec(Array(realdots(X, Y)))

# the relative differences of two vectors of inner products, entry by entry
relerr(a, b) = abs.(a .- b) ./ abs.(a)

# the memory that a second call of f allocates, the first having compiled it and built its storage:
# on the CPU, as @allocated counts it; on the GPU, the memory of the device, as CUDA.@allocated
# counts it, since the CPU there holds only the wrappers of the arrays and the launches of the
# kernels, a few hundred bytes each. Defined by @eval, so that the macro of CUDA is expanded only
# once CUDA is loaded
allocated(f) = (f(); @allocated f())
DEVICE == "gpu" && @eval allocated(f) = (f(); CUDA.@allocated f())


# ---------------------------------------------------------------------------- #
# OKF and Flows: the equation, its tangent, its adjoint, and their flows       #
# ---------------------------------------------------------------------------- #

const ex, im_   = OKF.splitexim(OKF.ForwardEquation(n, m, Re, kf, FFTW.ESTIMATE))
const lex, lim  = OKF.splitexim(OKF.LinearisedEquation(n, m, Re, OKF.TangentMode(), FFTW.ESTIMATE))
const aex, aim  = OKF.splitexim(OKF.LinearisedEquation(n, m, Re, OKF.AdjointMode(), FFTW.ESTIMATE))
const G  = flow(ex, im_, CB3R2R3e(OKF.FTField(n, m)), TimeStepConstant(dt))
const L  = flow(couple(ex, lex), couple(im_, lim), CB3R2R3e(couple(OKF.FTField(n, m), OKF.FTField(n, m))),
                TimeStepConstant(dt))
const LA = flow(aex, aim, CB3R2R3e(OKF.FTField(n, m), DiscreteMode(true)), TimeStepFromCache())


# ---------------------------------------------------------------------------- #
# the tests, by topic                                                          #
# ---------------------------------------------------------------------------- #

@info "testing on $(DEVICE), with batches of $(ARRAY)"

include("fields.jl")
include("operators.jl")
include("terms.jl")
include("propagator.jl")
include("adjoint.jl")
include("multipleshooting.jl")
