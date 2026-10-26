# The cost of one propagate! of B fields of Kolmogorov flow over τ = 10, n = 33 on
# a grid of 100² (m = 49), Re = 40, kf = 4, CB3R2R3e with dt = 0.01, and of one
# tangent! of B bases and B perturbations, against the integration of one field
# with OpenKolmogorovFlow and Flows.jl on the CPU.
#
#     julia --project=<this folder> timing.jl cpu [τ] [B₁ B₂ …]
#     julia --project=<an environment with CUDA> timing.jl gpu [τ] [B₁ B₂ …]
#
# On the GPU the environment needs CUDA next to the packages of this folder; the
# batches are CuArrays and cuFFT does the transforms. Each case is timed after a
# first call that compiles and plans, with a synchronisation of the device
# before the clock stops. Defaults: τ = 10, B = 1 4 16 on the CPU and
# B = 1 4 16 64 256 1024 on the GPU.

using Printf
using FFTW
using Flows
using OpenKolmogorovFlow
using KolmogorovShootingGPU

const DEVICE = length(ARGS) > 0 ? ARGS[1] : "cpu"
const TAU    = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 10.0
const BS     = length(ARGS) > 2 ? parse.(Int, ARGS[3:end]) :
               DEVICE == "gpu" ? [1, 4, 16, 64, 256, 1024] : [1, 4, 16]

if DEVICE == "gpu"
    using CUDA
end

const n  = 33
const m  = 49
const Re = 40.0
const kf = 4

# ---- the device: the array type and the barrier that waits for its kernels ----
const ARRAY = DEVICE == "gpu" ? CuArray : Array
wait!()     = DEVICE == "gpu" ? CUDA.synchronize() : nothing

# ---- a field on the attractor: the laminar flow perturbed and advanced by 50 time units ----
ex, im_ = splitexim(ForwardEquation(n, m, Re, kf, FFTW.ESTIMATE))
G       = flow(ex, im_, CB3R2R3e(FTField(n, m)), TimeStepConstant(0.01))
Ω       = laminarflow(n, m, Re, kf)
Ω     .+= FFT(Field(m, (x, y) -> 0.1*sin(x + 2y) + 0.1*cos(3x - y)), n)
G(Ω, (0, 50))

# ---- the reference: one field with OKF, on the CPU ----
G(copy(Ω), (0, 0.1))
t_okf = @elapsed G(copy(Ω), (0, TAU))
@printf "OKF on the CPU, one field over τ = %g: %.3f s\n" TAU t_okf

# ---- batches ----
P = KolmogorovPropagator(n, m, Re, kf; dt=0.01, array=ARRAY)
for B in BS
    X = ARRAY(repeat(parent(Ω), 1, 1, B))
    V = ARRAY(repeat(parent(Ω), 1, 1, B))

    # ---- propagate! ----
    propagate!(P, copy(X), 0.1); wait!()
    t = @elapsed (propagate!(P, X, TAU); wait!())
    @printf "%s propagate! of %5d fields over τ = %g: %9.3f s, %.4f s per field, %6.3f × OKF\n" DEVICE B TAU t t/B t/(B*t_okf)

    # ---- tangent! ----
    tangent!(P, copy(X), copy(V), 0.1); wait!()
    t = @elapsed (tangent!(P, X, V, TAU); wait!())
    @printf "%s tangent!   of %5d pairs  over τ = %g: %9.3f s, %.4f s per pair,  %6.3f × OKF\n" DEVICE B TAU t t/B t/(B*t_okf)

    # free the work arrays of this size before the next one
    empty!(P.cache)
    GC.gc()
end
