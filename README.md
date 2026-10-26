# KolmogorovShootingGPU.jl

A batched propagator of two-dimensional Kolmogorov flow, for multiple shooting
with all the segments integrated together, on the CPU or on the GPU.

The vorticity equation is that of OpenKolmogorovFlow.jl (OKF), advanced by
the IMEX Runge–Kutta scheme `CB3R2R3e` of Flows.jl with constant steps. Both
are dependencies, not registered: develop them in the environment. A batch of `B` fields is one complex array, and every
operation acts on all its fields at once: the derivatives, the masks and the
products are broadcasts, the transforms of the nonlinear term are plans over
the first two dimensions, batched over the third.

## The state

A batch is a complex array `X` of size `(2m+2, m+2, B)`. Slice `X[:, :, i]` is
the data of an `FTField{n, m}` of OKF, `parent(Ω)`: the signed wavenumber `k`
along `y` down the rows in FFT order, `j ≥ 0` along `x` across the columns.
`FTField(X[:, :, i], n)` is that field in OKF. `n` is the largest active
wavenumber and `2m+2` the number of grid points along each direction; the
campaigns use `n = 33`, `m = 49`.

The code is generic over the array type. With `array=CuArray` (CUDA loaded by
the user, not a dependency of the package) the batches live on the GPU and
cuFFT does the transforms; nothing indexes scalars. The real-to-complex
transforms of cuFFT halve the first dimension of the region while the fields of
OKF halve the second, so the transforms work on transposed copies, written and
read through a `PermutedDimsArray` in the broadcasts that apply the
derivatives, the mask and the normalisation.

## Interface

```julia
using KolmogorovShootingGPU

P = KolmogorovPropagator(n, m, Re, kf; dt=0.01, batch=N, array=Array)
```

One constructor. The work arrays and the plans of batches of `batch` fields
are built at once, those of any other size at its first use and kept, one set
per size (`P.cache`): the Newton operator integrates `N` fields, the coarse
preconditioner `N k` at once.

| function | what it does |
|---|---|
| `propagate!(P, X, τ)` | advances every field of `X` in place by `τ`, `floor(τ/dt)` steps of `dt` and a last shorter step that ends on `τ`, as `Flows.TimeStepConstant`; returns `X` |
| `tangent!(P, X, V, τ)` | advances the fields `X` and the perturbations `V` together, `V` by the tangent equation of OKF (`LinearisedEquation(TangentMode())`) coupled to the equation, with the same scheme and steps: the exact derivative of the discrete flow map; returns `V` |
| `rhs!(P, out, X)` | the time derivative at every field, as the call of OKF's `ForwardEquation` |
| `shift!(P, X, s, ℓy=0)` | `ω(x, y) → ω(x + s, y + ℓy)` for every field of `X`, the coefficients times `cis(j s + k ℓy)`; OKF's `xshift!` for `ℓy = 0`. `X` may be a view, of one field or of a batch |
| `dshift!(P, out, X)` | the generator of the shift along `x`, `im j X`, as OKF's `ddx!` |
| `realdot(X, Y)` | the inner product of OKF (`dot` of `FTField`s), summed over the fields, one reduction |

`X` passed to `propagate!` and `tangent!` must be an array of the type of the
propagator, not a view (Flows dispatches on the type of the state). `shift!`,
`dshift!` and `realdot` take views and any number of trailing batch
dimensions. `shift!` is a method of OKF's `shift!`, so both packages can be
loaded together.

`realdot` is the spatial mean of `ω₁ ω₂`: twice the modes `j ≥ 1`, which stand
for their conjugates, once those of the column `j = 0`, and half the mean.
`realdot(X, X)` is the mean of `ω²` summed over the fields, `Re` times OKF's
`dissrate`. It counts all the stored modes, the same as OKF's `dot` for fields
with zero inactive modes and zero mean, as those of the propagator.

Field by field, `propagate!`, `tangent!` and `rhs!` reproduce OKF with Flows to
round-off (`1e-13` relative over a few hundred steps in the tests), the
implicit term exactly.

## Tests and examples

```
julia --project=test test/runtests.jl
julia --project=examples examples/timing.jl cpu 10 1 4 16
julia --project=<environment with CUDA> examples/timing.jl gpu 10 1 4 16 64 256 1024
```

The tests compare every function with OKF and Flows on a small grid (`n = 8`,
`m = 12`), slice by slice, for several batch sizes, and `tangent!` also with
a centred difference of `propagate!`. `examples/timing.jl` times one
`propagate!` and one `tangent!` of `B` fields over `τ = 10` at the resolution of
the campaigns. `examples/orbits.jl` reads the orbits and the seeds of the
campaigns (`nksearch-upos`, `orbits`, recurrences.jl) into batches of seeds.

On one core of the laptop (M-series, FFTW), one field over `τ = 10` at
`n = 33` takes about 1 s with `propagate!`, close to OKF, and 2.7 s with
`tangent!`;
the cost grows linearly with `B`. The batched code is single-threaded on the
CPU (Julia broadcasts), so OKF with one thread per segment is faster on a
multicore node: the package is for the GPU.
