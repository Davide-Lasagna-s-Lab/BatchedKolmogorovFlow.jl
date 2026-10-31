# KolmogorovShootingGPU.jl

A batched propagator of two-dimensional Kolmogorov flow, for multiple shooting
with all the segments integrated together, on the CPU or on the GPU.

The vorticity equation is that of OpenKolmogorovFlow.jl (OKF), advanced by
the IMEX Runge–Kutta scheme `CB3R2R3e` of Flows.jl with constant steps. Both
are dependencies, not registered: develop them in the environment. A batch of `B` fields is one complex array, and every
operation acts on all its fields at once: the derivatives and the
products are kernels of KernelAbstractions.jl, the transforms of the nonlinear
term are plans over the first two dimensions, batched over the others.

## The state

A batch is a complex array `X` of size `(2m+2, m+2, B)`. Slice `X[:, :, i]` is
the data of an `FTField{n, m}` of OKF, `parent(Ω)`: the signed wavenumber `k`
along `y` down the rows in FFT order, `j ≥ 0` along `x` across the columns.
`FTField(X[:, :, i], n)` is that field in OKF. `n` is the largest active
wavenumber and `2m+2` the number of grid points along each direction; the
campaigns use `n = 33`, `m = 49`.

The code is generic over the array type. With `array=CuArray` (CUDA loaded by
the user, not a dependency of the package) the batches live on the GPU and
cuFFT does the transforms; nothing indexes scalars. An extension then runs the
inverse transform without the copy that the `mul!` of cuFFT makes of its input.

Inside, the time stepper acts on compact batches of the active modes alone, a
complex array `(n+1, 2n+1, B)` with `j` down the rows and `k = 0, …, n, -n, …, -1`
across the columns, 45% of the stored entries. The real-to-complex transforms of
cuFFT halve the first dimension of the region, so `x` runs along the first
dimension of the compact batches and of the transforms. `propagate!`, `tangent!`
and `rhs!` convert from the fields and back at their ends, which is the
projection onto the fields and one transposition.

A stage of `propagate!` is two kernels and two transforms. The kernel in Fourier
space takes the transforms of the products, makes the explicit term, updates the
state, solves the implicit stage and writes the velocity into the input of the
inverse transforms. The kernel in physical space makes the products. The time of
a few fields is the time to launch the kernels, and of many the memory they read
and write, so both ends gain from having two kernels and few arrays. `tangent!`
uses the same terms in the step of Flows.

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

`propagate!`, `tangent!` and `rhs!` take batches of three dimensions, views too.
`shift!`, `dshift!` and `realdot` take views and any number of trailing batch
dimensions. `shift!` is a method of OKF's `shift!`, so both packages can be
loaded together.

`realdot` is the spatial mean of `ω₁ ω₂`: twice the modes `j ≥ 1`, which stand
for their conjugates, once those of the column `j = 0`, and half the mean.
`realdot(X, X)` is the mean of `ω²` summed over the fields, `Re` times OKF's
`dissrate`. It counts all the stored modes, the same as OKF's `dot` for fields
with zero inactive modes and zero mean, as those of the propagator.

Field by field, `propagate!`, `tangent!` and `rhs!` reproduce OKF with Flows to
round-off (`1e-13` relative over a few hundred steps in the tests), the
implicit term exactly. The nonlinear term is written in the form of Basdevant
(1983), `-(∂²/∂x² - ∂²/∂y²)(u v) - ∂²/∂x∂y (v² - u²)`, with two inverse and two
forward transforms against the four and one of OKF. The two forms agree to
round-off on a grid that dealiases the products, so `2m + 2 ≥ 3n + 1` is
required. The tangent adds two and two transforms, and takes the velocity of
the base from the term of the equation.

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
campaigns (`nksearch-upos`, `orbits`, recurrences.jl) into batches of seeds. A
state that must fill several segments is advanced by whole steps of `dt`, and
each shooting point takes the rest of a step on a copy. A shorter step that the
integration went on from would leave the trajectory of the simulation about
`1e-9` off, and the recurrence would be lost by `T ≈ 150`.

On one core of a laptop (M-series, FFTW), one field over `τ = 10` at `n = 33`
takes about 0.5 s with `propagate!`, half the time of OKF, and 1.2 s with
`tangent!`, and the cost grows linearly with `B`. The kernels run on the threads
of Julia. On an H200 one field takes 0.11 s and a field in a batch of thousands
2.7 ms, so the package is for the GPU.
