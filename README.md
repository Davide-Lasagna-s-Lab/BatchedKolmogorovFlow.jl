# BatchedKolmogorovFlow.jl

Two-dimensional Kolmogorov flow on batches of vorticity fields. The package
gives the vorticity equation, its tangent equation and the transpose of the
tangent, with all the fields of a batch advanced together, on a GPU, on one
core of a CPU or on several. A batch may hold the states of many flows, or the
segments of one orbit with their perturbations. The package gives the methods
that MultipleShooting.jl asks of a propagator, so that it serves the searches
of periodic orbits by multiple shooting.

The equation, its terms and their names are those of OpenKolmogorovFlow.jl
(OKF). The time stepping is that of Flows.jl, with constant steps: the IMEX
Runge–Kutta scheme `CB3R2R3e` by default, or `CNRK2`, `CB3R2R2`, `CB3R2R3c` or
`CB4R3R4` with the keyword `scheme`. OKF, Flows, MultipleShooting and NKRoots
are dependencies that are not registered: develop them in the environment.

## The state

A field is real and periodic on `[0, 2π)²`, with the Fourier coefficients

    ω(x, y) = Σⱼ Σₖ Ω[j, k] exp(i (j x + k y)),   j, k = -n, …, n.

Only the coefficients with `j ≥ 0` are kept, the others being their conjugates.
A batch of `B` fields is an `FTFields{n}`, an array `X` of size
`(n+1, 2n+1, B)`. Field `b` is `X[:, :, b]`, with `j = 0, …, n` along `x` down
the rows and `k` along `y` across the columns, in the order
`0, 1, …, n, -n, …, -1`. Two conditions make such a slice a field: in the row
`j = 0` the entries at `k` and `-k` are conjugates, and the mean is zero.
`apply_symmetry!` imposes them.

This is the `FTField{n, m}` of OKF without the modes that it keeps at zero, and
transposed. `FTFields(Ωs)` makes a batch from a vector of fields of OKF, and
`FTField(X, b, m)` gives one back. `FTFields(data)` wraps an array of
coefficients, and `parent(X)` returns it. A batch holds its wavenumbers, `X.j`
and `X.k`, computed once, as arrays that broadcast against it.

On the grid a batch is a `Fields{m}`, a real array `(2m+2, 2m+2, B)` with `x`
down the rows and `y` across the columns, the `Field{m}` of OKF transposed.

## On a GPU and on a CPU

The code is generic over the array of a batch. Every operation is a broadcast
or a transform over its first two dimensions, and acts on all the fields of
the batch. The keywords `array` and `tasks` of the propagator say where a
batch runs.

| keywords | where a batch runs |
|---|---|
| `array=CuArray` | On the GPU, which works on all the fields at once. cuFFT does the transforms. CUDA is loaded by the user and is not a dependency of the package. |
| `array=Array` | On one core of the CPU. FFTW does the transforms. An operation on an array runs on one core, so that a batch, however large, uses one core. |
| `array=Array, tasks=N` | On `N` cores of the CPU. The batch is cut into `N` chunks of fields, and `N` tasks advance one chunk each, at the same time, each as a batch of its own on its own thread. |

A batch of 10 fields with `tasks=4` is cut into the fields `1:2`, `3:5`, `6:7`
and `8:10`. The fields of a batch do not act on one another, since a field
with its perturbations is one segment of an orbit. The chunks are then
separate integrations, and the result is that of one task, to rounding. The
tasks run at the same time and cannot share arrays or plans, so that each has
its own storage. `propagate!`, `tangent!`, `tangents!` and `forced!` use the
tasks. The other functions run on one. Start Julia with at least `N` threads,
`julia -t N`.

## The files

| file | what it holds |
|---|---|
| `ftfields.jl` | `FTFields`, a batch in Fourier space, with its wavenumbers, `apply_symmetry!`, the inner product, the passage from and to OKF |
| `fields.jl` | `Fields`, a batch on the grid |
| `operators.jl` | `ddx!`, `ddy!`, `laplacian!`, `invlaplacian!` on batches |
| `fft.jl` | `InverseFFT!` and `ForwardFFT!`, between the two |
| `implicit.jl` | `ImplicitTerm`, the viscous term |
| `nonlinear.jl` | `ExplicitTerm`, advection and forcing |
| `linearised.jl` | `TangentTerm`, the linearised advection, and `ForcedTangentTerm` |
| `adjoint.jl` | `AdjointTerm`, the transpose of the linearised advection |
| `propagator.jl` | `KolmogorovPropagator`, the methods of MultipleShooting and `adjoint!`, then what serves them: the tasks, the storage |

The advection is written in the form of Basdevant (1983),
`(∂²/∂x² - ∂²/∂y²)(u v) + ∂²/∂x∂y (v² - u²)`, which needs the velocity alone on
the grid. The grid has `2m+2` points along each direction, at least `3n + 1`,
so that the products have no aliasing on the modes kept.

## Interface

```julia
using BatchedKolmogorovFlow

P = KolmogorovPropagator(n, m, Re, kf; dt=0.01, scheme=:CB3R2R3e, array=Array, tasks=1)
Q = KolmogorovPropagator(P, Re₂)
```

`scheme` is the scheme of Flows.jl, `:CNRK2`, `:CB3R2R2`, `:CB3R2R3e`,
`:CB3R2R3c` or `:CB4R3R4`. `array` and `tasks` are those of the section above.

The storage of the batches of `B` fields, the terms with their plans and
arrays and the registers of the scheme, lives in the propagator. It is built
once, at the first call with `B` fields, and every later call with `B` fields
only looks it up, so that a call allocates no batch, but for the stages that
`stages` returns. It is kept by batch size because MultipleShooting calls the
propagator with batches of different sizes, the seeds and the directions, and
by task. What the tangent equation adds for `K`
perturbations of `B` fields, and what `adjoint!` adds, its term and its
registers, are built at their own first calls, so that a search that never
calls them holds none of it. The second form is the propagator at another
Reynolds number. It shares all of it with `P` and costs nothing. The two must
not run at the same time.

| function | what it does |
|---|---|
| `propagate!(P, X, τ)` | advances every field of `X` in place by `τ`, in steps of `dt` and a last shorter step that ends on `τ` |
| `tangent!(P, X, V, τ)` | advances the fields `X` and the perturbations `V` together, `V` by the tangent equation, with the same scheme and steps: the derivative of the discrete flow |
| `tangents!(P, X, V, τ)` | as `tangent!`, for `k′` blocks of perturbations in `V`, one after the other, each about the fields of `X`, which are advanced once for all the blocks |
| `forced!(P, X, V, τ)` | as `tangent!`, with the tangent equation forced by the derivative of the right-hand side in the Reynolds number, `-Δω/Re²` |
| `rhs!(P, F, X)` | the time derivative of every field |
| `shift!(P, X, s)` | `ω(x, y) → ω(x + s, y)` for every field, the coefficients times `cis(j s)` |
| `dshift!(P, G, X, i)` | the generator of the shift, `∂ω/∂x` |
| `stages(P, X, τ)` | the stages of every step of the flow over `τ` from the fields `X`, which are left as they are, kept for `adjoint!` |
| `adjoint!(P, S, W)` | the transpose of `tangent!` applied to `W`, back over the stages `S` |
| `realdots(X, Y)` | the inner products of the fields one by one, the mean of `ω₁ ω₂`, a real array `(1, 1, B)` |
| `realdot(X, Y)` | their sum over the fields |

The first seven are methods of the functions of MultipleShooting, and
`adjoint!` is a method of `LinearAlgebra.adjoint!`, exported. Every function takes views
of a batch, `view(X, :, :, b)`. `propagate!`, `tangent!`, `tangents!`, `forced!`,
`stages`, `adjoint!` and `rhs!` first give their input the symmetry of fields, by
`apply_symmetry!`, so that the flow, its tangent and its adjoint vanish on what
is not a field.

Field by field, `propagate!`, `tangent!`, `adjoint!` and `rhs!` give the
results of OKF with Flows to rounding. `forced!` treats its forcing as an
explicit term, so that it gives the derivative of the flow in the Reynolds
number up to the error of the scheme.

## The adjoint

`adjoint!` is the transpose of the derivative of the discrete flow in the inner
product of the fields, `realdots`. For any batches `V` and `W`, field by field
and to rounding,

    realdots(tangent!(P, copy(X), V, τ), W) == realdots(V, adjoint!(P, stages(P, X, τ), W))

It is the adjoint of the scheme in Flows, which has none for `CB4R3R4`, with
the explicit term `AdjointTerm`, stepped backwards over the stages of the steps
of `propagate!`.
`AdjointTerm` is `TangentTerm` read backwards, each step replaced by its
transpose, the forward transform by the inverse one and the inverse by the
forward one. It equals the adjoint of the linearised advection,

    u ∂μ/∂x + v ∂μ/∂y + Δ⁻¹(∂μ/∂x ∂ω/∂y - ∂μ/∂y ∂ω/∂x),

the `LinearisedExTerm` of OKF in its adjoint mode, to rounding.

The adjoint needs the stages from the last step back, so that they are made
first, once, and kept:

```julia
S = stages(P, X, τ)       # a forward pass from the fields X, every stage kept
adjoint!(P, S, W)         # a backward pass over them, for this W
adjoint!(P, S, W₂)        # and for any other, with no new forward pass
```

`S` holds the stages of each of the `N` steps over `τ`, four with the default
scheme, `4N` batches like `X`. For one field with `n = 33` and steps of 0.01
this is 14.6 GB over
1000 time units, whatever the number of segments the orbit is cut into, and
146 GB over 10⁴. `W` has as many fields as the batch that made `S`. For
several adjoint fields about each field of `X`, call `adjoint!` once for each,
over the same `S`.

## With MultipleShooting

```julia
using MultipleShooting

z = MultiVector(X, [T, s])                     # X an FTFields, the seeds
F = Residual(P)
J = Jacobian(P, z)
```

The inner product of the fields, `realdots`, is the `dots` of MultipleShooting
for a batch of fields, so that the Krylov method, the phase conditions and the
preconditioner all use it. For a search or a
continuation with the Reynolds number free, the family of propagators is
`Re -> KolmogorovPropagator(P, Re)`. The segments have equal lengths.

## Tests and examples

```
julia --project=test test/runtests.jl
DEVICE=gpu julia --project=test test/runtests.jl
julia --project=examples examples/timing.jl cpu 10 1 4 16
julia --project=<environment with CUDA> examples/timing.jl gpu 10 1 4 16 64 256 1024
```

The tests compare every function with OKF and Flows on a small grid (`n = 8`,
`m = 12`), field by field, for several batch sizes and on views. They check
`tangent!` against a centred difference of `propagate!`, with its error
falling as the square of the increment, `rhs!` against a short step of
`propagate!`, `forced!` against a centred difference in the Reynolds number,
and `adjoint!` and `AdjointTerm` against `tangent!` and `TangentTerm` by the
identity of the transpose. They check that the flow commutes with the shift
along `x`, the rotation by π and the shift-reflect, that the advection keeps
the energy and the enstrophy, that the storage of a batch size is built once,
and the methods of MultipleShooting on a product with its Jacobian. The files
of `test/` hold them by topic.

With `DEVICE=gpu` the batches are `CuArray`s. CUDA is a dependency neither of
the package nor of the test environment: it is loaded from another environment
of the load path, the global one by default.

`examples/timing.jl` times one `propagate!` and one `tangent!` of `B` fields
over `τ = 10` at the resolution of the campaigns, `n = 33` and `m = 49`.
`examples/orbits.jl` reads the orbits and the seeds of the campaigns into
batches of seeds. A state that must fill several segments is advanced by whole
steps of `dt`, and each shooting point takes the rest of a step on a copy. A
shorter step that the integration went on from would leave the trajectory of
the simulation about `1e-9` off, and the recurrence would be lost by
`T ≈ 150`. An orbit that closes after a translation along `y` is refused.
