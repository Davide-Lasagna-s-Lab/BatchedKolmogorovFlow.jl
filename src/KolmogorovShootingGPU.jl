"""
    KolmogorovShootingGPU

Two-dimensional Kolmogorov flow for multiple shooting: the vorticity equation, its tangent
equation and the transpose of the tangent on batches of fields, all the fields of a batch
advanced together, on the CPU (`Array`, FFTW) or on the GPU (`CuArray`, cuFFT), with the methods
that MultipleShooting.jl asks of a propagator.

The equation, its terms and their names are those of OpenKolmogorovFlow.jl, and the time stepping
is that of Flows.jl. A batch of `B` fields is an [`FTFields`](@ref), the Fourier coefficients of
the fields in one array `(n+1, 2n+1, B)`. Every operation is a broadcast or a transform over the
first two dimensions of that array, so that a batch on a `CuArray` runs on the GPU once CUDA is
loaded, with no dependence of this package on CUDA.
"""
module KolmogorovShootingGPU

import LinearAlgebra
import LinearAlgebra: adjoint!
import Random
import AbstractFFTs
import FFTW
import Flows
import OpenKolmogorovFlow
import MultipleShooting: propagate!, tangent!, tangents!, forced!, rhs!, shift!, dshift!

export KolmogorovPropagator,
       FTFields,
       Fields,
       propagate!,
       tangent!,
       tangents!,
       forced!,
       rhs!,
       shift!,
       dshift!,
       stages,
       adjoint!,
       realdot,
       realdots

include("ftfields.jl")
include("fields.jl")
include("operators.jl")
include("fft.jl")
include("implicit.jl")
include("nonlinear.jl")
include("linearised.jl")
include("adjoint.jl")
include("propagator.jl")

end
