"""
    BatchedKolmogorovFlow

Two-dimensional Kolmogorov flow on batches of vorticity fields: the vorticity equation, its
tangent equation and the transpose of the tangent, with all the fields of a batch advanced
together, on a GPU (`CuArray`, cuFFT), on one core of a CPU (`Array`, FFTW) or on several, a
chunk of the batch on each. A batch may hold the states of many flows, or the segments of one
orbit with their perturbations, and the package gives the objects of the dynamics that
MultipleShooting.jl calls: the flow, the tangent, the right-hand side and the shift.

The equation, its terms and their names are those of OpenKolmogorovFlow.jl, and the time stepping
is that of Flows.jl. A batch of `B` fields is an [`FTFields`](@ref), the Fourier coefficients of
the fields in one array `(n+1, 2n+1, B)`. Every operation is a broadcast or a transform over the
first two dimensions of that array, so that a batch on a `CuArray` runs on the GPU once CUDA is
loaded, with no dependence of this package on CUDA.
"""
module BatchedKolmogorovFlow

import LinearAlgebra
import Random
import AbstractFFTs
import FFTW
import Flows
import OpenKolmogorovFlow
import MultipleShooting
import MultipleShooting: dots

export NonlinearPropagator,
       TangentPropagator,
       AdjointPropagator,
       RightHandSide,
       XShift,
       FTFields,
       Fields,
       stages,
       dots

include("ftfields.jl")
include("fields.jl")
include("operators.jl")
include("fft.jl")
include("implicit.jl")
include("nonlinear.jl")
include("linearised.jl")
include("dynamics/devices.jl")
include("dynamics/chunks.jl")
include("dynamics/flows.jl")
include("dynamics/nonlinearpropagator.jl")
include("dynamics/tangentpropagator.jl")
include("dynamics/righthandside.jl")
include("dynamics/xshift.jl")
include("dynamics/adjointpropagator.jl")

end
