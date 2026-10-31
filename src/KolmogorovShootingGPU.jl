"""
    KolmogorovShootingGPU

Batched time stepping of two-dimensional Kolmogorov flow, the propagator of a
multiple-shooting search with all the segments integrated together, on the CPU
(`Array`, FFTW) or on the GPU (`CuArray`, cuFFT).

A batch of `B` vorticity fields is one complex array of size `(2m+2, m+2, B)`,
each slice laid out as an `FTField{n, m}` of OpenKolmogorovFlow. The code is
generic over the array type: it uses kernels, broadcasts and AbstractFFTs plans
over the first two dimensions only, so that a batch of `CuArray`s runs on the GPU
when CUDA is loaded, with no dependence of this package on CUDA.
"""
module KolmogorovShootingGPU

import LinearAlgebra: mul!
import AbstractFFTs: plan_rfft, plan_brfft
import FFTW
import Flows
import OpenKolmogorovFlow: shift!
using KernelAbstractions: @kernel, @index, get_backend

export KolmogorovPropagator, propagate!, tangent!, rhs!, shift!, dshift!, realdot

include("terms.jl")
include("propagator.jl")

end
