# The inverse transform of the nonlinear term on the GPU. cuFFT overwrites the input of a
# complex-to-real transform, and `mul!` protects it by copying it into a buffer of the plan first.
# The term writes its input again at every stage, so here the input is that buffer and the transform
# runs on it directly, which saves a read and a write of the array per stage. The zeros of that input
# are a fill of its words, 2.4 times faster than a kernel on an H200.
module KolmogorovShootingGPUCUDAExt

using CUDA

import KolmogorovShootingGPU: _input, _inverse!, _zero!

_input(plan::cuFFT.CuFFTPlan, F̂::DenseCuArray) = plan.buffer

_zero!(F̂::DenseCuArray) = (fill!(reinterpret(UInt32, F̂), 0x00000000); F̂)

_inverse!(f::DenseCuArray, plan::cuFFT.CuFFTPlan, F̂::DenseCuArray) =
    (cuFFT.unsafe_execute_external_batches!(plan, F̂, f); f)

end
