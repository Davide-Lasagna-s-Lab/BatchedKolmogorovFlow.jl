# The batches of the objects of the dynamics on a device, the CPU or the GPU.

# the arrays of the batches on a device: Array on the CPU, CuArray on the GPU, a method that the
# extension of the package for CUDA adds once CUDA is loaded
_array(::Val{:cpu}) = Array
_array(::Val)       = throw(ArgumentError("device=:gpu needs CUDA: `using CUDA` first"))

# a batch of b fields of largest wavenumber n on the device
_batch(n::Int, b::Int, device::Symbol) =
    FTFields(_array(Val(device)){ComplexF64}(undef, n + 1, 2n + 1, b))
