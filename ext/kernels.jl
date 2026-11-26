# The terms of the equation on the GPU, by three kernels: the method of `_kernels!` for the
# batches on the GPU. What one entry of each kernel is, and why, is in src/kernels.jl; a kernel
# here gives every thread its entry.

const KF = BatchedKolmogorovFlow

# the entry of a thread among all those of a launch, in the order of the entries of an array
@inline _thread() = (CUDA.blockIdx().x - 1)*CUDA.blockDim().x + CUDA.threadIdx().x

# the kernel f launched over N entries; CUDA.jl compiles it at its first launch, and keeps it
_launch(f, N::Int, args...) = (CUDA.@cuda threads=256 blocks=cld(N, 256) f(args...); nothing)

function _velocity_kernel!(Û, Ω, n::Int)
    i = _thread()
    if i <= length(Û)
        @inbounds r, c, b = Tuple(CartesianIndices(Û)[i])
        @inbounds Û[i] = KF._velocity_entry(Ω, n, size(Û, 2), r, c, b)
    end

    return nothing
end

function _products_kernel!(pq, uv, k′::Int)
    i = _thread()
    N = length(uv) >> 1
    if i <= N
        p, q = KF._products_entry(uv, k′, i)
        @inbounds pq[i]     = p
        @inbounds pq[i + N] = q
    end

    return nothing
end

function _advection_kernel!(dZdt, P̂, Z, n::Int, kf::Int, k′::Int, γ::Float64)
    i = _thread()
    if i <= length(dZdt)
        @inbounds r, c, b = Tuple(CartesianIndices(dZdt)[i])
        @inbounds dZdt[i] = KF._advection_entry(P̂, Z, n, kf, k′, γ, r, c, b)
    end

    return nothing
end

function KF._kernels!(dZdt::KF.FTFields{n, <:CUDA.CuArray},
                        eq,
                         Z::KF.FTFields{n, <:CUDA.CuArray},
                        k′::Int,
                         γ::Float64) where {n}
    Û  = eq.ifft!.padded
    P̂  = eq.fft!.padded
    uv = parent(eq.uv)
    pq = parent(eq.pq)

    # ---- in Fourier space: the velocity, among the wavenumbers of the grid ----
    _launch(_velocity_kernel!, length(Û), Û, parent(Z), n)

    # ---- to the grid, the products there, and back ----
    LinearAlgebra.mul!(uv, eq.ifft!.plan, Û)
    _launch(_products_kernel!, length(uv) >> 1, pq, uv, k′)
    LinearAlgebra.mul!(P̂, eq.fft!.plan, pq)

    # ---- in Fourier space: the advection and the forcings ----
    _launch(_advection_kernel!, length(dZdt), parent(dZdt), P̂, parent(Z), n, eq.kf, k′, γ)

    return dZdt
end
