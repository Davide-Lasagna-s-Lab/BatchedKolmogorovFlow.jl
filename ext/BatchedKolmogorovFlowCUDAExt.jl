# The batches on the GPU: loaded with CUDA, which is not a dependency of the package, so that
# `device=:gpu` makes the batches of a propagator CuArrays.
module BatchedKolmogorovFlowCUDAExt

import CUDA
import BatchedKolmogorovFlow

BatchedKolmogorovFlow._array(::Val{:gpu}) = CUDA.CuArray

end
