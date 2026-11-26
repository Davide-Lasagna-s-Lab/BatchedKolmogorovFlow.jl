# Draws assets/logo.svg: a batch of fields as one array, a block of slabs one behind the other, as
# in the figures of Lasagna et al. (2026), with a vorticity field of Kolmogorov flow, computed with
# the package, on the face of the first.
#
#     julia --project=<an environment with the package and CairoMakie> assets/make_logo.jl
using BatchedKolmogorovFlow
using Base64
using CairoMakie
using Printf
using Random

const KF = BatchedKolmogorovFlow


# ---- the field: one of a batch of four, from smooth random ones, long after their start ----

n, m, Re, kf = 42, 63, 40.0, 4

Random.seed!(4)
X  = randn!(FTFields(n, 4))
X .= X ./ (1 .+ X.j.^2 .+ X.k.^2)
G  = NonlinearPropagator(n, m, Re, kf; B=4)
G(X, 174.0)

# the vorticity of the second field on the grid, as a picture with no margin, for the svg
ω   = Array(parent(KF.InverseFFT!(X, m)(KF.Fields(X, m), copy(X))))[:, :, 2]
fig = Figure(size=(600, 600), figure_padding=0)
ax  = Axis(fig[1, 1])
hidedecorations!(ax)
hidespines!(ax)
heatmap!(ax, ω; colormap=Reverse(:RdBu), colorrange=(-6, 6), interpolate=true)
file = tempname() * ".png"
save(file, fig; px_per_unit=1)
picture = base64encode(read(file))


# ---- the block: six fields, the first seen from the front, the others from above and from the
#      side, in the blue of the figures ----

const BLUE = (0x19, 0x76, 0xd2)

hex(c)        = @sprintf("#%02x%02x%02x", round.(Int, c)...)
tint(p)       = hex(BLUE .* (p/100) .+ 255 .* (1 - p/100))     # p percent of the blue, on white
const EDGE    = hex(BLUE .* 0.8)                                # the blue, darker, of the edges
points(ps...) = join([@sprintf("%.1f,%.1f", p...) for p in ps], " ")

fields = 6
O = (78.0, 224.0)       # the lower left corner of the first field
W = (150.0, 38.0)       # its lower edge
H = (0.0, -142.0)       # its left edge
D = (11.0, -6.8)        # from a field to the one behind it

at(a, b, c) = O .+ a .* W .+ b .* H .+ c .* D

face(colour, ps...) = @sprintf("<polygon points=\"%s\" fill=\"%s\" stroke=\"%s\" stroke-width=\"1.8\" stroke-linejoin=\"round\"/>",
                               points(ps...), colour, EDGE)

block = String[]
for i in fields:-1:1
    # the top and the side of field i, darker from the first to the last
    push!(block, face(tint(14 + 7*(i - 1)), at(0, 1, i - 1), at(1, 1, i - 1), at(1, 1, i), at(0, 1, i)))
    push!(block, face(tint(36 + 7*(i - 1)), at(1, 0, i - 1), at(1, 0, i), at(1, 1, i), at(1, 1, i - 1)))
end

# the field on the face of the first: the picture, sheared onto the face, and its edge
tl    = at(0, 1, 0)
image = @sprintf("<image href=\"data:image/png;base64,%s\" x=\"0\" y=\"0\" width=\"100\" height=\"100\" preserveAspectRatio=\"none\" transform=\"matrix(%.4f %.4f %.4f %.4f %.1f %.1f)\"/>",
                 picture, W[1]/100, W[2]/100, -H[1]/100, -H[2]/100, tl...)
edge  = @sprintf("<polygon points=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"2.2\" stroke-linejoin=\"round\"/>",
                 points(at(0, 0, 0), at(1, 0, 0), at(1, 1, 0), at(0, 1, 0)), EDGE)


# ---- the svg ----

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 300" role="img" aria-labelledby="title desc">
  <title id="title">BatchedKolmogorovFlow.jl logo</title>
  <desc id="desc">A batch of fields drawn as one array, a block of slabs one behind the other, with a vorticity field of Kolmogorov flow on its front face, beside the BatchedKolmogorovFlow.jl wordmark.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <filter id="panel-shadow" x="-12%" y="-25%" width="124%" height="150%">
      <feDropShadow dx="0" dy="9" stdDeviation="11" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#172033" flood-opacity="0.18"/>
    </filter>
  </defs>

  <!-- Main panel -->
  <rect x="24" y="24" width="1032" height="252" rx="24" fill="url(#bg)" filter="url(#panel-shadow)"/>

  <!-- The batch: the block of fields, and the field on its face -->
  <g filter="url(#soft)">
    $(join(block, "\n    "))
    $image
    $edge
  </g>

  <!-- Wordmark and subtitle -->
  <g transform="translate(372 100)">
    <text x="0" y="48" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="46" font-weight="820" fill="#222734" letter-spacing="0">BatchedKolmogorovFlow.jl</text>
    <text x="2" y="90" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">Kolmogorov flow on batches of fields, its tangent</text>
    <text x="2" y="118" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">and its adjoint, on a GPU or on a CPU.</text>
  </g>
</svg>
"""

write(joinpath(@__DIR__, "logo.svg"), svg)
println("written assets/logo.svg, ", round(Int, sizeof(svg)/1000), " kB")
