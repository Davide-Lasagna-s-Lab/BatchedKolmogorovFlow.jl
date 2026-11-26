# Draws assets/logo.svg: a batch of fields as one array, the fields one behind the other along its
# third dimension, as in the figures of Lasagna et al. (2026), and Andrey Kolmogorov, who gave the
# flow its name, in the glasses that the batches deserve.
#
#     julia assets/make_logo.jl
using Printf


# ---- the colours: a tint of a colour, p percent of it on white, and a shade, on black ----

const BLUE = (0x19, 0x76, 0xd2)

hex(c)        = @sprintf("#%02x%02x%02x", round.(Int, c)...)
tint(c, p)    = hex(c .* (p/100) .+ 255 .* (1 - p/100))
shade(c, p)   = hex(c .* (p/100))
points(ps...) = join([@sprintf("%.1f,%.1f", p...) for p in ps], " ")


# ---- the batch: four fields, slabs of 4 × 8 cells, seen from the front and from above ----

O  = (58.0, 192.0)          # the lower left corner of the first field
e₁ = (16.5, 5.4)            # a cell along the columns of a field
e₂ = (0.0, -24.0)           # a cell along its rows
e₃ = (13.0, -8.0)           # from a field to the one behind it

at(a, b, c) = O .+ a .* e₁ .+ b .* e₂ .+ c .* e₃

line(p, q, colour) = @sprintf("<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"1\"/>",
                              p..., q..., colour)
face(colour, ps...) = @sprintf("<polygon points=\"%s\" fill=\"%s\" stroke=\"%s\" stroke-width=\"1.6\" stroke-linejoin=\"round\"/>",
                               points(ps...), colour, shade(BLUE, 80))

batch = String[]
for i in 4:-1:1
    # the top and the side of field i, darker from the first to the last, with their cells
    push!(batch, face(tint(BLUE, 14 + 14*(i - 1)), at(0, 4, i - 1), at(8, 4, i - 1), at(8, 4, i), at(0, 4, i)))
    push!(batch, face(tint(BLUE, 36 + 14*(i - 1)), at(8, 0, i - 1), at(8, 0, i), at(8, 4, i), at(8, 4, i - 1)))
    append!(batch, [line(at(a, 4, i - 1), at(a, 4, i), tint(BLUE, 58 + 10*(i - 1))) for a in 1:7])
    append!(batch, [line(at(8, b, i - 1), at(8, b, i), tint(BLUE, 76 + 8*(i - 1))) for b in 1:3])
end

# the first field, from the front
push!(batch, face(tint(BLUE, 24), at(0, 0, 0), at(8, 0, 0), at(8, 4, 0), at(0, 4, 0)))
append!(batch, [line(at(a, 0, 0), at(a, 4, 0), tint(BLUE, 52)) for a in 1:7])
append!(batch, [line(at(0, b, 0), at(8, b, 0), tint(BLUE, 52)) for b in 1:3])


# ---- the glasses: black pixels, and the glint of each lens ----

pixels = ["######################",
          "..#oo#o####..#oo#o####",
          "..##oo#o###..##oo#o###",
          "...##oo#o#....##oo#o#.",
          "....#####......#####.."]
px     = 4.6
corner = (-12px, -37.0)     # the bridge of the glasses over the nose

glasses = [@sprintf("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"%s\"/>",
                    corner[1] + (c - 1)*px, corner[2] + (r - 1)*px, px + 0.3, px + 0.3,
                    pixels[r][c] == '#' ? "#111318" : "#f4f6f9")
           for r in eachindex(pixels) for c in 1:length(pixels[r]) if pixels[r][c] != '.']


# ---- the svg ----

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 300" role="img" aria-labelledby="title desc">
  <title id="title">BatchedKolmogorovFlow.jl logo</title>
  <desc id="desc">A batch of fields drawn as one array, the fields one behind the other, and a portrait of Andrey Kolmogorov in pixel sunglasses, beside the BatchedKolmogorovFlow.jl wordmark.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <filter id="panel-shadow" x="-12%" y="-25%" width="124%" height="150%">
      <feDropShadow dx="0" dy="9" stdDeviation="11" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <clipPath id="medal">
      <circle cx="0" cy="0" r="86"/>
    </clipPath>
  </defs>

  <!-- Main panel -->
  <rect x="24" y="24" width="1032" height="252" rx="24" fill="url(#bg)" filter="url(#panel-shadow)"/>

  <!-- The batch -->
  <g filter="url(#soft)">
    $(join(batch, "\n    "))
  </g>

  <!-- Kolmogorov -->
  <g transform="translate(332 150)">
    <circle cx="0" cy="0" r="86" fill="#e7edf5" stroke="#c6ccd6" stroke-width="3"/>
    <g clip-path="url(#medal)">
      <!-- the jacket, the shirt and the tie -->
      <path d="M-96 96 Q-90 50 -46 40 L-18 30 L18 30 L46 40 Q90 50 96 96 Z" fill="#2b3447"/>
      <path d="M-18 30 L0 64 L18 30 Z" fill="#ffffff"/>
      <path d="M-6 37 L6 37 L4 47 L9 96 L-9 96 L-4 47 Z" fill="#a23b45"/>
      <path d="M-18 30 L-34 48 L-8 74 M18 30 L34 48 L8 74" fill="none" stroke="#1d2433" stroke-width="2.5" stroke-linejoin="round"/>
      <!-- the neck, the ears and the head -->
      <path d="M-15 10 L-15 33 Q0 46 15 33 L15 10 Z" fill="#dfae88"/>
      <ellipse cx="-39" cy="-14" rx="7" ry="11" fill="#eec2a0"/>
      <ellipse cx="39" cy="-14" rx="7" ry="11" fill="#eec2a0"/>
      <path d="M-36 -34 Q-38 -66 0 -68 Q38 -66 36 -34 L35 -4 Q31 23 0 29 Q-31 23 -35 -4 Z" fill="#f3cfae"/>
      <!-- the hair, white, combed back -->
      <path d="M-40 -16 Q-54 -58 -26 -73 Q2 -87 32 -72 Q54 -56 40 -16 Q39 -40 28 -50 Q6 -62 -18 -54 Q-35 -46 -40 -16 Z" fill="#f6f6f4" stroke="#cfd3d8" stroke-width="1.5"/>
      <!-- the eyebrows, the nose and the mouth -->
      <path d="M-30 -36 Q-19 -42 -7 -36 M7 -36 Q19 -42 30 -36" fill="none" stroke="#9aa0a8" stroke-width="4.5" stroke-linecap="round"/>
      <path d="M-2 -10 Q-9 6 -1 9 Q6 10 5 4" fill="none" stroke="#c99a74" stroke-width="2.6" stroke-linecap="round"/>
      <path d="M-13 19 Q2 26 15 16" fill="none" stroke="#9b5a4a" stroke-width="3" stroke-linecap="round"/>
    </g>
    <!-- the glasses -->
    $(join(glasses, "\n    "))
  </g>

  <!-- Wordmark and subtitle -->
  <g transform="translate(456 100)">
    <text x="0" y="44" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="41" font-weight="820" fill="#222734" letter-spacing="0">BatchedKolmogorovFlow.jl</text>
    <text x="2" y="84" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">Kolmogorov flow on batches of fields, its tangent</text>
    <text x="2" y="112" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">and its adjoint, on a GPU or on a CPU.</text>
  </g>
</svg>
"""

write(joinpath(@__DIR__, "logo.svg"), svg)
println("written assets/logo.svg")
