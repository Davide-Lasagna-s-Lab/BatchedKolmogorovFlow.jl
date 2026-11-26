# Draws assets/logo.svg, in the manner of the logos of the other packages: a batch of fields, four
# tiles in the colours of Julia, the first with the bands of the flow, and Andrey Kolmogorov, who
# gave the flow its name, in the glasses that the batches deserve.
#
#     julia assets/make_logo.jl
using Printf

const INK    = "#242936"
const COLORS = ["#cb3c33", "#9558b2", "#389826", "#4063d8"]    # from the last tile to the first


# ---- the batch: four tiles, each a little up and to the left of the one before it ----

side  = 138.0
first = (116.0, 108.0)
back  = (-19.0, -15.0)

tiles = [@sprintf("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" rx=\"22\" fill=\"%s\"/>",
                  first[1] + (4 - i)*back[1], first[2] + (4 - i)*back[2], side, side, COLORS[i])
         for i in 1:4]

# the bands of the flow on the first tile: three waves across it
wave(y) = "M" * join([@sprintf("%.1f %.1f", first[1] + 16 + x, y + 8*sin(2π*x/53))
                      for x in range(0, side - 32; length=60)], " L")
bands = [@sprintf("<path d=\"%s\" fill=\"none\" stroke=\"#ffffff\" stroke-opacity=\"0.92\" stroke-width=\"10\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
                  wave(first[2] + side*f)) for f in (0.25, 0.5, 0.75)]


# ---- the glasses: black pixels, and the glint of each lens ----

pixels = ["######################",
          "..#oo#o####..#oo#o####",
          "..##oo#o###..##oo#o###",
          "...##oo#o#....##oo#o#.",
          "....#####......#####.."]
px     = 6.0
corner = (322 - 12px, 120.0)    # the bridge over the nose

glasses = [@sprintf("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"%s\"/>",
                    corner[1] + (c - 1)*px, corner[2] + (r - 1)*px, px + 0.4, px + 0.4,
                    pixels[r][c] == '#' ? "#14161c" : "#ffffff")
           for r in eachindex(pixels) for c in 1:length(pixels[r]) if pixels[r][c] != '.']


# ---- the svg ----

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 300" role="img" aria-labelledby="title desc">
  <title id="title">BatchedKolmogorovFlow.jl logo</title>
  <desc id="desc">A batch of four tiles in the colours of Julia, the first with the bands of Kolmogorov flow, and Andrey Kolmogorov in pixel sunglasses, beside the BatchedKolmogorovFlow.jl wordmark.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <filter id="panel-shadow" x="-12%" y="-25%" width="124%" height="150%">
      <feDropShadow dx="0" dy="9" stdDeviation="11" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#172033" flood-opacity="0.16"/>
    </filter>
    <clipPath id="panel">
      <rect x="24" y="24" width="1032" height="252" rx="24"/>
    </clipPath>
  </defs>

  <!-- Main panel -->
  <rect x="24" y="24" width="1032" height="252" rx="24" fill="url(#bg)" filter="url(#panel-shadow)"/>

  <g clip-path="url(#panel)">
    <!-- The batch -->
    <g filter="url(#soft)">
      $(join(tiles, "\n      "))
    </g>
    $(join(bands, "\n    "))

    <!-- Kolmogorov -->
    <g filter="url(#soft)" stroke-linejoin="round" stroke-linecap="round">
      <path d="M236 300 Q240 226 294 210 L350 210 Q404 226 408 300 Z" fill="$INK"/>
      <path d="M300 208 L322 244 L344 208 Z" fill="#ffffff"/>
      <path d="M314 216 L330 216 L327 230 L334 300 L310 300 L317 230 Z" fill="#cb3c33"/>
      <rect x="306" y="186" width="32" height="26" fill="#e2b48f"/>
      <ellipse cx="270" cy="150" rx="9" ry="14" fill="#f3cda6" stroke="$INK" stroke-width="5"/>
      <ellipse cx="374" cy="150" rx="9" ry="14" fill="#f3cda6" stroke="$INK" stroke-width="5"/>
      <ellipse cx="322" cy="142" rx="50" ry="58" fill="#f3cda6" stroke="$INK" stroke-width="5"/>
      <path d="M270 146 Q254 96 286 74 Q322 52 358 72 Q392 94 374 146 Q372 122 360 112 Q340 126 322 104 Q306 124 284 112 Q272 122 270 146 Z" fill="#f6f7f9" stroke="$INK" stroke-width="5"/>
      <path d="M322 152 Q313 171 324 173" fill="none" stroke="$INK" stroke-width="4"/>
      <path d="M303 186 Q322 196 343 181" fill="none" stroke="$INK" stroke-width="5"/>
    </g>
    $(join(glasses, "\n    "))
  </g>

  <!-- Wordmark and subtitle -->
  <g transform="translate(452 100)">
    <text x="0" y="46" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="42" font-weight="820" fill="#222734" letter-spacing="0">BatchedKolmogorovFlow.jl</text>
    <text x="2" y="86" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">Kolmogorov flow on batches of fields, its tangent</text>
    <text x="2" y="114" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">and its adjoint, on a GPU or on a CPU.</text>
  </g>
</svg>
"""

write(joinpath(@__DIR__, "logo.svg"), svg)
println("written assets/logo.svg")
