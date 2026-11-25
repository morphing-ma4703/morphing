# =====================================================================
#  herramientas.jl — Cargar imágenes propias y generar GIFs del morphing
#  Requiere haber hecho antes:  include("src/morphing.jl")
#  Paquetes:  ] add FFTW Plots Images FileIO ImageIO
# =====================================================================
using Plots
using FileIO
using Images

# ---------------------------------------------------------------------
# 1. Imagen -> densidad
# ---------------------------------------------------------------------

# Lee un PNG/JPG como matriz de grises en [0,1] (1 = blanco).
# Si la imagen tiene transparencia, se compone sobre fondo blanco.
# Devuelve f[x, y] con x hacia la derecha e y hacia arriba (convención del solver,
# así heatmap(f') muestra la imagen derecha).
function cargar_gris(ruta)
    img = load(ruta)
    if eltype(img) <: TransparentColor
        a = Float64.(alpha.(img))
        g = Float64.(Gray.(color.(img)))
        G = a .* g .+ (1 .- a)
    else
        G = Float64.(Gray.(img))
    end
    return reverse(permutedims(G, (2, 1)), dims=2)
end

# Recorta filas/columnas que son solo fondo (deja un 5 % de borde).
function recortar_margenes(G, fondo; tol=0.05)
    hay = abs.(G .- fondo) .> tol
    xs = findall(vec(any(hay, dims=2)))
    ys = findall(vec(any(hay, dims=1)))
    isempty(xs) && return G
    A, H = size(G)
    bx = max(1, round(Int, 0.05 * (xs[end] - xs[1] + 1)))
    by = max(1, round(Int, 0.05 * (ys[end] - ys[1] + 1)))
    return G[max(1, xs[1]-bx):min(A, xs[end]+bx), max(1, ys[1]-by):min(H, ys[end]+by)]
end

# Rellena G (centrada, sin deformarla) hasta que tenga proporción ancho/alto = razon.
function ajustar_proporcion(G, razon, relleno)
    A, H = size(G)                          # f[x, y]: A = ancho, H = alto
    if A / H < razon
        A2, H2 = round(Int, H * razon), H
    else
        A2, H2 = A, round(Int, A / razon)
    end
    out = fill(relleno, A2, H2)
    i0 = (A2 - A) ÷ 2
    j0 = (H2 - H) ÷ 2
    out[i0+1:i0+A, j0+1:j0+H] .= G
    return out
end

# Imagen -> densidad de masa 1, lista para el solver.
#   tam = n        -> grilla cuadrada n×n
#   tam = (N, P)   -> grilla rectangular de N celdas de ancho y P de alto (p. ej. (160, 40)
#                     para una palabra). Ambas imágenes deben usar el mismo tam.
#   La imagen se rellena con fondo hasta la proporción de la grilla: nunca se deforma.
#   invertir=true  : lo OSCURO es masa (dibujo negro sobre fondo blanco)   <- lo habitual
#   invertir=false : lo CLARO es masa (objeto claro sobre fondo oscuro)
#   frac_fondo     : fracción de masa en el fondo (más chico = más contraste, pero más difícil)
#   margen         : celdas de borde sin masa (el método no deja salir masa por el borde)
#   umbral  : número en (0,1) -> pasa la imagen a blanco y negro puro (silueta): todo lo más
#             oscuro que el umbral es tinta. Útil para fotos con el fondo ya quitado
#             (PNG transparente): con umbral=0.95 todo el objeto queda como silueta negra.
#   recortar: true -> recorta los márgenes vacíos para que la figura llene la grilla
#             (no usarlo si la POSICIÓN de la figura importa, p. ej. un disco que se desplaza).
function densidad_desde_imagen(ruta, tam; invertir=true, frac_fondo=0.02, margen=2,
                               umbral=nothing, recortar=false)
    N, P = tam isa Integer ? (tam, tam) : tam
    G = cargar_gris(ruta)
    fondo = invertir ? 1.0 : 0.0
    if umbral !== nothing
        tinta = invertir ? (G .< umbral) : (G .> umbral)
        G = ifelse.(tinta, 1.0 - fondo, fondo)
    end
    recortar && (G = recortar_margenes(G, fondo))
    G = ajustar_proporcion(G, N / P, fondo)
    return a_densidad(G, (N, P); invertir=invertir, frac_fondo=frac_fondo, margen=margen)
end

# ---------------------------------------------------------------------
# 2. Morphing
# ---------------------------------------------------------------------

# Resuelve el transporte óptimo entre f0 y f1.
# Devuelve un arreglo (N, P, Q+1): la densidad en t = 0, 1/Q, ..., 1 (incluye los extremos).
# Si la grilla es rectangular, el dominio físico es [0, N/P] × [0, 1] (celdas cuadradas), para
# que mover masa en horizontal y en vertical cueste lo mismo.
function morphing(f0, f1; Q=24, β=1.0, niter=600, γ=1/230, verbose=100)
    @assert size(f0) == size(f1) "f0 y f1 deben tener el mismo tamaño"
    N, P = size(f0)
    ε = min(minimum(f0), minimum(f1))
    Xu, _, hist = solve_bb(f0, f1, (N, P, Q); β=β, ε=ε, γ=γ, niter=niter, verbose=verbose,
                           lx=N / P, ly=1.0)
    @printf("Listo: ||div|| final = %.2e\n", hist.div[end])
    return Xu.M[3]
end

# Mezcla lineal (1-t) f0 + t f1, mismo formato que morphing() para comparar.
mezcla(f0, f1; Q=24) = blending(f0, f1, Q)

# ---------------------------------------------------------------------
# 3. GIFs
# ---------------------------------------------------------------------

# Colores sugeridos:
#   :viridis                      -> como las figuras del póster
#   cgrad(:grays, rev=true)       -> negro sobre blanco (se ve como el dibujo original)
const COLORES_DIBUJO = cgrad(:grays, rev=true)

# Un cuadro: cada instante se normaliza a su propio máximo (se ve mejor a mitad de camino,
# cuando la masa está más repartida). suavizar=k agranda k veces con interpolación.
function _panel(ρ, colores, titulo; suavizar=4, tam=(400, 400))
    ρ = max.(ρ, 0.0)
    if suavizar > 1
        ρ = imresize(ρ, size(ρ) .* suavizar)
    end
    return heatmap(ρ', c=colores, clims=(0, maximum(ρ)), legend=false, colorbar=false,
                   axis=false, ticks=false, framestyle=:none, aspect_ratio=:equal,
                   title=titulo, titlefontsize=14, size=tam)
end

# GIF con uno o varios paneles lado a lado (todos con el mismo número de instantes).
#   secuencias : lista de arreglos (N,P,K), p. ej. [ρ_to] o [ρ_to, ρ_mezcla]
#   titulos    : lista de títulos, uno por panel ("" para ninguno)
#   ida_y_vuelta=true : el GIF vuelve al inicio (loop suave)
#   pausa      : cuadros quietos al inicio y al final
#   tam        : alto de cada panel en píxeles (el ancho sale de la proporción de la grilla)
#   apilar     : paneles uno encima del otro (por defecto, sí cuando la imagen es apaisada)
function hacer_gif(secuencias, archivo; titulos=fill("", length(secuencias)),
                   fps=12, ida_y_vuelta=true, pausa=6, colores=:viridis, suavizar=4, tam=400,
                   apilar=nothing)
    K = size(secuencias[1], 3)
    N, P = size(secuencias[1], 1), size(secuencias[1], 2)
    ancho = round(Int, tam * N / P)
    apilar === nothing && (apilar = N / P > 1.5)
    orden = vcat(fill(1, pausa), collect(1:K), fill(K, pausa))
    ida_y_vuelta && (orden = vcat(orden, collect(K-1:-1:2)))
    n = length(secuencias)
    anim = Animation()
    for k in orden
        paneles = [_panel(secuencias[i][:, :, k], colores, titulos[i]; suavizar=suavizar,
                          tam=(ancho, tam)) for i in 1:n]
        lay, tot = apilar ? ((n, 1), (ancho, tam * n)) : ((1, n), (ancho * n, tam))
        frame(anim, plot(paneles..., layout=lay, size=tot))
    end
    mkpath(dirname(abspath(archivo)))
    return gif(anim, archivo, fps=fps)
end

# Atajo: un panel solo.
hacer_gif(ρs::AbstractArray{<:Real,3}, archivo; kw...) = hacer_gif([ρs], archivo; kw...)

# Vista rápida de las dos densidades de entrada (para revisar antes de resolver).
function ver_entradas(f0, f1; colores=:viridis, tam=300)
    N, P = size(f0)
    ancho = round(Int, tam * N / P)
    apilar = N / P > 1.5
    plot(_panel(f0, colores, "Inicio"; suavizar=1, tam=(ancho, tam)),
         _panel(f1, colores, "Final"; suavizar=1, tam=(ancho, tam)),
         layout=apilar ? (2, 1) : (1, 2), size=apilar ? (ancho, 2tam) : (2ancho, tam))
end

# ---------------------------------------------------------------------
# 4. Cadenas de transformaciones: A -> B -> C (-> A)
# ---------------------------------------------------------------------

# Carga varias imágenes con los mismos parámetros (mismo tam para todas).
densidades_desde_imagenes(rutas, tam; kw...) = [densidad_desde_imagen(r, tam; kw...) for r in rutas]

# Resuelve un morphing por cada par consecutivo de la lista.
#   cerrar=true : agrega el tramo última -> primera, para un GIF en loop perfecto.
#   kw          : se pasan a morphing() (Q, niter, β, ...).
function morphing_cadena(densidades; cerrar=false, kw...)
    fs = cerrar ? vcat(densidades, [densidades[1]]) : densidades
    tramos = Array{Float64,3}[]
    for i in 1:length(fs)-1
        println("Tramo $i de $(length(fs)-1)")
        push!(tramos, morphing(fs[i], fs[i+1]; kw...))
    end
    return tramos
end

# Une los tramos en una sola secuencia, con una pausa en cada imagen intermedia.
function unir_tramos(tramos; pausa=6, cerrar=false)
    partes = Array{Float64,3}[]
    for ρ in tramos
        push!(partes, repeat(ρ[:, :, 1:1], 1, 1, pausa))   # quieto en la imagen de partida
        push!(partes, ρ[:, :, 1:end-1])                     # el tramo (su último cuadro = inicio del siguiente)
    end
    if !cerrar                                              # si no es loop, pausa en la imagen final
        push!(partes, repeat(tramos[end][:, :, end:end], 1, 1, pausa + 1))
    end
    return cat(partes...; dims=3)
end

# Todo junto: rutas -> GIF.
#   gif_cadena(["imagenes/control.png", "imagenes/optimo.png", "imagenes/corazon.png"],
#              (160, 40), "assets/cadena.gif"; cerrar=true)
function gif_cadena(rutas, tam, archivo; cerrar=true, pausa=6, Q=24, niter=600,
                    invertir=true, umbral=nothing, recortar=false,
                    colores=COLORES_DIBUJO, fps=12, kw...)
    fs = densidades_desde_imagenes(rutas, tam; invertir=invertir, umbral=umbral, recortar=recortar)
    tramos = morphing_cadena(fs; cerrar=cerrar, Q=Q, niter=niter, kw...)
    ρ = unir_tramos(tramos; pausa=pausa, cerrar=cerrar)
    return hacer_gif(ρ, archivo; pausa=0, ida_y_vuelta=false, colores=colores, fps=fps)
end
