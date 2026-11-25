# =====================================================================
#  morphing.jl — Solver de Benamou–Brenier por splitting proximal (PPXA)
#  Extraído de proyecto.ipynb (mismas funciones, sin los tests).
#  Uso:  include("src/morphing.jl")
# =====================================================================
using FFTW
using LinearAlgebra
using Printf
using Random


# Campo escalonado: M[1],M[2]=momento, M[3]=densidad rho (eje temporal).
struct Staggered
    dim
    M
end

function Staggered(dim)
    N, P, Q = dim
    M1 = zeros(N+1, P,   Q  )
    M2 = zeros(N,   P+1, Q  )
    M3 = zeros(N,   P,   Q+1)
    return Staggered(dim, [M1, M2, M3])
end

import Base: +, -, *
function +(A::Staggered, B::Staggered)
    C = Staggered(A.dim)
    for k in 1:3; C.M[k] = A.M[k] + B.M[k]; end
    return C
end
function -(A::Staggered, B::Staggered)
    C = Staggered(A.dim)
    for k in 1:3; C.M[k] = A.M[k] - B.M[k]; end
    return C
end
function *(a::Real, A::Staggered)
    C = Staggered(A.dim)
    for k in 1:3; C.M[k] = a * A.M[k]; end
    return C
end

copia(A::Staggered) = Staggered(A.dim, [copy(A.M[k]) for k in 1:3])

function dotp_stag(X, Y)
    s = 0.0
    for k in 1:3; s = s + sum(X.M[k] .* Y.M[k]); end
    return s
end

function random_staggered(dim)
    U = Staggered(dim)
    for k in 1:3; U.M[k] = randn(size(U.M[k])); end
    return U
end


# Energia cinetica discreta: suma de |m|^2 / (2 rho^beta)  (convencion con 1/2).
function energia_BB(V; ε=1e-10, β=1.0)
    m2  = V[:,:,:,1].^2 .+ V[:,:,:,2].^2
    ρ   = max.(V[:,:,:,3], ε)
    return sum(m2 ./ (2 .* ρ.^β))
end


# interp: escalonada -> centrada (promedio de caras vecinas /2). Devuelve (N,P,Q,3).
function interp(U)
    N, P, Q = U.dim
    V = zeros(N, P, Q, 3)
    V[:,:,:,1] = (U.M[1][1:end-1,:,:] .+ U.M[1][2:end,:,:]) ./ 2
    V[:,:,:,2] = (U.M[2][:,1:end-1,:] .+ U.M[2][:,2:end,:]) ./ 2
    V[:,:,:,3] = (U.M[3][:,:,1:end-1] .+ U.M[3][:,:,2:end]) ./ 2
    return V
end

# interp_adj: adjunto de interp. Cumple <U, interp_adj(V)> = <interp(U), V>.
function interp_adj(V)
    N, P, Q = size(V,1), size(V,2), size(V,3)
    U = Staggered((N,P,Q))
    U.M[1][1,:,:]   = V[1,:,:,1]
    U.M[1][2:N,:,:] = V[1:N-1,:,:,1] .+ V[2:N,:,:,1]
    U.M[1][N+1,:,:] = V[N,:,:,1]
    U.M[1] ./= 2
    U.M[2][:,1,:]   = V[:,1,:,2]
    U.M[2][:,2:P,:] = V[:,1:P-1,:,2] .+ V[:,2:P,:,2]
    U.M[2][:,P+1,:] = V[:,P,:,2]
    U.M[2] ./= 2
    U.M[3][:,:,1]   = V[:,:,1,3]
    U.M[3][:,:,2:Q] = V[:,:,1:Q-1,3] .+ V[:,:,2:Q,3]
    U.M[3][:,:,Q+1] = V[:,:,Q,3]
    U.M[3] ./= 2
    return U
end

# div: divergencia espacio-temporal  d_t rho + div m.
function div_stag(U; lx=1.0, ly=1.0, lt=1.0)
    N, P, Q = U.dim
    v  = diff(U.M[1], dims=1) .* (N/lx)
    v  = v .+ diff(U.M[2], dims=2) .* (P/ly)
    v  = v .+ diff(U.M[3], dims=3) .* (Q/lt)
    return v
end


# prox_energia: prox de la energia, punto a punto. Convencion con 1/2.
function prox_energia(V, γ; ε=1e-8, β=1.0)
    mx = V[:,:,:,1]
    my = V[:,:,:,2]
    ρ0 = V[:,:,:,3]
    m02 = mx.^2 .+ my.^2
    if β == 0.0
        ρ = max.(ρ0, ε)
    elseif β == 1.0
        # Newton sobre P(ρ)=(ρ-ρ0)(ρ+γ)^2-(γ/2)m02, inicial a la derecha de la raiz
        ρ = ρ0 .+ m02 ./ (2γ) .+ ε
        for _ in 1:60
            Pr  = (ρ .- ρ0) .* (ρ .+ γ).^2 .- (γ/2) .* m02
            dPr = (ρ .+ γ) .* (3 .* ρ .+ γ .- 2 .* ρ0)
            ρ   = ρ .- Pr ./ dPr
            ρ   = max.(ρ, ε)
        end
    else
        # Newton sobre P_beta (derivada numerica)
        ρ = max.(ρ0, ε)
        for _ in 1:80
            rb  = ρ .^ β
            Pr  = ρ.^(1-β) .* (ρ .- ρ0) .* (rb .+ γ).^2 .- (γ*β/2) .* m02
            h   = 1e-7 .* max.(ρ, 1.0)
            ρ2  = ρ .+ h
            rb2 = ρ2 .^ β
            Pr2 = ρ2.^(1-β) .* (ρ2 .- ρ0) .* (rb2 .+ γ).^2 .- (γ*β/2) .* m02
            dPr = (Pr2 .- Pr) ./ h
            ρ   = ρ .- Pr ./ dPr
            ρ   = max.(ρ, ε)
        end
    end
    rb = ρ .^ β
    out = similar(V)
    out[:,:,:,1] = (rb ./ (rb .+ γ)) .* mx
    out[:,:,:,2] = (rb ./ (rb .+ γ)) .* my
    out[:,:,:,3] = ρ
    return out
end


# Poisson 2D Neumann por DCT. Resuelve -Lap(u)=f, devuelve u de media nula.
function poisson_neumann_2d(f; lx=1.0, ly=1.0)
    N, M = size(f)
    hx = lx/N; hy = ly/M
    λx = (2 .* cos.(π .* (0:N-1) ./ N) .- 2) ./ hx^2
    λy = (2 .* cos.(π .* (0:M-1) ./ M) .- 2) ./ hy^2
    denom = zeros(N, M)
    for i in 1:N, j in 1:M
        denom[i,j] = λx[i] + λy[j]
    end
    denom[1,1] = 1.0
    uhat = -dct(f) ./ denom
    uhat[1,1] = 0.0
    return idct(uhat)
end

# Poisson 3D Neumann (x,y,t). Mismo metodo.
function poisson_neumann_3d(f; lx=1.0, ly=1.0, lt=1.0)
    N, M, Q = size(f)
    hx = lx/N; hy = ly/M; ht = lt/Q
    λx = (2 .* cos.(π .* (0:N-1) ./ N) .- 2) ./ hx^2
    λy = (2 .* cos.(π .* (0:M-1) ./ M) .- 2) ./ hy^2
    λt = (2 .* cos.(π .* (0:Q-1) ./ Q) .- 2) ./ ht^2
    denom = zeros(N, M, Q)
    for i in 1:N, j in 1:M, k in 1:Q
        denom[i,j,k] = λx[i] + λy[j] + λt[k]
    end
    denom[1,1,1] = 1.0
    uhat = -dct(f) ./ denom
    uhat[1,1,1] = 0.0
    return idct(uhat)
end


# div_proj: proyeccion sobre {div U = 0}. Corrige solo caras interiores.
function div_proj(U; lx=1.0, ly=1.0, lt=1.0)
    N, P, Q = U.dim
    du = div_stag(U; lx=lx, ly=ly, lt=lt)
    p  = poisson_neumann_3d(-du; lx=lx, ly=ly, lt=lt)
    v  = copia(U)
    v.M[1][2:end-1,:,:] .-= diff(p, dims=1) .* (N/lx)
    v.M[2][:,2:end-1,:] .-= diff(p, dims=2) .* (P/ly)
    v.M[3][:,:,2:end-1] .-= diff(p, dims=3) .* (Q/lt)
    return v
end


# Matrices de proyeccion para una linea de longitud centrada n.
# Restriccion: v = S u (interpolacion), u(1)=a, u(end)=b.
function proj_ops(n)
    S = zeros(n, n+1)
    for i in 1:n
        S[i, i]   = 0.5
        S[i, i+1] = 0.5
    end
    A = zeros(n+2, 2n+1)
    A[1:n, 1:n+1]      = S
    A[1:n, n+2:2n+1]   = -Matrix(I, n, n)
    A[n+1, 1]          = 1.0      # u(1)=a
    A[n+2, n+1]        = 1.0      # u(end)=b
    pA = pinv(A)
    B  = Matrix(I, 2n+1, 2n+1) - pA * A
    return pA, B
end

# interp_proj_lento: versión original (referencia). Proyecta (U,V) sobre {V=interp(U)} con rho(.,0)=f0, rho(.,1)=f1.
function interp_proj_lento(U, V, f0, f1)
    N, P, Q = U.dim
    U = copia(U); V = copy(V)
    # --- eje temporal (k=3): a=f0, b=f1 ---
    pA, B = proj_ops(Q)
    for i in 1:N, j in 1:P
        u0 = U.M[3][i,j,:]
        v0 = V[i,j,:,3]
        g = zeros(Q+2); g[Q+1] = f0[i,j]; g[Q+2] = f1[i,j]
        uv = B * vcat(u0, v0) + pA * g
        U.M[3][i,j,:] = uv[1:Q+1]
        V[i,j,:,3]    = uv[Q+2:end]
    end
    # --- eje x (k=1): no-flujo, a=b=0 ---
    pA, B = proj_ops(N)
    g = zeros(N+2)
    for j in 1:P, t in 1:Q
        u0 = U.M[1][:,j,t]
        v0 = V[:,j,t,1]
        uv = B * vcat(u0, v0) + pA * g
        U.M[1][:,j,t] = uv[1:N+1]
        V[:,j,t,1]    = uv[N+2:end]
    end
    # --- eje y (k=2): no-flujo, a=b=0 ---
    pA, B = proj_ops(P)
    g = zeros(P+2)
    for i in 1:N, t in 1:Q
        u0 = U.M[2][i,:,t]
        v0 = V[i,:,t,2]
        uv = B * vcat(u0, v0) + pA * g
        U.M[2][i,:,t] = uv[1:P+1]
        V[i,:,t,2]    = uv[P+2:end]
    end
    return U, V
end


# interp_proj vectorizado (batch por eje). Reemplazo directo de interp_proj.
function interp_proj_vec(U, V, f0, f1)
    N, P, Q = U.dim
    U = copia(U); V = copy(V)

    # --- eje temporal (k=3): datos a=f0 (t=0), b=f1 (t=1) ---
    pA, B = proj_ops(Q)
    Um = reshape(permutedims(U.M[3], (3,1,2)), Q+1, N*P)
    Vm = reshape(permutedims(V[:,:,:,3], (3,1,2)), Q,   N*P)
    G  = zeros(Q+2, N*P)
    G[Q+1, :] .= vec(f0); G[Q+2, :] .= vec(f1)
    UV = B * vcat(Um, Vm) .+ pA * G
    U.M[3]     = permutedims(reshape(UV[1:Q+1, :],   Q+1, N, P), (2,3,1))
    V[:,:,:,3] = permutedims(reshape(UV[Q+2:end, :], Q,   N, P), (2,3,1))

    # --- eje x (k=1): no-flujo (a=b=0); el eje 1 ya está al frente ---
    pA, B = proj_ops(N)
    Um = reshape(U.M[1],     N+1, P*Q)
    Vm = reshape(V[:,:,:,1], N,   P*Q)
    UV = B * vcat(Um, Vm)
    U.M[1]     = reshape(UV[1:N+1, :],   N+1, P, Q)
    V[:,:,:,1] = reshape(UV[N+2:end, :], N,   P, Q)

    # --- eje y (k=2): no-flujo (a=b=0) ---
    pA, B = proj_ops(P)
    Um = reshape(permutedims(U.M[2], (2,1,3)), P+1, N*Q)
    Vm = reshape(permutedims(V[:,:,:,2], (2,1,3)), P,   N*Q)
    UV = B * vcat(Um, Vm)
    U.M[2]     = permutedims(reshape(UV[1:P+1, :],   P+1, N, Q), (2,1,3))
    V[:,:,:,2] = permutedims(reshape(UV[P+2:end, :], P,   N, Q), (2,1,3))

    return U, V
end

# solve_bb usa la versión rápida (equivalente a interp_proj_lento, ver test §3.6 del notebook).
interp_proj(U, V, f0, f1) = interp_proj_vec(U, V, f0, f1)


# solve_bb: resuelve el problema BB por PPXA. Devuelve Xu, V=interp(Xu) e historiales.
# Rendimiento: las actualizaciones PPXA se hacen IN-PLACE (@. sobre los arreglos, con buffers
# reutilizados) para no crear un Staggered nuevo por operación. El resultado es idéntico; evita
# asignar decenas de GiB y pasar el tiempo en recolección de basura. El registro se muestrea
# cada `log_every` iteraciones (hist.J no se usa en figuras, así que ese Poisson extra es barato).
# lx, ly: tamaño físico del dominio (por defecto el cuadrado unitario). Para imágenes
# rectangulares N×P conviene lx = N/P, ly = 1, así las celdas son cuadradas.
function solve_bb(f0, f1, dim; β=1.0, ε=1e-6, μ=1.98, γ=1/230, niter=1000, verbose=0, log_every=10, lx=1.0, ly=1.0)
    N, P, Q = dim
    # inicializacion: blending lineal en M[3]
    Xu = Staggered(dim)
    ts = range(0, 1; length=Q+1)
    for i in 1:(Q+1)
        Xu.M[3][:,:,i] = (1-ts[i]) .* f0 .+ ts[i] .* f1
    end
    Xv = interp(Xu)
    ω = [1/3, 1/3, 1/3]
    Yu = [copia(Xu) for _ in 1:3]
    Yv = [copy(Xv) for _ in 1:3]
    Jhist = Float64[]; divhist = Float64[]; minhist = Float64[]
    Zu = Vector{Any}(undef, 3); Zv = Vector{Any}(undef, 3)
    Zu_s = Staggered(dim); Zv_s = zeros(N, P, Q, 3)   # buffers del promedio (reutilizados)
    for it in 1:niter
        # prox 1: energia (toca V) ; prox 2: continuidad (toca U) ; prox 3: consenso
        Zu[1] = Yu[1];              Zv[1] = prox_energia(Yv[1], γ/ω[1]; ε=ε, β=β)
        Zu[2] = div_proj(Yu[2]; lx=lx, ly=ly);    Zv[2] = Yv[2]
        Zu[3], Zv[3] = interp_proj(Yu[3], Yv[3], f0, f1)

        # promedio Z_s = sum_k omega_k Z_k  (acumulado in-place en los buffers)
        for c in 1:3; Zu_s.M[c] .= 0.0; end
        Zv_s .= 0.0
        for k in 1:3
            wk = ω[k]
            for c in 1:3
                zs = Zu_s.M[c]; zk = Zu[k].M[c]
                @. zs = zs + wk * zk
            end
            zvk = Zv[k]
            @. Zv_s = Zv_s + wk * zvk
        end

        # actualizaciones PPXA (in-place): Y_k += mu (2 Z_s - X - Z_k)
        for k in 1:3
            for c in 1:3
                yk = Yu[k].M[c]; zk = Zu[k].M[c]; zs = Zu_s.M[c]; xu = Xu.M[c]
                @. yk = yk + μ * (2*zs - xu - zk)
            end
            yv = Yv[k]; zv = Zv[k]
            @. yv = yv + μ * (2*Zv_s - Xv - zv)
        end

        # actualizacion de X (in-place): X += mu (Z_s - X)
        for c in 1:3
            xu = Xu.M[c]; zs = Zu_s.M[c]
            @. xu = xu + μ * (zs - xu)
        end
        @. Xv = Xv + μ * (Zv_s - Xv)

        # registro: solo cada log_every iteraciones (mas la primera y la ultima)
        if it == 1 || it == niter || it % log_every == 0
            Vc = interp(div_proj(Xu; lx=lx, ly=ly))
            push!(Jhist, energia_BB(Vc; ε=1e-3, β=β))
            push!(divhist, maximum(abs.(div_stag(Xu; lx=lx, ly=ly))))
            push!(minhist, minimum(interp(Xu)[:,:,:,3]))
            if verbose > 0
                @printf("    it=%4d/%d  ||div||=%.2e  minρ=%.2e\n", it, niter, divhist[end], minhist[end])
            end
        end
    end
    return Xu, interp(Xu), (J=Jhist, div=divhist, minρ=minhist)
end


# gaussiana normalizada a masa 1.
function gaussiana(dim2d, centro, σ)
    N, P = dim2d
    cx, cy = centro
    x = range(0, 1; length=N)
    y = range(0, 1; length=P)
    G = [exp(-((xi-cx)^2 + (yj-cy)^2)/(2σ^2)) for xi in x, yj in y]
    return G ./ sum(G)
end

# blending lineal rho_t=(1-t)f0+t f1.
function blending(f0, f1, Q)
    N, P = size(f0)
    out = zeros(N, P, Q+1)
    ts = range(0, 1; length=Q+1)
    for i in 1:(Q+1)
        out[:,:,i] = (1-ts[i]) .* f0 .+ ts[i] .* f1
    end
    return out
end

# dispersion espacial de una densidad 2D (varianza respecto al centro de masa).
function dispersion(ρ2d)
    N, P = size(ρ2d)
    x = range(0,1;length=N); y = range(0,1;length=P)
    m = sum(ρ2d)
    cx = sum([x[i]*ρ2d[i,j] for i in 1:N, j in 1:P]) / m
    cy = sum([y[j]*ρ2d[i,j] for i in 1:N, j in 1:P]) / m
    v  = sum([((x[i]-cx)^2+(y[j]-cy)^2)*ρ2d[i,j] for i in 1:N, j in 1:P]) / m
    return v, (cx, cy)
end


# derivada centrada a lo largo de un eje (interior centrado, bordes unilateral).
function deriv(f, h, dim)
    g = zeros(size(f))
    if dim == 1
        g[2:end-1,:,:] = (f[3:end,:,:] .- f[1:end-2,:,:]) ./ (2h)
        g[1,:,:]       = (f[2,:,:]   .- f[1,:,:])     ./ h
        g[end,:,:]     = (f[end,:,:] .- f[end-1,:,:]) ./ h
    elseif dim == 2
        g[:,2:end-1,:] = (f[:,3:end,:] .- f[:,1:end-2,:]) ./ (2h)
        g[:,1,:]       = (f[:,2,:]   .- f[:,1,:])     ./ h
        g[:,end,:]     = (f[:,end,:] .- f[:,end-1,:]) ./ h
    else
        g[:,:,2:end-1] = (f[:,:,3:end] .- f[:,:,1:end-2]) ./ (2h)
        g[:,:,1]       = (f[:,:,2]   .- f[:,:,1])     ./ h
        g[:,:,end]     = (f[:,:,end] .- f[:,:,end-1]) ./ h
    end
    return g
end

# residuos de optimalidad, medidos sobre la velocidad v = m/rho.
function residuos_optimalidad(V)
    N, P, Q = size(V,1), size(V,2), size(V,3)
    mx = V[:,:,:,1]; my = V[:,:,:,2]; ρ = V[:,:,:,3]
    εv = 1e-2 * maximum(ρ)              # regularizacion moderada: en el vacio v~0
    vx = mx ./ (ρ .+ εv)
    vy = my ./ (ρ .+ εv)
    hx = 1/N; hy = 1/P; ht = 1/Q
    masa = sum(max.(ρ, 0.0))
    # flujo potencial: rot(v) = d vy/dx - d vx/dy debe ser 0
    rot = deriv(vy, hx, 1) .- deriv(vx, hy, 2)
    Rv  = sqrt(sum(max.(ρ,0.0) .* rot.^2) / masa)
    # Hamilton-Jacobi: d_t v + 1/2 grad|v|^2 = 0
    v2  = vx.^2 .+ vy.^2
    hjx = deriv(vx, ht, 3) .+ 0.5 .* deriv(v2, hx, 1)
    hjy = deriv(vy, ht, 3) .+ 0.5 .* deriv(v2, hy, 2)
    RHJ = sqrt(sum(max.(ρ,0.0) .* (hjx.^2 .+ hjy.^2)) / masa)
    return Rv, RHJ
end


# --- Redimensión por promedio de área (separable) ---
# Matriz de remuestreo 1D por solapamiento de intervalos: nin -> nout. Cada fila suma 1 (promedio).
function remuestreo_1d(nin, nout)
    W = zeros(nout, nin)
    for o in 1:nout
        a = (o-1)/nout; b = o/nout                # intervalo de salida [a,b]
        for i in 1:nin
            c = (i-1)/nin; d = i/nin              # intervalo de entrada [c,d]
            W[o,i] = max(0.0, min(b,d) - max(a,c)) # longitud de solapamiento
        end
        W[o,:] ./= sum(W[o,:])                     # normaliza a promedio
    end
    return W
end

# Redimensiona una matriz (H,Wd) -> (N,P) por promedio de área.
function redimensionar(A, dim2d)
    N, P = dim2d
    H, Wd = size(A)
    return remuestreo_1d(H, N) * A * remuestreo_1d(Wd, P)'
end

# --- Formas sintéticas de prueba (intensidad en [0,1], 1 = tinta) ---
function forma_disco(dim2d, centro, r)
    N, P = dim2d; cx, cy = centro
    x = range(0,1;length=N); y = range(0,1;length=P)
    return Float64[ (xi-cx)^2 + (yj-cy)^2 <= r^2 ? 1.0 : 0.0 for xi in x, yj in y]
end

function forma_cuadrado(dim2d, centro, lado)
    N, P = dim2d; cx, cy = centro
    x = range(0,1;length=N); y = range(0,1;length=P)
    return Float64[ (abs(xi-cx) <= lado/2 && abs(yj-cy) <= lado/2) ? 1.0 : 0.0 for xi in x, yj in y]
end

# --- Conversión intensidad -> densidad válida (masa 1, rho>0) ---
#   invertir    : true  => tinta oscura (valores bajos) pasa a masa alta (PNG típico, fondo blanco)
#                 false => los valores altos ya son masa (formas sintéticas con tinta=1)
#   frac_fondo  : fracción de la masa total asignada al fondo (controla contraste y rho>0)
#   margen      : nº de celdas de borde forzadas a fondo (deja la masa lejos de la frontera)
function a_densidad(raw, dim2d; invertir=true, frac_fondo=0.02, margen=1)
    N, P = dim2d
    f = redimensionar(Float64.(raw), dim2d)
    f .-= minimum(f); mx = maximum(f); mx > 0 && (f ./= mx)   # intensidad a [0,1]
    invertir && (f .= 1 .- f)                                 # tinta -> masa
    if margen > 0                                             # margen de fondo
        f[1:margen,:] .= 0; f[end-margen+1:end,:] .= 0
        f[:,1:margen] .= 0; f[:,end-margen+1:end] .= 0
    end
    s = sum(f)                                                # piso por fracción de masa de fondo:
    c = frac_fondo*s / (N*P*(1-frac_fondo))                   #   frac ≈ c*N*P/(s + c*N*P)
    f .+= c
    f ./= sum(f)                                              # masa 1
    return f
end
