"""
STS-Vision — processar_video_sts_v23.py   (versão do Artigo 3, congelada)
=========================================================================
Cronometragem automática do Supine-to-Stand (STS) a partir de vídeo.
INÍCIO: recuo até 5 quadros parados (2º maior movimento de 13 pontos / ruído < 6).
FIM: velocidade vertical do ombro do lado visível < 20% do pico, acima de 80%
da altura da tentativa. Só as 3 primeiras subidas valem.
Convenção de tempo: quadro k (0, 1, 2...) -> t = (k + 1) / fps.
=========================================================================
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from scipy.ndimage import uniform_filter1d

VERSAO = "v23"

# ----------------------------- Parâmetros -----------------------------
PCT_NIVEL_DEITADO = 90        # percentil alto de ombro.y = nível deitado
PCT_NIVEL_EM_PE = 5           # percentil baixo de ombro.y = nível em pé
LIMIAR_LEVANTOU = 0.5         # h acima disto marca a subida
LIMIAR_DEITOU = 0.2           # h abaixo disto rearma (voltou a deitar)

PONTOS_MOVIMENTO = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
SUAV_VELOCIDADE = 3           # quadros de média móvel na velocidade
PCT_RUIDO = 20                # percentil da velocidade = ruído de repouso
LIMIAR_MOVIMENTO = 6.0        # "parado" = 2º ponto mais rápido < 6x seu ruído
QUADROS_PARADO = 5            # quadros seguidos parados que encerram o recuo
RECUO_MAX_S = 3.0             # recua no máximo 3 s antes de deitar
AJUSTE_INICIO_QUADROS = 0     # v23: sem ajuste

SUAV_ALTURA = 7               # quadros de média móvel na altura do ombro
FRACAO_VEL_FIM = 0.2          # fim quando a velocidade cai abaixo de 20% do pico
FRACAO_ALTURA_FIM = 0.8       # ... e já passou 80% da altura da tentativa

MAX_TENTATIVAS = 3            # protocolo: só as 3 primeiras subidas valem

LADO_ESQ = (11, 13, 15, 23, 25, 27)
LADO_DIR = (12, 14, 16, 24, 26, 28)
OMBRO = {"LEFT": 11, "RIGHT": 12}


# ----------------------------- Rastreio -----------------------------
def rastrear_33_pontos(caminho_video):
    """pts (n_quadros, 33, 4: x, y, z, visibilidade), fps, largura, altura.
    Usa a variável global `options` (criada na célula de setup)."""
    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(caminho_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vazio = [(np.nan, np.nan, np.nan, np.nan)] * 33
    pts, i, ultimo = [], 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        ts = max(int(round(i / fps * 1000)), ultimo + 1)
        ultimo = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(img, ts)
        pts.append([(l.x, l.y, l.z, l.visibility or 0.0) for l in res.pose_landmarks[0]]
                   if res.pose_landmarks else vazio)
    cap.release()
    detector.close()
    return np.array(pts, dtype=np.float32), fps, largura, altura


# ----------------------------- Sinais -----------------------------
def _interpolar_nan(a):
    """Preenche NaN por interpolação linear, coluna por coluna."""
    a = np.array(a, dtype=float)
    plano = a.reshape(len(a), -1)
    idx = np.arange(len(a))
    for j in range(plano.shape[1]):
        col = plano[:, j]
        nan = np.isnan(col)
        if nan.any() and (~nan).any():
            col[nan] = np.interp(idx[nan], idx[~nan], col[~nan])
    return plano.reshape(a.shape)


def preparar_pontos(pts, largura, altura):
    """Coordenadas (n, 33, 2) sem NaN, com x na mesma escala de y, e o lado
    do corpo mais visível."""
    xy = _interpolar_nan(pts[:, :, :2])
    xy[:, :, 0] *= largura / altura
    vis_esq = np.nanmean(pts[:, LADO_ESQ, 3])
    vis_dir = np.nanmean(pts[:, LADO_DIR, 3])
    lado = "LEFT" if vis_esq >= vis_dir else "RIGHT"
    return xy, lado


def movimento_normalizado(xy):
    """Velocidade de cada ponto dividida pelo seu ruído de repouso; devolve,
    por quadro, o 2º maior valor (robusto a um ponto isolado com ruído)."""
    p = xy[:, PONTOS_MOVIMENTO, :]
    vel = np.linalg.norm(np.diff(p, axis=0), axis=2)
    vel = np.vstack([vel[:1], vel])
    vel = uniform_filter1d(vel, SUAV_VELOCIDADE, axis=0)
    ruido = np.percentile(vel, PCT_RUIDO, axis=0) + 1e-9
    return np.sort(vel / ruido, axis=1)[:, -2]


def altura_normalizada(ombro_y):
    deitado = np.percentile(ombro_y, PCT_NIVEL_DEITADO)
    em_pe = np.percentile(ombro_y, PCT_NIVEL_EM_PE)
    return (deitado - ombro_y) / max(deitado - em_pe, 1e-6)


# ----------------------------- Segmentação -----------------------------
def segmentar_subidas(h):
    """[(idx_deitou, idx_cruzou, idx_limite)] por tentativa."""
    eventos, armado, idx_deitou = [], False, None
    for k, v in enumerate(h):
        if not armado and v < LIMIAR_DEITOU:
            armado, idx_deitou = True, k
        elif armado and v >= LIMIAR_LEVANTOU:
            eventos.append([idx_deitou, k, None])
            armado = False
    for j, ev in enumerate(eventos):
        ev[2] = eventos[j + 1][0] if j + 1 < len(eventos) else len(h)
    return [tuple(e) for e in eventos]


def detectar_inicio(mov, idx_cruzou, idx_min):
    """Recua a partir da subida até QUADROS_PARADO quadros seguidos parados."""
    parados = 0
    for k in range(idx_cruzou, idx_min - 1, -1):
        if mov[k] < LIMIAR_MOVIMENTO:
            parados += 1
            if parados >= QUADROS_PARADO:
                return k + QUADROS_PARADO
        else:
            parados = 0
    return idx_min


def detectar_fim(altura_ombros, idx_ini, idx_cruzou, idx_limite):
    """Fim da subida: velocidade vertical dos ombros cai abaixo de
    FRACAO_VEL_FIM do pico, já acima de FRACAO_ALTURA_FIM da altura."""
    s = altura_ombros
    k_topo = idx_cruzou + int(np.argmax(s[idx_cruzou:idx_limite]))
    k_baixo = idx_ini + int(np.argmin(s[idx_ini:k_topo + 1]))
    vel = np.gradient(s)
    k_pico = k_baixo + int(np.argmax(vel[k_baixo:k_topo + 1]))
    alvo = s[k_baixo] + FRACAO_ALTURA_FIM * (s[k_topo] - s[k_baixo])
    for k in range(k_pico, idx_limite):
        if vel[k] < FRACAO_VEL_FIM * vel[k_pico] and s[k] >= alvo:
            return k
    return k_topo


# ----------------------------- Pipeline -----------------------------
def processar_pontos_v23(pts, fps, largura, altura, nome="", debug=False):
    """Núcleo da v23: recebe os 33 pontos já rastreados."""
    xy, lado = preparar_pontos(pts, largura, altura)
    h = altura_normalizada(xy[:, OMBRO[lado], 1])
    altura_ombros = uniform_filter1d(-xy[:, OMBRO[lado], 1], SUAV_ALTURA)
    mov = movimento_normalizado(xy)
    eventos = segmentar_subidas(h)
    recuo = int(RECUO_MAX_S * fps)

    print(f"📹 {nome} | FPS={fps:.2f} | Lado: {lado} | {len(eventos)} subida(s)")
    if len(eventos) > MAX_TENTATIVAS:
        print(f"   ⚠️ {len(eventos)} subidas; pelo protocolo só as {MAX_TENTATIVAS} "
              f"primeiras são consideradas.")
        eventos = eventos[:MAX_TENTATIVAS]
    tentativas, fim_anterior = [], -1
    for j, (idx_deitou, idx_cruzou, idx_limite) in enumerate(eventos):
        idx_min = max(idx_deitou - recuo, fim_anterior + 1, 0)
        q_ini = detectar_inicio(mov, idx_cruzou, idx_min)
        q_fim = detectar_fim(altura_ombros, q_ini, idx_cruzou, idx_limite)
        fim_anterior = q_fim
        if q_fim <= q_ini:
            if debug:
                print(f"   (subida {j + 1}: fim antes do início, descartada)")
            continue
        q_ini = max(q_ini - AJUSTE_INICIO_QUADROS, 0)
        delta_t = (q_fim - q_ini) / fps
        tentativas.append({
            "numero": len(tentativas) + 1, "delta_t": delta_t,
            "quadro_inicio": q_ini + 1, "quadro_fim": q_fim + 1,
            "tempo_inicio": (q_ini + 1) / fps, "tempo_fim": (q_fim + 1) / fps,
            "lado_usado": lado,
        })
        print(f"🚀🏁 Tentativa {len(tentativas)}: início={(q_ini + 1) / fps:.2f}s "
              f"fim={(q_fim + 1) / fps:.2f}s Δt={delta_t:.3f}s")

    melhor = min((t["delta_t"] for t in tentativas), default=None)
    if melhor is None:
        print("⚠️ Nenhuma tentativa válida encontrada.")
    else:
        print(f"📊 Melhor tempo: {melhor:.3f}s (de {len(tentativas)} tentativa(s))")
    return {"video": nome, "tentativas": tentativas, "melhor_tempo": melhor,
            "lado_usado": lado, "fps": fps, "versao": VERSAO}


def processar_video_sts_v23(caminho_video, debug=False, pasta_cache=None):
    """Roda a v23 num vídeo. Se `pasta_cache` tiver o ID{...}.npz do vídeo,
    reaproveita o rastreio; senão, rastreia e salva no cache."""
    nome = os.path.basename(caminho_video)
    cache = None
    if pasta_cache:
        cache = os.path.join(pasta_cache, nome.split("_")[0] + ".npz")
    if cache and os.path.exists(cache):
        d = np.load(cache)
        pts, fps = d["pts"], float(d["fps"])
        largura, altura = int(d["largura"]), int(d["altura"])
    else:
        pts, fps, largura, altura = rastrear_33_pontos(caminho_video)
        if cache:
            np.savez_compressed(cache, pts=pts, fps=fps, largura=largura,
                                altura=altura, video=nome, mediapipe=mp.__version__)
    return processar_pontos_v23(pts, fps, largura, altura, nome, debug)


print("✅ Função v23 pronta")
