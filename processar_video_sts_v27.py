"""
STS-Vision — processar_video_sts_v27.py   (VERSÃO CONGELADA)
=========================================================================
Cronometragem automática do Supine-to-Stand (STS) a partir de vídeo.

Base: v26 SEM o fim pelo joelho (o fim volta a ser o da v24/v25: média dos 2
ombros, 15% do pico). Mantém da v26: descarte de subida que não chega em pé
(< 75%) e de duração fora de 0,9–8 s; 3 primeiras subidas válidas; suavização
das posições quando os pontos tremem.

O que a v27 muda (início):
  "Parado" só conta enquanto a pessoa ainda está DEITADA (altura normalizada
  do ombro < 0,2, o mesmo nível que já rearma a segmentação), e o nível de
  repouso local é medido só nesses quadros. Basta 0,1 s (3 quadros a 30 q/s)
  abaixo do limiar. Motivo: na v25/v26, uma pausa no meio da subida (sentado
  ou ajoelhado, altura ~0,4) contava como "repouso", e o início saía tarde
  (até 1,4 s), encurtando a tentativa — e ela virava o melhor tempo.

Origem (transparência): a regra nasceu do diagnóstico PÓS-HOC do 2º teste
independente (vídeo 1653, T1). O número de quadros (3) foi escolhido nos 41
vídeos anteriores (2–4 dão o mesmo resultado; 5 falha no 1653). Resultado no
2º teste é pós-hoc; precisa de nova amostra para ser independente.

Convenção de tempo: quadro k (0, 1, 2...) -> t = (k + 1) / fps, igual ao
anotador do gabarito. Δt = (quadro_fim - quadro_inicio) / fps.

Uso no Colab (depois da célula de setup, que cria `options`):
    resultado = processar_video_sts_v27(caminho_video, pasta_cache=...)
=========================================================================
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from scipy.ndimage import uniform_filter1d

VERSAO = "v27"

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
AJUSTE_INICIO_QUADROS = 1     # v24: início 1 quadro antes do detectado
PCT_REPOUSO_LOCAL = 10        # v25: nível de repouso = percentil 10 do movimento no descanso
FATOR_REPOUSO_LOCAL = 2.0     # v25: "parado" = abaixo de max(6, 2 x repouso local)

SUAV_ALTURA = 7               # quadros de média móvel na altura dos ombros
FRACAO_VEL_FIM = 0.15         # v24: fim quando a velocidade cai abaixo de 15% do pico
FRACAO_ALTURA_FIM = 0.8       # ... e já passou 80% da altura da tentativa
QUADROS_PARADO_DEITADO = 3     # v27: 0,1 s parado, ainda deitado, encerra o recuo
H_EM_PE = 0.75                # v26: a subida só vale se a altura chegar a 75% (em pé)
DURACAO_MIN_S = 0.9           # v26: durações fora de 0,9–8 s são descartadas
DURACAO_MAX_S = 8.0
LIMIAR_TREMOR = 0.0010        # v26: tremor dos pontos parados acima disto -> suaviza posições
SUAV_POSICAO_TREMOR_S = 0.2   # v26: média centrada de 0,2 s nas posições quando há tremor

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


def medir_tremor(xy):
    """v26: deslocamento típico por quadro dos 13 pontos no trecho mais quieto do vídeo
    (mediana entre pontos do percentil 20); mesma escala do app."""
    p = xy[:, PONTOS_MOVIMENTO, :]
    vel = np.linalg.norm(np.diff(p, axis=0), axis=2)
    vel = uniform_filter1d(vel, SUAV_VELOCIDADE, axis=0)
    return float(np.median(np.percentile(vel, PCT_RUIDO, axis=0)))


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


def detectar_inicio(mov, h, idx_cruzou, idx_min):
    """Recua a partir da subida até QUADROS_PARADO_DEITADO quadros seguidos
    parados COM A PESSOA AINDA DEITADA (h < LIMIAR_DEITOU).
    v25: o limiar de "parado" se adapta ao repouso LOCAL; sem trecho parado,
    o início é o momento mais quieto do descanso.
    v27: pausas no meio da subida (sentado/ajoelhado) não contam como repouso,
    e o repouso local é medido só nos quadros deitados."""
    n = QUADROS_PARADO_DEITADO
    trecho = mov[idx_min:idx_cruzou + 1]
    deitado = h[idx_min:idx_cruzou + 1] < LIMIAR_DEITOU
    base = trecho[deitado] if deitado.sum() >= n else trecho
    limiar = max(LIMIAR_MOVIMENTO,
                 FATOR_REPOUSO_LOCAL * np.percentile(base, PCT_REPOUSO_LOCAL))
    parados = 0
    for k in range(idx_cruzou, idx_min - 1, -1):
        if mov[k] < limiar and h[k] < LIMIAR_DEITOU:
            parados += 1
            if parados >= n:
                return k + n
        else:
            parados = 0
    suave = uniform_filter1d(np.where(deitado, trecho, np.inf), n)
    if np.isfinite(suave).any():
        return idx_min + int(np.argmin(suave))
    return idx_min + int(np.argmin(uniform_filter1d(trecho, n)))


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
def processar_pontos_v27(pts, fps, largura, altura, nome="", debug=False):
    """Núcleo da v27: recebe os 33 pontos já rastreados."""
    xy, lado = preparar_pontos(pts, largura, altura)
    h = altura_normalizada(xy[:, OMBRO[lado], 1])
    tremor = medir_tremor(xy)
    if tremor > LIMIAR_TREMOR:                         # v26: pontos tremidos -> suaviza posições
        n = max(3, int(round(SUAV_POSICAO_TREMOR_S * fps)) | 1)
        xy = uniform_filter1d(xy, n, axis=0)
    altura_ombros = uniform_filter1d(-(xy[:, 11, 1] + xy[:, 12, 1]) / 2, SUAV_ALTURA)
    mov = movimento_normalizado(xy)
    eventos = segmentar_subidas(h)
    recuo = int(RECUO_MAX_S * fps)

    print(f"📹 {nome} | FPS={fps:.2f} | Lado: {lado} | {len(eventos)} subida(s)")
    # v26: subidas descartadas (não chegou em pé, duração impossível) não contam;
    # valem as 3 primeiras subidas válidas (protocolo)
    tentativas, fim_anterior = [], -1
    for j, (idx_deitou, idx_cruzou, idx_limite) in enumerate(eventos):
        if len(tentativas) >= MAX_TENTATIVAS:
            break
        # v25: o início nunca vem antes de a pessoa deitar (exceto na 1ª tentativa)
        idx_min = max(idx_deitou - recuo if j == 0 else idx_deitou, fim_anterior + 1, 0)
        q_ini = detectar_inicio(mov, h, idx_cruzou, idx_min)
        q_fim = detectar_fim(altura_ombros, q_ini, idx_cruzou, idx_limite)
        fim_anterior = q_fim
        if np.max(h[idx_cruzou:idx_limite]) < H_EM_PE:   # v26: não chegou em pé
            if debug:
                print(f"   (subida {j + 1}: não chegou em pé, descartada)")
            continue
        if q_fim <= q_ini:
            if debug:
                print(f"   (subida {j + 1}: fim antes do início, descartada)")
            continue
        q_ini = max(q_ini - AJUSTE_INICIO_QUADROS, 0)
        delta_t = (q_fim - q_ini) / fps
        if not DURACAO_MIN_S <= delta_t <= DURACAO_MAX_S:   # v26: duração impossível
            if debug:
                print(f"   (subida {j + 1}: duração {delta_t:.2f} s, descartada)")
            continue
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


def processar_video_sts_v27(caminho_video, debug=False, pasta_cache=None):
    """Roda a v27 num vídeo. Se `pasta_cache` tiver o ID{...}.npz do vídeo,
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
    return processar_pontos_v27(pts, fps, largura, altura, nome, debug)


print("✅ Função v27 pronta")
