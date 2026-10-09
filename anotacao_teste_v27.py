# ===== CÉLULA 1: montar o Drive =====
from google.colab import drive
drive.mount('/content/drive')


# ===== CÉLULA 2: definições do teste e conferência da pasta no Drive =====
import os
import re

PASTA_TESTE = '/content/drive/MyDrive/STS-Vision/videos_teste_v27'
EXTENSOES = ('.mp4', '.mov', '.avi', '.m4v')

# ordem do sorteio: [principal, reserva 1, reserva 2, reserva 3] (sortear_teste_v27.py, semente 46300)
FILA_TESTE = {
    1: [2119, 6486, 2105, 6536],   2: [1598, 7338, 2154, 7546],
    3: [7314, 1028, 2961, 2365],   4: [7791, 7811, 7728, 3415],
    5: [1906, 1284, 1672, 1346],   6: [6542, 1766, 3224, 2133],
    7: [7199, 2948, 2327, 2408],   8: [2147, 7547, 1083, 1601],
    9: [7434, 7239, 1860, 7435],  10: [7746, 3401, 7668, 7750],
    11: [2252, 3160, 7178, 2522], 12: [2794, 3236, 3328, 2915],
    13: [1319, 2168, 7531, 7558], 14: [1787, 7965, 2120, 3205],
    15: [2736, 2775, 1902, 1675], 16: [2977, 3163, 2325, 1537],
    17: [1483, 7253, 1477, 7543], 18: [7675, 7720, 7723, 7793],
    19: [2006, 3440, 3223, 6501], 20: [3201, 4068, 2113, 2132],
    21: [7786, 7660, 7826, 3418], 22: [4026, 7974, 4071, 7978],
    23: [3072, 1954, 1865, 3120], 24: [7298, 2923, 7035, 7012],
    25: [7569, 7504, 3321, 2797], 26: [1921, 1898, 2770, 1253],
    27: [7610, 2866, 2811, 7559], 28: [7409, 7372, 7374, 7351],
    29: [1043, 7145, 7311, 7281], 30: [2904, 2871, 3352, 3251],
}


def _id_do_nome(nome):
    """Extrai o ID do nome do arquivo: 'ID1464_13_05_22.mp4' -> 1464."""
    m = re.match(r'^\s*ID[\s_-]*0*(\d{3,5})(?!\d)', nome, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


assert os.path.isdir(PASTA_TESTE), f"❌ Pasta não encontrada: {PASTA_TESTE}"
ids_na_pasta = {_id_do_nome(a) for a in os.listdir(PASTA_TESTE) if a.lower().endswith(EXTENSOES)}
print(f"📁 {len(ids_na_pasta)} vídeos na pasta do teste")
for ordem, candidatos in FILA_TESTE.items():
    presente = next((i for i in candidatos if i in ids_na_pasta), None)
    if presente is None:
        print(f"❌ Vaga {ordem}: nenhum vídeo na pasta")
    elif presente != candidatos[0]:
        print(f"Vaga {ordem}: reserva ID{presente} (principal ID{candidatos[0]} não existe no HD)")
print("✅ Definições prontas")


# ===== CÉLULA 3: anotador quadro a quadro (versão para qualquer pasta de vídeos) =====
import glob
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import ipywidgets as w
from IPython.display import display

ARQ_GABARITO = '/content/drive/MyDrive/STS-Vision/resultados/gabarito_teste_v27.csv'
ARQ_PROTOCOLO = '/content/drive/MyDrive/STS-Vision/resultados/protocolo_teste_v27.csv'
COLUNAS = ["id", "video", "tentativa", "quadro_inicio", "t_inicio",
           "quadro_fim", "t_fim", "duracao_s", "atualizado_em"]
LARGURA_EXIBICAO = 360


def _carregar_gabarito():
    if os.path.exists(ARQ_GABARITO):
        return pd.read_csv(ARQ_GABARITO)
    return pd.DataFrame(columns=COLUNAS)


def _caminho_video(id_):
    """Procura o vídeo do ID na pasta do teste."""
    for a in sorted(os.listdir(PASTA_TESTE)):
        if a.lower().endswith(EXTENSOES) and _id_do_nome(a) == id_:
            return os.path.join(PASTA_TESTE, a)
    return None


def _carregar_quadros(caminho, largura=LARGURA_EXIBICAO):
    """Lê o vídeo inteiro em sequência e guarda cada quadro como JPEG reduzido."""
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS)
    quadros = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, l = frame.shape[:2]
        frame = cv2.resize(frame, (largura, int(h * largura / l)))
        quadros.append(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    cap.release()
    return quadros, fps


def anotar(id_, n_tentativas=3):
    caminho = _caminho_video(id_)
    if caminho is None:
        print(f"❌ Vídeo do ID{id_} não encontrado em {PASTA_TESTE}")
        return
    video = os.path.basename(caminho)
    print(f"⏳ Carregando {video}...")
    quadros, fps = _carregar_quadros(caminho)
    n = len(quadros)

    marcas = {k: {"inicio": None, "fim": None} for k in range(1, n_tentativas + 1)}
    gab = _carregar_gabarito()
    for _, r in gab[gab["id"] == id_].iterrows():
        k = int(r["tentativa"])
        if k in marcas:
            marcas[k]["inicio"] = None if pd.isna(r["quadro_inicio"]) else int(r["quadro_inicio"])
            marcas[k]["fim"] = None if pd.isna(r["quadro_fim"]) else int(r["quadro_fim"])

    img = w.Image(value=quadros[0], format='jpg', layout=w.Layout(width=f'{LARGURA_EXIBICAO}px'))
    info = w.HTML()
    slider = w.IntSlider(min=1, max=n, value=1, description='Quadro',
                         continuous_update=True, layout=w.Layout(width='640px'))
    pulos = [("−1 s", -round(fps)), ("−10", -10), ("−1", -1),
             ("+1", 1), ("+10", 10), ("+1 s", round(fps))]
    botoes_pulo = [w.Button(description=r, layout=w.Layout(width='70px')) for r, _ in pulos]
    tentativa = w.ToggleButtons(options=list(range(1, n_tentativas + 1)), description='Tentativa')
    b_inicio = w.Button(description='Marcar INÍCIO', button_style='success',
                        layout=w.Layout(width='200px'))
    b_fim = w.Button(description='Marcar FIM', button_style='danger',
                     layout=w.Layout(width='200px'))
    tabela = w.HTML()

    def _t(q):
        return q / fps

    def _render_tabela(msg=""):
        linhas = ""
        for k, m in marcas.items():
            qi, qf = m["inicio"], m["fim"]
            ini = f"{_t(qi):.2f} s (q{qi})" if qi else "—"
            fim = f"{_t(qf):.2f} s (q{qf})" if qf else "—"
            if qi and qf:
                dur = (qf - qi) / fps
                dur_txt = (f"<b>{dur:.2f} s</b>" if dur > 0
                           else "<span style='color:red'>fim antes do início!</span>")
            else:
                dur_txt = "—"
            linhas += f"<tr><td>T{k}</td><td>{ini}</td><td>{fim}</td><td>{dur_txt}</td></tr>"
        tabela.value = (f"<table style='border-collapse:collapse' cellpadding='4' border='1'>"
                        f"<tr><th>Tentativa</th><th>Início</th><th>Fim</th><th>Duração</th></tr>"
                        f"{linhas}</table><p>{msg}</p>")

    def _salvar():
        gab = _carregar_gabarito()
        gab = gab[gab["id"] != id_]
        agora = datetime.now().isoformat(timespec='seconds')
        novas = []
        for k, m in marcas.items():
            qi, qf = m["inicio"], m["fim"]
            if qi is None and qf is None:
                continue
            novas.append({
                "id": id_, "video": video, "tentativa": k,
                "quadro_inicio": qi, "t_inicio": _t(qi) if qi else np.nan,
                "quadro_fim": qf, "t_fim": _t(qf) if qf else np.nan,
                "duracao_s": (qf - qi) / fps if (qi and qf) else np.nan,
                "atualizado_em": agora,
            })
        if novas:
            novas = pd.DataFrame(novas)
            gab = pd.concat([gab, novas], ignore_index=True) if len(gab) else novas
        os.makedirs(os.path.dirname(ARQ_GABARITO), exist_ok=True)
        gab.sort_values(["id", "tentativa"]).to_csv(ARQ_GABARITO, index=False)

    def _mostrar_quadro(change=None):
        q = slider.value
        img.value = quadros[q - 1]
        info.value = (f"<b>{video}</b> &nbsp;|&nbsp; Quadro <b>{q}</b> de {n} "
                      f"&nbsp;|&nbsp; <b>t = {_t(q):.2f} s</b>")

    def _pular(delta):
        slider.value = int(np.clip(slider.value + delta, 1, n))

    def _marcar(tipo):
        marcas[tentativa.value][tipo] = slider.value
        _salvar()
        nome = "INÍCIO" if tipo == "inicio" else "FIM"
        _render_tabela(f"✅ {nome} da T{tentativa.value} = {_t(slider.value):.2f} s — salvo.")

    slider.observe(_mostrar_quadro, names='value')
    for b, (_, delta) in zip(botoes_pulo, pulos):
        b.on_click(lambda _, d=delta: _pular(d))
    b_inicio.on_click(lambda _: _marcar("inicio"))
    b_fim.on_click(lambda _: _marcar("fim"))

    _mostrar_quadro()
    _render_tabela()
    display(w.VBox([info, img, slider, w.HBox(botoes_pulo), tentativa,
                    w.HBox([b_inicio, b_fim]), tabela]))


print("✅ Anotador pronto")


# ===== CÉLULA 4: painel sequencial do teste (cego), com troca por reserva =====
TENTATIVAS_POR_VIDEO = 3
MOTIVOS = ["posição inicial fora do protocolo", "não completou a posição em pé",
           "vídeo corrompido ou incompleto", "participante fora do quadro"]


def _rejeitados():
    """IDs marcados como fora do protocolo (ou sem vídeo)."""
    if os.path.exists(ARQ_PROTOCOLO):
        return set(pd.read_csv(ARQ_PROTOCOLO)["id"].astype(int))
    return set()


def _registrar_rejeicao(ordem, id_, motivo):
    linha = pd.DataFrame([{"ordem": ordem, "id": id_, "motivo": motivo,
                           "registrado_em": datetime.now().isoformat(timespec='seconds')}])
    if os.path.exists(ARQ_PROTOCOLO):
        linha = pd.concat([pd.read_csv(ARQ_PROTOCOLO), linha], ignore_index=True)
    linha.to_csv(ARQ_PROTOCOLO, index=False)
    gab = _carregar_gabarito()
    if len(gab):
        gab[gab["id"] != id_].to_csv(ARQ_GABARITO, index=False)


def _video_da_vaga(ordem):
    """Primeiro candidato da vaga que não foi rejeitado e cujo vídeo existe na pasta."""
    rejeitados = _rejeitados()
    for id_ in FILA_TESTE[ordem]:
        if id_ in rejeitados or _caminho_video(id_) is None:
            continue
        return id_
    return None


def _completas_por_id():
    gab = _carregar_gabarito()
    if gab.empty:
        return {}
    ok = gab.dropna(subset=["quadro_inicio", "quadro_fim"])
    return ok.groupby("id").size().to_dict()


def _proxima_vaga():
    feitas = _completas_por_id()
    for ordem in FILA_TESTE:
        id_ = _video_da_vaga(ordem)
        if id_ is not None and feitas.get(id_, 0) < TENTATIVAS_POR_VIDEO:
            return ordem
    return None


def _status_html():
    feitas = _completas_por_id()
    itens, n_ok = [], 0
    for ordem in FILA_TESTE:
        id_ = _video_da_vaga(ordem)
        if id_ is None:
            itens.append(f"<span style='color:red'>✖ {ordem}</span>")
            continue
        n = feitas.get(id_, 0)
        reserva = "" if id_ == FILA_TESTE[ordem][0] else "ʳ"
        if n >= TENTATIVAS_POR_VIDEO:
            itens.append(f"<span style='color:green'>✅ {ordem}{reserva}</span>")
            n_ok += 1
        elif n > 0:
            itens.append(f"<span style='color:orange'>⏳ {ordem}{reserva} ({n}/3)</span>")
        else:
            itens.append(f"<span style='color:gray'>○ {ordem}{reserva}</span>")
    return (f"<b>Teste v27: {n_ok}/{len(FILA_TESTE)} vídeos completos</b> "
            f"&nbsp;|&nbsp; ʳ = reserva &nbsp;|&nbsp; salvando em <code>gabarito_teste_v27.csv</code><br>"
            + " &nbsp; ".join(itens))


def painel_teste():
    assert ARQ_GABARITO.endswith('gabarito_teste_v27.csv'), "PARE: arquivo de gabarito errado!"
    status = w.HTML()
    escolha = w.Dropdown(options=list(FILA_TESTE), description="Vaga:",
                         layout=w.Layout(width="160px"))
    b_abrir = w.Button(description="Abrir esta", layout=w.Layout(width="110px"))
    b_proximo = w.Button(description="Próximo pendente ▶", button_style="primary",
                         layout=w.Layout(width="180px"))
    motivo = w.Dropdown(options=MOTIVOS, description="Motivo:", layout=w.Layout(width="420px"))
    b_rejeitar = w.Button(description="⛔ Fora do protocolo → reserva", button_style="warning",
                          layout=w.Layout(width="260px"))
    saida = w.Output()
    atual = {"ordem": None}

    def _abrir(ordem):
        status.value = _status_html()
        saida.clear_output()
        with saida:
            if ordem is None:
                print("🎉 Anotação do teste completa! Rode a célula de conferência.")
                return
            id_ = _video_da_vaga(ordem)
            if id_ is None:
                print(f"❌ Vaga {ordem}: nenhum candidato disponível (principal e 3 reservas).")
                return
            atual["ordem"] = ordem
            escolha.value = ordem
            print(f"Vaga {ordem} de {len(FILA_TESTE)}")
            anotar(id_)

    def _rejeitar(_):
        ordem = atual["ordem"]
        if ordem is None:
            return
        id_ = _video_da_vaga(ordem)
        _registrar_rejeicao(ordem, id_, motivo.value)
        _abrir(ordem)

    b_abrir.on_click(lambda _: _abrir(escolha.value))
    b_proximo.on_click(lambda _: _abrir(_proxima_vaga()))
    b_rejeitar.on_click(_rejeitar)

    display(w.VBox([status, w.HBox([b_proximo, escolha, b_abrir]),
                    w.HBox([motivo, b_rejeitar]), saida]))
    _abrir(_proxima_vaga())


painel_teste()


# ===== CÉLULA 5: conferência final =====
gab = _carregar_gabarito()
usados = [_video_da_vaga(o) for o in FILA_TESTE]
print(f"Vagas com vídeo: {sum(u is not None for u in usados)}/30")
print(f"Linhas: {len(gab)} (esperado: 90) | IDs: {gab['id'].nunique()} (esperado: 30) | "
      f"sem duração: {gab['duracao_s'].isna().sum()}")
for o in FILA_TESTE:
    u = _video_da_vaga(o)
    if u != FILA_TESTE[o][0]:
        sem_video = [i for i in FILA_TESTE[o] if _caminho_video(i) is None]
        print(f"Vaga {o}: usando {u} no lugar de {FILA_TESTE[o][0]} | sem vídeo na pasta: {sem_video}")
if os.path.exists(ARQ_PROTOCOLO):
    print("\nRejeitados por protocolo:")
    print(pd.read_csv(ARQ_PROTOCOLO).to_string(index=False))


# ===== CÉLULA 6: congelar a referência do teste (antes de qualquer contato com a v27) =====
import hashlib
import shutil

PASTA_RES = '/content/drive/MyDrive/STS-Vision/resultados'


def sha256(caminho):
    """Impressão digital do arquivo: muda se qualquer byte mudar."""
    with open(caminho, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


gab = _carregar_gabarito()
assert len(gab) == 90 and gab['id'].nunique() == 30 and gab['duracao_s'].notna().all(), \
    "PARE: a anotação não está completa (esperado 90 linhas, 30 IDs, todas com duração)."

data = datetime.now().strftime('%Y%m%d')
for nome in ['gabarito_teste_v27.csv', 'protocolo_teste_v27.csv']:
    origem = f'{PASTA_RES}/{nome}'
    if not os.path.exists(origem):
        pd.DataFrame(columns=["ordem", "id", "motivo", "registrado_em"]).to_csv(origem, index=False)
    congelado = origem.replace('.csv', f'_CONGELADO_{data}.csv')
    shutil.copy2(origem, congelado)
    print(f"🔒 {congelado.split('/')[-1]}\n   SHA-256: {sha256(congelado)}")
print("\nCopie as duas linhas de SHA-256 e me mande (só o hash, não o arquivo).")
