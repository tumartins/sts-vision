# ===== CÉLULA 1: montar o Drive e instalar o MediaPipe =====
from google.colab import drive
drive.mount('/content/drive')

!pip install -q mediapipe
import mediapipe as mp
print(f"MediaPipe {mp.__version__}  (o desenvolvimento usou 1.1.0)")


# ===== CÉLULA 2: conferir os arquivos congelados e carregar a v24, a v25 e a v27 =====
import glob
import hashlib
import os
import re
import urllib.request

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

RAIZ = '/content/drive/MyDrive/STS-Vision'
PASTA_RES = f'{RAIZ}/resultados'
PASTA_CODIGO = f'{RAIZ}/codigo'
PASTA_TESTE = f'{RAIZ}/videos_teste_v27'
PASTA_PONTOS = f'{PASTA_RES}/export_teste_v27_mp{mp.__version__}'
PASTA_SAIDA = f'{PASTA_RES}/teste_v27'
MODELO = 'heavy'
VERSOES_TESTE = ['v27', 'v25', 'v24']      # v27 = análise principal (MELHOR TEMPO)

# impressões digitais registradas ANTES de rodar o algoritmo nestes vídeos
HASH = {
    'gabarito_teste_v27_CONGELADO': 'f4ab72e590e6d7d82a06772bc78e263b2386c74ca6607d0756457fa7dd4a9942',
    'protocolo_teste_v27_CONGELADO': '519d7391d09a80a0a381af54b4e5210b8c19cce5e0a4027b64614135c2e7cdf1',
    'processar_video_sts_v24.py': '8fc8796999db2871339eb554ff195bdfa50c39355a65119459a91f75f38a733b',
    'processar_video_sts_v25.py': '2d3c0b1f2f748e68e97023440c904e2f84da4072cdd6aba954bcb92eae429254',
    'processar_video_sts_v27.py': '8b62c9665a8221ca82c7524fe58f5ca05720d0c552b3d2924bbb9f6fcef6650e',
}


def sha256(caminho):
    with open(caminho, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def conferir(caminho, chave):
    ok = sha256(caminho) == HASH[chave]
    print(f"{'🔒 OK  ' if ok else '❌ DIFERENTE'} {os.path.basename(caminho)}")
    assert ok, f"PARE: {os.path.basename(caminho)} não é o arquivo congelado."
    return caminho


def achar_congelado(prefixo):
    """Entre os arquivos congelados com esse nome, pega o que bate com o hash."""
    candidatos = sorted(glob.glob(f'{PASTA_RES}/{prefixo}_*.csv'))
    assert candidatos, f"PARE: nenhum {prefixo}_*.csv em {PASTA_RES}"
    for c in candidatos:
        if sha256(c) == HASH[prefixo]:
            return conferir(c, prefixo)
    return conferir(candidatos[-1], prefixo)   # nenhum bate -> mostra ❌ e para


ARQ_GAB = achar_congelado('gabarito_teste_v27_CONGELADO')
achar_congelado('protocolo_teste_v27_CONGELADO')

# cada versão roda no SEU espaço de nomes (têm constantes e funções com o mesmo nome)
VERSOES = {}
for versao in VERSOES_TESTE:
    nome = f'processar_video_sts_{versao}.py'
    VERSOES[versao] = {'__name__': versao}
    exec(open(conferir(f'{PASTA_CODIGO}/{nome}', nome)).read(), VERSOES[versao])

arq_modelo = f'/content/pose_landmarker_{MODELO}.task'
if not os.path.exists(arq_modelo):
    urllib.request.urlretrieve(
        f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{MODELO}/float16/latest/pose_landmarker_{MODELO}.task", arq_modelo)
options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=arq_modelo),
                                       running_mode=vision.RunningMode.VIDEO)
for ns in VERSOES.values():
    ns['options'] = options          # o rastreio usa `options` como variável global
os.makedirs(PASTA_PONTOS, exist_ok=True)
os.makedirs(PASTA_SAIDA, exist_ok=True)
print(f"✅ Tudo conferido | modelo {MODELO} | pontos em {PASTA_PONTOS}")


# ===== CÉLULA 3: rastrear os 30 vídeos do teste (uma vez só; ~50 min) =====
# Se o Colab cair, rode as Células 1, 2 e 3 de novo: vídeos já feitos são pulados.
import time

import numpy as np
import pandas as pd

gab = pd.read_csv(ARQ_GAB).astype({"id": int, "tentativa": int})
ids = sorted(gab["id"].unique())
assert len(ids) == 30 and len(gab) == 90, "PARE: o gabarito congelado não tem 30 vídeos / 90 tentativas"


def video_do_id(id_):
    for a in os.listdir(PASTA_TESTE):
        m = re.match(r'^\s*ID[\s_-]*0*(\d{3,5})(?!\d)', a, flags=re.IGNORECASE)
        if m and int(m.group(1)) == id_ and a.lower().endswith(('.mp4', '.mov', '.avi', '.m4v')):
            return os.path.join(PASTA_TESTE, a)
    return None


t_total = time.time()
for i, id_ in enumerate(ids, start=1):
    destino = f'{PASTA_PONTOS}/ID{id_}.npz'
    if os.path.exists(destino):
        print(f"[{i}/{len(ids)}] ID{id_}: já feito")
        continue
    caminho = video_do_id(id_)
    assert caminho, f"Vídeo do ID{id_} não está em {PASTA_TESTE}"
    t0 = time.time()
    # o rastreio é o mesmo nas três versões; uso o da v27
    pts, fps, largura, altura = VERSOES['v27']['rastrear_33_pontos'](caminho)
    np.savez_compressed(destino, pts=pts, fps=fps, largura=largura, altura=altura,
                        video=os.path.basename(caminho), mediapipe=mp.__version__, modelo=MODELO)
    print(f"[{i}/{len(ids)}] ID{id_}: {len(pts)} quadros em {time.time() - t0:.0f} s")
print(f"\n⏱️ {(time.time() - t_total) / 60:.1f} min | "
      f"{len(glob.glob(f'{PASTA_PONTOS}/*.npz'))}/30 prontos")


# ===== CÉLULA 4: v27 (principal = melhor tempo), v25 e v24 contra a referência congelada =====
import contextlib
import io

from scipy import stats

ALFA = 0.05
LIMIAR_GROSSEIRO = 0.30   # erro absoluto (s) considerado falha grosseira


def icc_duas_vias(x, y, alfa=ALFA):
    """ICC(C,1) e ICC(A,1) para 2 avaliadores (McGraw & Wong, 1996), IC pela F."""
    dados = np.column_stack([x, y]).astype(float)
    n, k = dados.shape
    media = dados.mean()
    ms_linhas = k * np.sum((dados.mean(axis=1) - media) ** 2) / (n - 1)
    ms_colunas = n * np.sum((dados.mean(axis=0) - media) ** 2) / (k - 1)
    resid = (dados - dados.mean(axis=1, keepdims=True)
             - dados.mean(axis=0, keepdims=True) + media)
    ms_erro = np.sum(resid ** 2) / ((n - 1) * (k - 1))
    icc_c = (ms_linhas - ms_erro) / (ms_linhas + (k - 1) * ms_erro)
    icc_a = (ms_linhas - ms_erro) / (ms_linhas + (k - 1) * ms_erro
                                     + k / n * (ms_colunas - ms_erro))
    gl1, gl2 = n - 1, (n - 1) * (k - 1)
    f_obs = ms_linhas / ms_erro
    f_l = f_obs / stats.f.ppf(1 - alfa / 2, gl1, gl2)
    f_u = f_obs * stats.f.ppf(1 - alfa / 2, gl2, gl1)
    c_inf, c_sup = (f_l - 1) / (f_l + k - 1), (f_u - 1) / (f_u + k - 1)
    a = k * icc_a / (n * (1 - icc_a))
    b = 1 + k * icc_a * (n - 1) / (n * (1 - icc_a))
    v = ((a * ms_colunas + b * ms_erro) ** 2
         / ((a * ms_colunas) ** 2 / (k - 1) + (b * ms_erro) ** 2 / ((n - 1) * (k - 1))))
    f1 = stats.f.ppf(1 - alfa / 2, n - 1, v)
    f2 = stats.f.ppf(1 - alfa / 2, v, n - 1)
    termo = k * ms_colunas + (k * n - k - n) * ms_erro
    a_inf = n * (ms_linhas - f1 * ms_erro) / (f1 * termo + n * ms_linhas)
    a_sup = n * (f2 * ms_linhas - ms_erro) / (termo + n * f2 * ms_linhas)
    return {"icc_c": icc_c, "icc_c_inf": c_inf, "icc_c_sup": c_sup,
            "icc_a": icc_a, "icc_a_inf": a_inf, "icc_a_sup": a_sup}


def concordancia(alg, ref):
    """Erro, Bland-Altman, viés proporcional, razão log (IC pela t), EMD e ICC."""
    alg, ref = np.asarray(alg, float), np.asarray(ref, float)
    dif = alg - ref
    vies, dp = dif.mean(), dif.std(ddof=1)
    reg = stats.linregress((alg + ref) / 2, dif)
    dlog = np.log(alg) - np.log(ref)
    ep_log = dlog.std(ddof=1) / np.sqrt(len(dlog))
    t_crit = stats.t.ppf(1 - ALFA / 2, len(dlog) - 1)
    res = {"n": len(dif), "erro_abs_medio": np.abs(dif).mean(),
           "erro_abs_max": np.abs(dif).max(),
           "ate_0_10s": int((np.abs(dif) <= 0.1 + 1e-9).sum()),
           "acima_0_30s": int((np.abs(dif) > LIMIAR_GROSSEIRO).sum()),
           "vies": vies, "loa_inf": vies - 1.96 * dp, "loa_sup": vies + 1.96 * dp,
           "p_vies_prop": reg.pvalue,
           "razao_log": np.exp(dlog.mean()),
           "razao_inf": np.exp(dlog.mean() - t_crit * ep_log),
           "razao_sup": np.exp(dlog.mean() + t_crit * ep_log),
           "p_razao_wilcoxon": stats.wilcoxon(dlog).pvalue,
           "emd": 1.96 * dp}            # EMD = 1,96 × √2 × EPM, com EPM = DP/√2
    res.update(icc_duas_vias(alg, ref))
    return res


def rodar(func):
    """Roda uma versão nos vídeos do teste e pareia cada tentativa da referência
    com a tentativa do algoritmo que se sobrepõe a ela no tempo."""
    linhas = []
    for id_ in ids:
        d = np.load(f'{PASTA_PONTOS}/ID{id_}.npz')
        with contextlib.redirect_stdout(io.StringIO()):
            r = func(d["pts"], float(d["fps"]), int(d["largura"]), int(d["altura"]))
        for _, g in gab[gab["id"] == id_].sort_values("tentativa").iterrows():
            par = [t for t in r["tentativas"]
                   if t["tempo_fim"] > g["t_inicio"] and t["tempo_inicio"] < g["t_fim"]]
            t = par[0] if par else None
            linhas.append({"id": id_, "tentativa": int(g["tentativa"]), "ref_s": g["duracao_s"],
                           "alg_s": t["delta_t"] if t else np.nan,
                           "erro_inicio_s": t["tempo_inicio"] - g["t_inicio"] if t else np.nan,
                           "erro_fim_s": t["tempo_fim"] - g["t_fim"] if t else np.nan})
    return pd.DataFrame(linhas)


tabs = {v: rodar(VERSOES[v][f"processar_pontos_{v}"]) for v in VERSOES_TESTE}
resumo = {}
for versao, tab in tabs.items():
    tab["erro_s"] = tab["alg_s"] - tab["ref_s"]
    ok = tab.dropna(subset=["alg_s"])
    melhor = ok.groupby("id").agg(ref_s=("ref_s", "min"), alg_s=("alg_s", "min"))
    resumo[f"{versao}_tentativa"] = concordancia(ok["alg_s"], ok["ref_s"])
    resumo[f"{versao}_tentativa"]["detectadas"] = len(ok)
    resumo[f"{versao}_tentativa"]["erro_inicio_abs"] = ok["erro_inicio_s"].abs().mean()
    resumo[f"{versao}_tentativa"]["vies_inicio"] = ok["erro_inicio_s"].mean()
    resumo[f"{versao}_tentativa"]["erro_fim_abs"] = ok["erro_fim_s"].abs().mean()
    resumo[f"{versao}_tentativa"]["vies_fim"] = ok["erro_fim_s"].mean()
    resumo[f"{versao}_melhor"] = concordancia(melhor["alg_s"], melhor["ref_s"])
    tab.to_csv(f'{PASTA_SAIDA}/tentativas_{versao}_teste.csv', index=False)
resumo = pd.DataFrame(resumo)
resumo.to_csv(f'{PASTA_SAIDA}/resumo_teste_v27.csv')

print("===== 3º TESTE INDEPENDENTE (v27): 30 vídeos, 90 tentativas =====")
print("Análise principal: v27_melhor\n")
print(resumo.astype(float).round(3).to_string())

# critério de sucesso pré-especificado (PLANO_TESTE_V27.md)
p = resumo["v27_melhor"]
ok_icc, ok_erro = p["icc_a_inf"] >= 0.75, p["erro_abs_medio"] <= 0.15
print(f"\nDESFECHO PRINCIPAL — v27, melhor tempo (n = {int(p['n'])}): "
      f"ICC(A,1) = {p['icc_a']:.3f} [IC95% {p['icc_a_inf']:.3f}–{p['icc_a_sup']:.3f}] | "
      f"erro absoluto médio = {p['erro_abs_medio']:.3f} s")
print(f"  limite inferior do IC ≥ 0,75: {'SIM' if ok_icc else 'NÃO'} | "
      f"erro ≤ 0,15 s: {'SIM' if ok_erro else 'NÃO'} → "
      f"{'CRITÉRIO ATINGIDO' if ok_icc and ok_erro else 'CRITÉRIO NÃO ATINGIDO'}")


def pareado(nova, antiga, semente=2026):
    """Diferença no erro absoluto médio de cada vídeo (nova − antiga)."""
    ea = {v: tabs[v]["erro_s"].abs().groupby(tabs[v]["id"]).mean() for v in (nova, antiga)}
    dif = (ea[nova] - ea[antiga]).dropna().values
    rng = np.random.default_rng(semente)
    boot = [rng.choice(dif, len(dif)).mean() for _ in range(10000)]
    p = stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
    print(f"{nova} − {antiga} (erro absoluto por vídeo): {dif.mean():+.3f} s "
          f"[IC95% {np.percentile(boot, 2.5):+.3f} a {np.percentile(boot, 97.5):+.3f}] | "
          f"{nova} melhor em {(dif < 0).sum()}, pior em {(dif > 0).sum()}, "
          f"igual em {(dif == 0).sum()} | Wilcoxon p = {p:.3f}")


print()
pareado("v27", "v25")      # secundária pré-especificada
pareado("v27", "v24")
print(f"\n✅ Tabelas salvas em {PASTA_SAIDA}")


# ===== CÉLULA 5: tentativas com falha grosseira (erro > 0,3 s) =====
for versao in VERSOES_TESTE:
    tab = tabs[versao]
    nao_det = tab[tab["alg_s"].isna()]
    graves = tab[tab["erro_s"].abs() > LIMIAR_GROSSEIRO]
    print(f"\n===== {versao}: {len(graves)} tentativa(s) com erro > {LIMIAR_GROSSEIRO} s | "
          f"{len(nao_det)} não detectada(s) =====")
    if len(graves):
        print(graves[["id", "tentativa", "ref_s", "alg_s", "erro_s",
                      "erro_inicio_s", "erro_fim_s"]].round(3).to_string(index=False))
    if len(nao_det):
        print("Não detectadas:", nao_det[["id", "tentativa"]].values.tolist())
