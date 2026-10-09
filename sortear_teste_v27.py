"""
STS-Vision — sortear_teste_v27.py
=========================================================================
Sorteia 30 vídeos NOVOS do banco do Projeto Atitude para o 3º teste
independente (v27 congelada e app v27 congelado), estratificados por sexo,
faixa de tempo, idade e avaliador. Gera também uma lista de reservas por vídeo.

Regras (definidas ANTES de ver qualquer vídeo):
  - Elegíveis: 14 a 19 anos; T_FINAL > 0; T1, T2 e T3 registrados; ID sem
    duplicata no banco; um dos 6 avaliadores do Artigo 1; fora dos 21
    vídeos já usados no desenvolvimento (IDs 7100–7132) e fora de TODOS os
    IDs sorteados nos testes da v24 e da v26 (principais e reservas);
    vídeo presente no acervo (lista ids_disponiveis.txt, gerada dos HDs).
  - Estratos de tempo por sexo (percentis de T_FINAL entre os elegíveis do
    mesmo sexo): rápido < P25 (3 por sexo), mediano P25–P75 (5),
    lento P75–P90 (4), muito lento > P90 (3). Total: 15 meninas, 15 meninos.
  - Restrições: cada avaliador com 4 a 6 vídeos; pelo menos 3 vídeos de
    cada idade de 15 a 18 e pelo menos 2 de 14 e 2 de 19; no máximo 2
    vídeos por escola.
  - A primeira semente a partir de 2028 que cumpre todas as restrições é a
    usada (o script informa qual).
  - Reposição: se um vídeo não existir ou não seguir o protocolo, usar a
    1ª reserva do MESMO vídeo (mesmo sexo, estrato de tempo e avaliador);
    se também falhar, a 2ª, e assim por diante.

Códigos do banco: ATT_Sexo 1 = feminino, 2 = masculino;
idade = ATT_Idade + 11 (conferido com os 21 vídeos do desenvolvimento).

Uso: python sortear_teste_v27.py STS_ATITUDE_MEDIDAS_FINAL.xlsx ids_disponiveis.txt
Saídas: sorteio_teste_v27.csv e reservas_teste_v27.csv (na pasta atual)
=========================================================================
"""

import sys

import numpy as np
import pandas as pd

AVALIADORES = ["DARLEY", "ERIANY", "ISA", "KAROL", "MARIA", "SABRINA"]
IDS_TESTE_V24 = {1464, 7378, 2191, 1854, 2635, 6500, 6904, 7001, 7778, 7781, 7835, 7757, 7814, 7717, 7774, 7740,
                 7171, 3036, 2230, 1567, 7244, 1090, 1885, 6817, 2790, 3051, 3064, 6097, 7513, 3265, 7483, 7562,
                 2247, 2932, 7048, 1584, 7593, 2891, 3257, 7563, 3241, 2843, 2831, 7456, 3151, 7923, 1890, 6133,
                 1641, 1349, 1361, 1263, 1637, 6955, 3114, 1371, 7780, 7842, 3391, 7816, 2959, 2926, 2293, 7161,
                 3312, 3292, 7458, 3334, 6971, 2129, 2470, 1761, 7421, 6783, 7322, 7392, 4061, 4067, 3234, 2103}
FILA_TESTE_V26 = {1: [1357, 1300, 1916, 1887], 2: [7501, 3250, 2865, 3273], 3: [7484, 7605, 2884, 3339],
                  4: [1338, 1501, 7087, 7155], 5: [7734, 7651, 7637, 7650], 6: [7737, 7764, 7836, 7648],
                  7: [3303, 2878, 7514, 3258], 8: [6912, 6502, 6979, 1763], 9: [7672, 3390, 7800, 7809],
                  10: [2336, 3039, 7044, 2324], 11: [6935, 1258, 1903, 1014], 12: [2155, 2161, 1085, 1308],
                  13: [7441, 7411, 7268, 1310], 14: [7398, 2180, 7538, 1620], 15: [6484, 3200, 2717, 2648],
                  16: [6077, 1653, 7913, 3484], 17: [1571, 7136, 7140, 1580], 18: [6986, 6547, 1776, 4039],
                  19: [6877, 2111, 7940, 6507], 20: [2169, 7376, 7522, 6776]}
IDS_TESTE_V26 = {i for fila in FILA_TESTE_V26.values() for i in fila}
IDS_USADOS = set(range(7100, 7133)) | IDS_TESTE_V24 | IDS_TESTE_V26   # dev + sorteios v24 e v26
COTAS = {"rapido": 3, "mediano": 5, "lento": 4, "muito_lento": 3}   # por sexo
N_RESERVAS = 3
SEMENTE_INICIAL = 2028


def carregar(arquivo, disponiveis):
    d = pd.read_excel(arquivo, sheet_name="STS_ATITUDE_MEDIDAS")
    d = d[["ID", "AVALIADOR", "RODADA", "DATA", "T1", "T2", "T3", "T_FINAL",
           "ATT_IDGRE", "ATT_IDEscola", "ATT_Sexo", "ATT_Idade"]].copy()
    d["sexo"] = d["ATT_Sexo"].map({1: "F", 2: "M"})
    d["idade"] = d["ATT_Idade"] + 11
    duplicados = d["ID"][d["ID"].duplicated(keep=False)].unique()
    elegivel = (d["idade"].between(14, 19) & (d["T_FINAL"] > 0)
                & d[["T1", "T2", "T3"]].notna().all(axis=1)
                & ~d["ID"].isin(duplicados) & d["AVALIADOR"].isin(AVALIADORES)
                & ~d["ID"].isin(IDS_USADOS) & d["ID"].isin(disponiveis))
    d = d[elegivel].copy()
    for sexo, g in d.groupby("sexo"):
        p25, p75, p90 = g["T_FINAL"].quantile([0.25, 0.75, 0.90])
        d.loc[g.index, "estrato"] = pd.cut(
            g["T_FINAL"], [-np.inf, p25, p75, p90, np.inf],
            labels=["rapido", "mediano", "lento", "muito_lento"], right=False).astype(str)
        print(f"Sexo {sexo}: P25 {p25:.2f} s | P75 {p75:.2f} s | P90 {p90:.2f} s | n = {len(g)}")
    return d


def sortear(d, rng):
    partes = []
    for sexo in ["F", "M"]:
        for estrato, n in COTAS.items():
            g = d[(d["sexo"] == sexo) & (d["estrato"] == estrato)]
            partes.append(g.sample(n, random_state=rng))
    return pd.concat(partes)


def cumpre(amostra):
    por_aval = amostra["AVALIADOR"].value_counts().reindex(AVALIADORES, fill_value=0)
    idades = amostra["idade"].value_counts()
    return (por_aval.between(4, 6).all()
            and all(idades.get(i, 0) >= 3 for i in (15, 16, 17, 18))
            and idades.get(14, 0) >= 2 and idades.get(19, 0) >= 2
            and amostra["ATT_IDEscola"].value_counts().max() <= 2)


def main(arquivo, arq_ids):
    disponiveis = {int(x) for x in open(arq_ids).read().split()}
    d = carregar(arquivo, disponiveis)
    for semente in range(SEMENTE_INICIAL, SEMENTE_INICIAL + 200000):
        rng = np.random.RandomState(semente)
        amostra = sortear(d, rng)
        if cumpre(amostra):
            break
    else:
        raise RuntimeError("Nenhuma semente cumpriu as restrições.")
    print(f"\nSemente usada: {semente}")

    amostra = amostra.sample(frac=1, random_state=rng).reset_index(drop=True)
    amostra.insert(0, "ordem", range(1, len(amostra) + 1))

    reservas = []
    usados = set(amostra["ID"])
    for _, v in amostra.iterrows():
        pool = d[(d["sexo"] == v["sexo"]) & (d["estrato"] == v["estrato"])
                 & (d["AVALIADOR"] == v["AVALIADOR"]) & ~d["ID"].isin(usados)]
        escolhidos = pool.sample(min(N_RESERVAS, len(pool)), random_state=rng)
        usados.update(escolhidos["ID"])
        for k, (_, r) in enumerate(escolhidos.iterrows(), start=1):
            reservas.append({"substitui_ordem": v["ordem"], "substitui_ID": v["ID"],
                             "reserva_n": k, **r.to_dict()})

    colunas = ["ordem", "ID", "AVALIADOR", "RODADA", "DATA", "ATT_IDGRE", "ATT_IDEscola",
               "sexo", "idade", "estrato", "T1", "T2", "T3", "T_FINAL"]
    amostra[colunas].to_csv("sorteio_teste_v27.csv", index=False)
    pd.DataFrame(reservas).to_csv("reservas_teste_v27.csv", index=False)

    print("\nPor avaliador:", amostra["AVALIADOR"].value_counts().to_dict())
    print("Por idade:", amostra["idade"].value_counts().sort_index().to_dict())
    print("Por estrato:", amostra.groupby(["sexo", "estrato"]).size().to_dict())
    fila = {int(v["ordem"]): [int(v["ID"])] + [int(r["ID"]) for r in reservas if r["substitui_ordem"] == v["ordem"]]
            for _, v in amostra.iterrows()}
    print("FILA_TESTE =", fila)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
