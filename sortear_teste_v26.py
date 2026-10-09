"""
STS-Vision — sortear_teste_v26.py
=========================================================================
Sorteia 20 vídeos NOVOS do banco do Projeto Atitude para o teste
independente da v24, estratificados por sexo, faixa de tempo, idade e
avaliador. Gera também uma lista de reservas por vídeo.

Regras (definidas ANTES de ver qualquer vídeo):
  - Elegíveis: 14 a 19 anos; T_FINAL > 0; T1, T2 e T3 registrados; ID sem
    duplicata no banco; um dos 6 avaliadores do Artigo 1; fora dos 21
    vídeos já usados no desenvolvimento (IDs 7100–7132).
  - Estratos de tempo por sexo (percentis de T_FINAL entre os elegíveis do
    mesmo sexo): rápido < P25 (2 por sexo), mediano P25–P75 (3),
    lento P75–P90 (3), muito lento > P90 (2). Total: 10 meninas, 10 meninos.
  - Restrições: cada avaliador com 3 ou 4 vídeos; pelo menos 2 vídeos de
    cada idade de 15 a 18 e pelo menos 1 de 14 e 1 de 19; no máximo 2
    vídeos por escola.
  - A primeira semente a partir de 2026 que cumpre todas as restrições é a
    usada (o script informa qual).
  - Reposição: se um vídeo não existir ou não seguir o protocolo, usar a
    1ª reserva do MESMO vídeo (mesmo sexo, estrato de tempo e avaliador);
    se também falhar, a 2ª, e assim por diante.

Códigos do banco: ATT_Sexo 1 = feminino, 2 = masculino;
idade = ATT_Idade + 11 (conferido com os 21 vídeos do desenvolvimento).

Uso: python sortear_teste_v26.py STS_ATITUDE_MEDIDAS_FINAL.xlsx
Saídas: sorteio_teste_v26.csv e reservas_teste_v26.csv (na pasta atual)
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
IDS_USADOS = set(range(7100, 7133)) | IDS_TESTE_V24      # desenvolvimento + todo o sorteio do teste da v24
COTAS = {"rapido": 2, "mediano": 3, "lento": 3, "muito_lento": 2}   # por sexo
N_RESERVAS = 3
SEMENTE_INICIAL = 2027


def carregar(arquivo):
    d = pd.read_excel(arquivo, sheet_name="STS_ATITUDE_MEDIDAS")
    d = d[["ID", "AVALIADOR", "RODADA", "DATA", "T1", "T2", "T3", "T_FINAL",
           "ATT_IDGRE", "ATT_IDEscola", "ATT_Sexo", "ATT_Idade"]].copy()
    d["sexo"] = d["ATT_Sexo"].map({1: "F", 2: "M"})
    d["idade"] = d["ATT_Idade"] + 11
    duplicados = d["ID"][d["ID"].duplicated(keep=False)].unique()
    elegivel = (d["idade"].between(14, 19) & (d["T_FINAL"] > 0)
                & d[["T1", "T2", "T3"]].notna().all(axis=1)
                & ~d["ID"].isin(duplicados) & d["AVALIADOR"].isin(AVALIADORES)
                & ~d["ID"].isin(IDS_USADOS))
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
    return (por_aval.between(3, 4).all()
            and all(idades.get(i, 0) >= 2 for i in (15, 16, 17, 18))
            and idades.get(14, 0) >= 1 and idades.get(19, 0) >= 1
            and amostra["ATT_IDEscola"].value_counts().max() <= 2)


def main(arquivo):
    d = carregar(arquivo)
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
    amostra[colunas].to_csv("sorteio_teste_v26.csv", index=False)
    pd.DataFrame(reservas).to_csv("reservas_teste_v26.csv", index=False)

    print("\nPor avaliador:", amostra["AVALIADOR"].value_counts().to_dict())
    print("Por idade:", amostra["idade"].value_counts().sort_index().to_dict())
    print("Por estrato:", amostra.groupby(["sexo", "estrato"]).size().to_dict())
    fila = {int(v["ordem"]): [int(v["ID"])] + [int(r["ID"]) for r in reservas if r["substitui_ordem"] == v["ordem"]]
            for _, v in amostra.iterrows()}
    print("FILA_TESTE =", fila)


if __name__ == "__main__":
    main(sys.argv[1])
