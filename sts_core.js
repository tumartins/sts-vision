
/*
 * STS-Vision — sts_core.js
 * Núcleo do cronômetro do Supine-to-Stand em tempo real — algoritmo v27
 * (v27: "parado" do início só com a pessoa ainda deitada, 0,1 s; repouso local só nos quadros deitados)
 * (v23 + v24: fim pela média dos 2 ombros, limiar 15%, início 1 quadro antes;
 *  v25: "parado" relativo ao repouso local, sem recuo cego, início nunca antes
 *  de deitar; app: só registra a tentativa depois que a pessoa ficou EM PÉ).
 * Uso: const c = new CronometroSTS(largura, altura);
 *      const st = c.processar(tSegundos, pontos)  // pontos: 33 x [x,y,z,vis] ou null
 */

const PONTOS_MOVIMENTO = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const LADO_ESQ = [11, 13, 15, 23, 25, 27];
const LADO_DIR = [12, 14, 16, 24, 26, 28];
const OMBRO = { LEFT: 11, RIGHT: 12 };
const TORNOZELO = { LEFT: 27, RIGHT: 28 };

const CALIB_S = 1.5;
const CALIB_MAX_OSCILACAO = 0.015;
const CALIB_MAX_INCLINACAO = 0.35;
const LIMIAR_LEVANTOU = 0.5;
const LIMIAR_DEITOU = 0.2;
const SUAV_VELOCIDADE_S = 0.10;
const PCT_RUIDO = 50;
const LIMIAR_MOVIMENTO = 5.0;
const PARADO_S = 5 / 30;
const PARADO_DEITADO_S = 5 / 30;     // v27 (app): mesmo 5/30 s da v25, agora só com a pessoa deitada
const RECUO_MAX_S = 3.0;
const SUAV_ALTURA_S = 7 / 30;
const FRACAO_VEL_FIM = 0.15;          // v24
const FRACAO_ALTURA_FIM = 0.8;
const CONFIRMA_FIM_S = 0.3;
const MAX_TENTATIVAS = 3;
const AJUSTE_INICIO_QUADROS = 1;      // v24: início 1 quadro antes do detectado
const PCT_REPOUSO_LOCAL = 10;         // v25: repouso local = percentil 10 do movimento
const FATOR_REPOUSO_LOCAL = 2.0;      // v25: parado = abaixo de max(limiar, 2 x repouso local)
const H_EM_PE = 0.75;                 // app: só fecha a tentativa depois de passar daqui (em pé >= 0,89)
const SUBINDO_MAX_S = 6.0;            // app: fecha mesmo assim depois de 6 s (nunca trava)
const VERSAO_ALGORITMO = "v27";
// Travas de plausibilidade do app (não mudam nada nos 41 vídeos de validação):
const DURACAO_MIN_S = 0.9;            // STS mais rápido observado: 1,30 s
const DURACAO_MAX_S = 8.0;
const COMPRIMENTO_MIN = 0.12;         // ombro–tornozelo deitado (fração da altura da imagem); validação: 0,21–0,30
const VIS_MIN_CALIB = 0.5;
const LIMIAR_TREMOR = 0.001;          // app: tremor dos pontos parados (validação: ≤ 0,0007; modelos leves: mais)
let SUAV_POSICAO_TREMOR_S = 0.2;      // app: se houver tremor, suaviza as posições (média centrada de 0,2 s)            // ombro e tornozelo bem visíveis na calibração

const AGUARDANDO = "AGUARDANDO", CALIBRANDO = "CALIBRANDO", PRONTO = "PRONTO",
  SUBINDO = "SUBINDO", EM_PE = "EM_PE", FINALIZADO = "FINALIZADO";

// ---------------- utilidades numéricas (equivalentes ao numpy/scipy) ----------------
function arredondarPar(x) {                       // round() do Python (meio -> par)
  const f = Math.floor(x), d = x - f;
  if (Math.abs(d - 0.5) < 1e-12) return f % 2 === 0 ? f : f + 1;
  return Math.round(x);
}
function quadros(seg, fps, impar = false) {
  let n = Math.max(1, arredondarPar(seg * fps));
  return impar && n % 2 === 0 ? n + 1 : n;
}
function percentil(valores, p) {                  // np.percentile (linear)
  const a = Float64Array.from(valores).sort();
  const r = (p / 100) * (a.length - 1), i = Math.floor(r), f = r - i;
  return i + 1 < a.length ? a[i] * (1 - f) + a[i + 1] * f : a[i];
}
const mediana = (v) => percentil(v, 50);
function desvio(v) {                              // np.std (ddof = 0)
  const m = v.reduce((s, x) => s + x, 0) / v.length;
  return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length);
}
function filtroMedia(serie, tam) {                // scipy uniform_filter1d, mode='reflect'
  const n = serie.length, out = new Float64Array(n);
  const ini = -Math.floor(tam / 2), fim = ini + tam - 1;
  const idx = (i) => {
    while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
    return i;
  };
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let j = ini; j <= fim; j++) s += serie[idx(k + j)];
    out[k] = s / tam;
  }
  return out;
}
function gradiente(s) {                           // np.gradient
  const n = s.length, g = new Float64Array(n);
  if (n < 2) return g;
  g[0] = s[1] - s[0]; g[n - 1] = s[n - 1] - s[n - 2];
  for (let k = 1; k < n - 1; k++) g[k] = (s[k + 1] - s[k - 1]) / 2;
  return g;
}
function argmax(a, i0, i1) { let k = i0; for (let i = i0; i < i1; i++) if (a[i] > a[k]) k = i; return k; }
function argmin(a, i0, i1) { let k = i0; for (let i = i0; i < i1; i++) if (a[i] < a[k]) k = i; return k; }

function detectarInicio(mov, h, idxCruzou, idxMin, limiarBase, nParado) {
  // v25: o limiar se adapta ao repouso LOCAL; sem trecho parado -> momento mais quieto
  // v27: "parado" só conta com a pessoa AINDA DEITADA (h < LIMIAR_DEITOU), e o repouso
  //      local é medido só nesses quadros (pausa sentado/ajoelhado no meio da subida não vale)
  const trecho = Array.from(mov.slice(idxMin, idxCruzou + 1));
  const deitado = trecho.map((_, i) => h[idxMin + i] < LIMIAR_DEITOU);
  const base = trecho.filter((_, i) => deitado[i]);
  const limiar = Math.max(limiarBase, FATOR_REPOUSO_LOCAL *
    percentil(base.length >= nParado ? base : trecho, PCT_REPOUSO_LOCAL));
  let parados = 0;
  for (let k = idxCruzou; k >= idxMin; k--) {
    if (mov[k] < limiar && h[k] < LIMIAR_DEITOU) { if (++parados >= nParado) return k + nParado; }
    else parados = 0;
  }
  if (deitado.some(Boolean)) {
    const suave = filtroMedia(trecho.map((v, i) => (deitado[i] ? v : Infinity)), nParado);
    return idxMin + argmin(suave, 0, suave.length);
  }
  const suave = filtroMedia(trecho, nParado);
  return idxMin + argmin(suave, 0, suave.length);
}
function detectarFim(s, idxIni, idxCruzou, idxLimite) {
  const kTopo = argmax(s, idxCruzou, idxLimite);
  const kBaixo = argmin(s, idxIni, kTopo + 1);
  const vel = gradiente(s);
  const kPico = argmax(vel, kBaixo, kTopo + 1);
  const alvo = s[kBaixo] + FRACAO_ALTURA_FIM * (s[kTopo] - s[kBaixo]);
  for (let k = kPico; k < idxLimite; k++)
    if (vel[k] < FRACAO_VEL_FIM * vel[kPico] && s[k] >= alvo) return k;
  return kTopo;
}

// ---------------------------------- cronômetro ----------------------------------
class CronometroSTS {
  constructor(largura, altura, fpsNominal = 30) {
    this.escalaX = largura / altura;
    this.fps = fpsNominal;
    this.reiniciar();
  }

  reiniciar() {
    this.t = []; this.xy = []; this.vel = []; this.vis = [];
    this.estado = AGUARDANDO; this.lado = null;
    this.nivelDeitado = null; this.comprimento = null; this.ruidoCalib = null;
    this.idxRepouso = []; this.tentativas = [];
    this.idxDeitou = null; this.idxCruzou = null;
    this.topo = -Infinity; this.idxTopo = null; this.fimAnterior = -1; this.chegouEmPe = false;
    this.tProvisorio = null; this.parado = 0; this.descartes = []; this.tremor = null;
    this.mensagem = "Deite de lado para a câmera";
  }

  processar(t, pontos) {
    let xy, vis;
    if (!pontos) {
      if (!this.xy.length) return this.status();
      xy = this.xy[this.xy.length - 1]; vis = this.vis[this.vis.length - 1];
    } else {
      xy = new Float64Array(66); vis = new Float64Array(33);
      const ant = this.xy[this.xy.length - 1];
      for (let i = 0; i < 33; i++) {
        let x = pontos[i][0] * this.escalaX, y = pontos[i][1];
        if (Number.isNaN(x) && ant) x = ant[2 * i];
        if (Number.isNaN(y) && ant) y = ant[2 * i + 1];
        xy[2 * i] = x; xy[2 * i + 1] = y; vis[i] = pontos[i][3] ?? 0;
      }
    }
    const k = this.xy.length;
    this.t.push(t); this.xy.push(xy); this.vis.push(vis);
    const ant = k ? this.xy[k - 1] : xy, v = new Float64Array(PONTOS_MOVIMENTO.length);
    PONTOS_MOVIMENTO.forEach((p, j) => {
      v[j] = Math.hypot(xy[2 * p] - ant[2 * p], xy[2 * p + 1] - ant[2 * p + 1]);
    });
    this.vel.push(v);
    if (k >= 10) this.fps = 10 / Math.max(this.t[k] - this.t[k - 10], 1e-6);

    if (this.estado === AGUARDANDO || this.estado === CALIBRANDO) this._calibrar(k);
    else if (this.estado !== FINALIZADO) this._acompanhar(k);
    return this.status();
  }

  _calibrar(k) {
    const n = quadros(CALIB_S, this.fps);
    if (k + 1 < n) return;
    const ids = []; for (let i = k + 1 - n; i <= k; i++) ids.push(i);
    const media = (lista) => lista.reduce((s, x) => s + x, 0) / lista.length;
    const visE = media(ids.flatMap((i) => LADO_ESQ.map((p) => this.vis[i][p])));
    const visD = media(ids.flatMap((i) => LADO_DIR.map((p) => this.vis[i][p])));
    const lado = visE >= visD ? "LEFT" : "RIGHT";
    const o = OMBRO[lado], a = TORNOZELO[lado];
    const dist = ids.map((i) => Math.hypot(this.xy[i][2 * o] - this.xy[i][2 * a],
      this.xy[i][2 * o + 1] - this.xy[i][2 * a + 1]));
    const incl = mediana(ids.map((i, j) =>
      Math.abs(this.xy[i][2 * o + 1] - this.xy[i][2 * a + 1]) / (dist[j] + 1e-9)));
    const osc = Math.max(...[2 * o, 2 * o + 1, 2 * a, 2 * a + 1]
      .map((c) => desvio(ids.map((i) => this.xy[i][c]))));
    if (incl > CALIB_MAX_INCLINACAO) {
      this.estado = AGUARDANDO; this.mensagem = "Aguardando a pessoa deitar"; return;
    }
    if (osc > CALIB_MAX_OSCILACAO) {
      this.estado = CALIBRANDO; this.mensagem = "Deitada — fique parada..."; return;
    }
    const visCalib = media(ids.map((i) => Math.min(this.vis[i][o], this.vis[i][a])));
    if (mediana(dist) < COMPRIMENTO_MIN || visCalib < VIS_MIN_CALIB) {
      this.estado = AGUARDANDO;
      this.mensagem = "Corpo inteiro de lado para a câmera, mais perto"; return;
    }
    this.lado = lado;
    this.nivelDeitado = mediana(ids.map((i) => this.xy[i][2 * o + 1]));
    this.comprimento = mediana(dist);
    const tam = quadros(SUAV_VELOCIDADE_S, this.fps);
    this.ruidoCalib = PONTOS_MOVIMENTO.map((_, j) =>
      percentil(filtroMedia(ids.map((i) => this.vel[i][j]), tam), PCT_RUIDO) + 1e-9);
    this.idxRepouso = ids.slice();
    this.tremor = mediana(this.ruidoCalib);
    this.idxDeitou = k + 1;
    this.estado = PRONTO; this.mensagem = "PRONTO — pode levantar";
  }

  _altura(k) {
    return (this.nivelDeitado - this.xy[k][2 * OMBRO[this.lado] + 1]) / this.comprimento;
  }

  _movimentoProvisorio(k) {
    const n = quadros(SUAV_VELOCIDADE_S, this.fps), i0 = Math.max(0, k - n + 1);
    const z = PONTOS_MOVIMENTO.map((_, j) => {
      let s = 0; for (let i = i0; i <= k; i++) s += this.vel[i][j];
      return s / (k - i0 + 1) / this.ruidoCalib[j];
    }).sort((x, y) => x - y);
    return z[z.length - 2];
  }

  _acompanhar(k) {
    const h = this._altura(k);
    if (this.estado === EM_PE) {
      if (h < LIMIAR_DEITOU) {
        this.estado = PRONTO; this.idxDeitou = k; this.tProvisorio = null;
        this.mensagem = `PRONTO — tentativa ${this.tentativas.length + 1} de ${MAX_TENTATIVAS}`;
      }
      return;
    }
    if (this.estado === PRONTO) {
      if (h < LIMIAR_DEITOU) this.idxRepouso.push(k);
      const nPar = quadros(PARADO_S, this.fps);
      if (this._movimentoProvisorio(k) > LIMIAR_MOVIMENTO) {
        if (this.tProvisorio === null) this.tProvisorio = this.t[k];
        this.parado = 0;
      } else if (this.tProvisorio !== null) {
        this.parado += 1;
        if (this.parado >= nPar && h < LIMIAR_DEITOU) this.tProvisorio = null;
      }
      if (h >= LIMIAR_LEVANTOU) {
        this.estado = SUBINDO; this.idxCruzou = k;
        this.topo = -Infinity; this.idxTopo = k; this.chegouEmPe = false;
        if (this.tProvisorio === null) this.tProvisorio = this.t[k];
        this.mensagem = "Levantando...";
      }
      return;
    }
    if (this.estado === SUBINDO) {
      const s = -(this.xy[k][23] + this.xy[k][25]) / 2;          // média dos 2 ombros (y)
      if (s > this.topo + 0.002) { this.topo = s; this.idxTopo = k; }
      if (h >= H_EM_PE) this.chegouEmPe = true;
      const parou = this.t[k] - this.t[this.idxTopo] >= CONFIRMA_FIM_S;
      const demorou = this.t[k] - this.t[this.idxCruzou] >= SUBINDO_MAX_S;
      // app: uma pausa no meio da subida (ajoelhado/agachado) não fecha mais a tentativa;
      // uma "subida" que não chega em pé (sentou e deitou, ruído) é descartada, não vira tentativa
      if (this.chegouEmPe && parou) this._registrarTentativa(k);
      else if (!this.chegouEmPe && (h < LIMIAR_DEITOU || demorou)) {
        this._descartar(k, "não chegou em pé", h < LIMIAR_DEITOU);
      } else if (this.chegouEmPe && (h < LIMIAR_DEITOU || demorou)) this._registrarTentativa(k);
    }
  }

  _descartar(k, motivo, jaDeitado) {
    this.descartes.push({ t: this.t[k], motivo });
    this.tProvisorio = null; this.fimAnterior = Math.max(this.fimAnterior, this.idxCruzou);
    if (jaDeitado) { this.estado = PRONTO; this.idxDeitou = k; }
    else this.estado = EM_PE;
    this.mensagem = `Subida descartada (${motivo}) — ${jaDeitado ? "pode levantar" : "volte a deitar"}`;
  }

  _registrarTentativa(k) {
    const fimBuf = k + 1, J = PONTOS_MOVIMENTO.length;
    const tam = quadros(SUAV_VELOCIDADE_S, this.fps);
    // app: posições suavizadas sem atraso (média centrada; o cálculo é retroativo) antes da
    // velocidade — reduz o efeito do tremor dos pontos dos modelos mais leves no celular
    const nPos = this.tremor > LIMIAR_TREMOR ? quadros(SUAV_POSICAO_TREMOR_S, this.fps, true) : 1;
    const colunas = [];
    for (let j = 0; j < J; j++) {
      const p = PONTOS_MOVIMENTO[j], xs = new Float64Array(fimBuf), ys = new Float64Array(fimBuf);
      for (let i = 0; i < fimBuf; i++) { xs[i] = this.xy[i][2 * p]; ys[i] = this.xy[i][2 * p + 1]; }
      const sx = nPos > 1 ? filtroMedia(xs, nPos) : xs, sy = nPos > 1 ? filtroMedia(ys, nPos) : ys;
      const serie = new Float64Array(fimBuf);
      for (let i = 1; i < fimBuf; i++) serie[i] = Math.hypot(sx[i] - sx[i - 1], sy[i] - sy[i - 1]);
      serie[0] = serie[1] || 0;
      colunas.push(filtroMedia(serie, tam));
    }
    const ruido = colunas.map((c) => percentil(this.idxRepouso.map((i) => c[i]), PCT_RUIDO) + 1e-9);
    const mov = new Float64Array(fimBuf);
    for (let i = 0; i < fimBuf; i++) {
      const z = colunas.map((c, j) => c[i] / ruido[j]).sort((x, y) => x - y);
      mov[i] = z[J - 2];
    }
    const t = this.t, alvoT = t[this.idxDeitou] - RECUO_MAX_S;
    let idxMin = 0; while (idxMin < t.length && t[idxMin] < alvoT) idxMin++;   // searchsorted
    if (this.tentativas.length > 0 || this.fimAnterior >= 0) idxMin = this.idxDeitou;   // v25
    idxMin = Math.max(idxMin, this.fimAnterior + 1, 0);
    const hs = new Float64Array(fimBuf);
    for (let i = 0; i < fimBuf; i++) hs[i] = this._altura(i);
    const qDet = detectarInicio(mov, hs, this.idxCruzou, idxMin, LIMIAR_MOVIMENTO,
      quadros(PARADO_DEITADO_S, this.fps));                                          // v27
    const alt = new Float64Array(fimBuf);
    for (let i = 0; i < fimBuf; i++) alt[i] = -(this.xy[i][23] + this.xy[i][25]) / 2;   // v24
    const altura = filtroMedia(alt, quadros(SUAV_ALTURA_S, this.fps, true));
    const qFim = detectarFim(altura, qDet, this.idxCruzou, fimBuf);
    const qIni = Math.max(qDet - AJUSTE_INICIO_QUADROS, 0);                          // v24
    this.fimAnterior = qFim;
    const dur = t[qFim] - t[qIni];
    if (!(dur >= DURACAO_MIN_S && dur <= DURACAO_MAX_S)) {
      this.descartes.push({ t: t[k], motivo: `duração impossível (${dur.toFixed(2)} s)` });
      this.tProvisorio = null; this.estado = EM_PE;
      this.mensagem = `Tentativa descartada (${dur.toFixed(2).replace(".", ",")} s) — volte a deitar`;
      return;
    }
    if (qFim > qIni) {
      this.tentativas.push({
        numero: this.tentativas.length + 1, t_inicio: t[qIni], t_fim: t[qFim],
        delta_t: t[qFim] - t[qIni], quadro_inicio: qIni, quadro_fim: qFim,
        latencia_s: t[k] - t[qFim],
      });
    }
    this.tProvisorio = null;
    if (this.tentativas.length >= MAX_TENTATIVAS) {
      this.estado = FINALIZADO; this.mensagem = "FIM DO TESTE";
    } else {
      this.estado = EM_PE; this.mensagem = "Registrado — volte a deitar";
    }
  }

  encerrar() {
    if (this.estado === SUBINDO) this._registrarTentativa(this.xy.length - 1);
    return this.status();
  }

  status() {
    const agora = this.t.length ? this.t[this.t.length - 1] : 0;
    const cron = (this.estado === PRONTO || this.estado === SUBINDO) && this.tProvisorio !== null
      ? agora - this.tProvisorio : null;
    const melhor = this.tentativas.length
      ? Math.min(...this.tentativas.map((x) => x.delta_t)) : null;
    return { estado: this.estado, mensagem: this.mensagem, cronometro_s: cron,
      tentativas: this.tentativas.slice(), melhor_s: melhor, lado: this.lado,
      descartes: this.descartes.slice(), tremor: this.tremor ?? null, versao: VERSAO_ALGORITMO };
  }
}

if (typeof module !== "undefined") module.exports = { CronometroSTS, VERSAO_ALGORITMO, ajustar: (v) => { SUAV_POSICAO_TREMOR_S = v; } };

