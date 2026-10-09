// node reproduzir_dados.js sts_dados_X.json  -> refaz a análise do celular, quadro a quadro
const fs = require("fs"); const { CronometroSTS } = require("./sts_core.js");
const d = JSON.parse(fs.readFileSync(process.argv[2]));
const c = new CronometroSTS(d.largura, d.altura); let ultimo = "";
for (const [t, p] of d.quadros) {
  const st = c.processar(t, p);
  if (st.estado !== ultimo) { console.log(t.toFixed(2), st.estado, c.lado ? c._altura(c.xy.length - 1).toFixed(2) : "", st.mensagem); ultimo = st.estado; }
}
const st = c.encerrar();
console.log("celular:", d.tentativas.map((x) => x.delta_t.toFixed(3)), "| refeito:", st.tentativas.map((x) => x.delta_t.toFixed(3)), "| descartes:", st.descartes, "| tremor:", st.tremor);
