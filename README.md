# STS-Vision

Cronometragem automática do **Supine-to-Stand Test (STS)** a partir de vídeo, por estimativa de pose
(MediaPipe Pose, 33 pontos). Código do artigo *"STS-Vision: desenvolvimento e validação independente de
um algoritmo de estimativa de pose para cronometrar o Supine-to-Stand Test em adolescentes"*
(Martins-Guimarães; Barros).

**Aplicativo (abre no navegador do celular):** https://tumartins.github.io/sts-vision/

## Conteúdo

| Pasta | O que tem |
|---|---|
| `index.html` | Aplicativo web (câmera em tempo real ou vídeo da galeria), algoritmo v27. As imagens são processadas no próprio aparelho e não são gravadas nem enviadas. |
| `app/sts_core.js` | Núcleo do cronômetro em tempo real (JavaScript), regras da v27. |
| `app/reproduzir_dados.js` | Refaz, no computador (Node.js), uma análise feita no celular a partir do arquivo JSON exportado pelo app. |
| `algoritmo/` | Versões congeladas do algoritmo offline em Python (v23 a v27). A v27 é a versão final. |
| `validacao/` | Scripts de sorteio, anotação às cegas (Google Colab) e análise dos testes independentes. |

## Versões congeladas (SHA-256)

| Arquivo | SHA-256 |
|---|---|
| `processar_video_sts_v23.py` | `cf09b10c8cb4f1168b780d1abc5a57fad66c9b4ff0d9ecf741cbbc9d46256efe` |
| `processar_video_sts_v24.py` | `8fc8796999db2871339eb554ff195bdfa50c39355a65119459a91f75f38a733b` |
| `processar_video_sts_v25.py` | `2d3c0b1f2f748e68e97023440c904e2f84da4072cdd6aba954bcb92eae429254` |
| `processar_video_sts_v26.py` | `75ab8ff0814ce36d0fced83cb2d348b12d9b0a75138239e86f2ffd08cbb80296` |
| `processar_video_sts_v27.py` | `8b62c9665a8221ca82c7524fe58f5ca05720d0c552b3d2924bbb9f6fcef6650e` |
| `index.html` (app usado no teste final) | `f2399144d5b97fe1fd471082965b85245579c2f019a587d3cc89d48bc45b83b4` |

## Uso rápido (Python)

```python
# pip install mediapipe opencv-python scipy numpy
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=vision.RunningMode.VIDEO)
exec(open("algoritmo/processar_video_sts_v27.py").read())
resultado = processar_video_sts_v27("meu_video.mp4")
print(resultado["tentativas"], resultado["melhor_tempo"])
```

Filmagem: celular fixo, plano sagital, corpo inteiro no quadro, 30 quadros/s; o participante começa
deitado e parado e volta a deitar entre as tentativas.

## Dados

Os vídeos e os tempos dos participantes do Projeto Atitude não estão no repositório (aprovação
CEP-HEMOPE, parecer nº 4.449.705). Os IDs que aparecem nos scripts de validação são códigos do estudo.

## Licença

MIT
