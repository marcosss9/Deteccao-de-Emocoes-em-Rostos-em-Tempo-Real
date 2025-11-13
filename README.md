## Detecção de Emoções em Rostos em Tempo Real

---

Este projeto é um aplicativo web simples construído com **Streamlit** que utiliza a biblioteca **DeepFace** para realizar a detecção de emoções em faces, tanto por meio de *upload* de imagens quanto por *stream* de vídeo da webcam em tempo real.

---

## Funcionalidades Principais

* **Análise de Imagem:** Permite fazer *upload* de uma imagem para detectar e rotular a emoção predominante em cada rosto.
* **Webcam em Tempo Real:** Inicia a câmera do dispositivo para análise contínua das emoções detectadas.
* **Visualização Clara:** Exibe a emoção dominante com sua porcentagem de confiança (entre 0% e 100%) diretamente sobre o rosto detectado (na imagem ou no vídeo).
* **Tratamento de Erros:** Utiliza `enforce_detection=True` para garantir que o programa só retorne resultados de emoção quando uma face for de fato detectada, evitando resultados incorretos em imagens sem rostos.

---

##  Instalação e Execução

### Dependências

Instale as bibliotecas necessárias usando `pip`:

```bash
pip install streamlit opencv-python numpy Pillow deepface tf-keras
