import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import time

if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

st.set_page_config(layout="wide", page_title="Detecção de Emoções")
st.title("Detecção de Emoções em Rostos")
st.markdown("---")

def detect_emotion(image):
    try:
        results = DeepFace.analyze(image, actions=['emotion'], enforce_detection=True, detector_backend='opencv')

        for result in results:
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
            emotion = result['dominant_emotion']
            confidence = float(result['emotion'][emotion])

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text_label = f"{emotion.capitalize()} ({confidence:.1f}%)"
            cv2.putText(image, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image, results
    except (ValueError, AttributeError) as e:
        return image, [{"dominant_emotion": "Nenhuma face detectada", "emotion": {"angry": 0.0, "fear": 0.0, "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 0.0, "disgust": 0.0}}]
    except Exception as e:
        return image, [{"dominant_emotion": "Erro no processamento", "emotion": {"angry": 0.0, "fear": 0.0, "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 0.0, "disgust": 0.0}}]

def webcam_stream():
    if st.session_state.webcam_active:
        frame_placeholder = st.empty()
        text_placeholder = st.empty()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Erro: Não foi possível acessar a webcam (câmera 0). Verifique as permissões.")
            st.session_state.webcam_active = False
            return

        while st.session_state.webcam_active:
            ret, frame = cap.read()

            if not ret:
                text_placeholder.error("Erro ao capturar frame. A webcam pode ter sido desconectada.")
                break

            frame = cv2.flip(frame, 1)

            frame_processed, emotions = detect_emotion(frame)
            frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame_rgb, caption="Vídeo em Tempo Real", use_container_width=True)

            if emotions and emotions[0]['dominant_emotion'] != "Nenhuma face detectada" and emotions[0]['dominant_emotion'] != "Erro no processamento":
                emotions_text = ", ".join([f"{e['dominant_emotion'].capitalize()} ({float(e['emotion'][e['dominant_emotion']]):.1f}%)" for e in emotions if 'dominant_emotion' in e and 'emotion' in e])
                text_placeholder.info(f"Emoções detectadas: **{emotions_text}**")
            else:
                text_placeholder.warning("Nenhuma face detectada.")

            time.sleep(0.01)

        cap.release()
        frame_placeholder.empty()
        text_placeholder.info("A captura de vídeo foi parada.")


option = st.selectbox("Escolha a entrada", ["Imagem", "Webcam"])

if option == "Imagem":
    st.header("Análise de Imagem")
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            img_processed, emotions = detect_emotion(img_cv.copy())
            img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption="Resultado da Análise", use_container_width=True)

            st.subheader("Resultados da Emoção:")
            
            if emotions and emotions[0]['dominant_emotion'] == "Nenhuma face detectada":
                st.info("Nenhuma face detectada na imagem.")
            elif emotions and emotions[0]['dominant_emotion'] == "Erro no processamento":
                 st.error("Ocorreu um erro no processamento da imagem.")
            else:
                for i, emotion in enumerate(emotions):
                    dominant_emotion_name = emotion['dominant_emotion'].capitalize()
                    dominant_emotion_confidence = float(emotion['emotion'][emotion['dominant_emotion']])
                    
                    st.success(f"Rosto {i+1}: Emoção predominante: **{dominant_emotion_name}** ({dominant_emotion_confidence:.1f}%)")

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar a imagem: {e}")

elif option == "Webcam":
    st.header("Captura e Análise em Tempo Real (Webcam)")

    if st.session_state.webcam_active:
        button_label = "Parar Captura"
        button_type = "secondary"
    else:
        button_label = "Iniciar Captura"
        button_type = "primary"

    if st.button(button_label, type=button_type):
        st.session_state.webcam_active = not st.session_state.webcam_active
        st.rerun()

    if st.session_state.webcam_active:
        webcam_stream()
    else:
        st.info("Clique em 'Iniciar Captura' para começar.")