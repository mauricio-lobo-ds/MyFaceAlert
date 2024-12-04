import cv2
from deepface import DeepFace
from pygame import mixer
import threading

# Inicializa o mixer para reprodução de sons
mixer.init()
ALERT_SOUND = "alert_sound.mp3"  # Caminho do som de alerta

def play_alert():
    """Reproduz o som de alerta em uma thread separada."""
    mixer.music.load(ALERT_SOUND)
    mixer.music.play()

def detect_faces_with_opencv(frame):
    """Detecta rostos no frame usando OpenCV."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detect_emotions():
    """Detecta emoções em tempo real usando a webcam."""
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    print("Pressione 'q' para sair.")
    CONFIDENCE_THRESHOLD = 60  # Ajuste de sensibilidade (em %)
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Erro ao capturar o frame.")
            break

        # Detecta rostos no frame
        faces = detect_faces_with_opencv(frame)
        if len(faces) > 0:
            try:
                # Analisa emoções no primeiro rosto detectado
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                emotions = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']
                confidence = emotions[dominant_emotion]

                # Apenas considere se a confiança for maior que o limiar
                if confidence >= CONFIDENCE_THRESHOLD:
                    print(f"Emoção detectada: {dominant_emotion} ({confidence:.2f}%)")

                    # Emite alerta se detectar emoções relacionadas a desânimo
                    if dominant_emotion in ['sad', 'fear', 'disgust']:
                        print("Desânimo detectado! Emitindo alerta...")
                        threading.Thread(target=play_alert).start()

                    # Exibe o frame com a emoção detectada
                    cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f}%)", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    print(f"Emoção detectada ({dominant_emotion}) abaixo do limiar de confiança ({confidence:.2f}%). Ignorada.")
            except Exception as e:
                print(f"Erro durante a análise: {e}")

        # Exibe o vídeo em tempo real
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("Detecção de Emoções", frame)

        # Sai do loop ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera a webcam e fecha as janelas
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detect_emotions()
    except KeyboardInterrupt:
        print("\nAplicativo encerrado pelo usuário.")
