# --------------------Importamos
import cv2
import os
import mediapipe as mp
from DataBase import direccion

# --------------------Importamos los nombres de las carpetas
direccion = 'D:/ClassTwo/DeteccionRostrosTwo/Fotos'
etiquetas = os.listdir(direccion)
print("Nombre: ", etiquetas)

# Inicializamos el modelo entrenado
modelo = cv2.face.LBPHFaceRecognizer_create()
modelo.read('ModeloEntrenado.xml')

# Inicialización del detector de rostros
detector = mp.solutions.face_detection
dibujo = mp.solutions.drawing_utils

# Inicializamos la captura de video
cap = cv2.VideoCapture(0)
umbral_confianza = 60  # Ajusta el umbral para tu caso

# Parámetros de detección de rostros
with detector.FaceDetection(min_detection_confidence=0.90) as rostros:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = rostros.process(rgb)

        if resultado.detections is not None:
            for rostro in resultado.detections:
                dibujo.draw_detection(frame, rostro, dibujo.DrawingSpec(color=(0, 255, 0)))

                # Extracción de coordenadas
                al, an, c = frame.shape
                x = rostro.location_data.relative_bounding_box.xmin
                y = rostro.location_data.relative_bounding_box.ymin
                ancho = rostro.location_data.relative_bounding_box.width
                alto = rostro.location_data.relative_bounding_box.height

                xi, yi = int(x * an), int(y * al)
                xf, yf = int((x + ancho) * an), int((y + alto) * al)

                if xi >= 0 and yi >= 0 and xf <= an and yf <= al:
                    cara = rgb[yi:yf, xi:xf]
                    cara = cv2.resize(cara, (150, 200), interpolation=cv2.INTER_CUBIC)
                    cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)

                    # Realizar predicción
                    prediccion = modelo.predict(cara)

                    # Mostrar los resultados en pantalla
                    nombre = 'Desconocido'
                    color = (0, 0, 255)
                    if prediccion[1] < umbral_confianza and prediccion[0] < len(etiquetas):
                        nombre = etiquetas[prediccion[0]]
                        color = (0, 255, 0)  # Cambia el color si es conocido

                    cv2.putText(frame, nombre, (xi, yi - 5), 1, 1.3, color, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi, yi), (xf, yf), color, 2)

        cv2.imshow("Reconocimiento Facial y de Tapabocas", frame)
        t = cv2.waitKey(1)
        if t == 27:
            break

cap.release()
cv2.destroyAllWindows()





