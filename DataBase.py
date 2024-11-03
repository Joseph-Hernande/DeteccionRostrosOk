import cv2
import mediapipe as mp
import os
import warnings

# Ignoramos las advertencias
warnings.filterwarnings("ignore")

# Creación de carpeta donde estarán las fotos
nombre = 'Joseph Tapabocas'
direccion = 'D:/ClassTwo/DeteccionRostrosTwo/Fotos'
carpeta = os.path.join(direccion, nombre)

if not os.path.exists(carpeta):
    print("Carpeta creada")
    os.makedirs(carpeta)

cont = 0

# Inicialización del detector de rostros
detector = mp.solutions.face_detection
dibujo = mp.solutions.drawing_utils

# Inicializamos la captura de video
cap = cv2.VideoCapture(0)

# Parámetros de detección de rostros
with detector.FaceDetection(min_detection_confidence=0.75) as rostros:
    while True:
        # Lectura de fotogramas
        ret, frame = cap.read()
        if not ret:
            break

        # Invertimos la cámara para efecto espejo
        frame = cv2.flip(frame, 1)

        # Conversión de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detección de rostros
        resultado = rostros.process(rgb)

        # Verificamos si se detectaron rostros
        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame, rostro, dibujo.DrawingSpec(color=(0, 255, 0)))

                # Extracción de coordenadas
                al, an, c = frame.shape
                x = rostro.location_data.relative_bounding_box.xmin
                y = rostro.location_data.relative_bounding_box.ymin
                ancho = rostro.location_data.relative_bounding_box.width
                alto = rostro.location_data.relative_bounding_box.height

                # Conversión de coordenadas a pixeles
                xi, yi = int(x * an), int(y * al)
                xf, yf = int((x + ancho) * an), int((y + alto) * al)

                # Verificación de los límites de la imagen para evitar errores
                if xi >= 0 and yi >= 0 and xf <= an and yf <= al:
                    cara = frame[yi:yf, xi:xf]
                    cv2.imshow("Reconocimiento Facial y de Tapabocas", cara)
                else:
                    print("Coordenadas fuera de los límites de la imagen.")

                #redimensionar las fotos
                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)

                #Guardamos las fotos de nuestro rostro
                cv2.imwrite(carpeta + "/rostro_{}.jpg".format(cont), cara)
                cont = cont + 1

        # Salir al presionar ESC
        t = cv2.waitKey(1)
        if t == 27 or cont > 1000:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
