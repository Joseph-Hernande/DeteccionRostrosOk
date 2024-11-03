# ----------------------Importamos librerías
import cv2
import numpy as np
import os
from DataBase import direccion

# ----------------------Importamos las fotos tomadas anteriormente
direccion = 'D:/ClassTwo/DeteccionRostrosTwo/Fotos'
lista = os.listdir(direccion)

etiquetas = []
rostros = []
cont = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir
    for fileName in os.listdir(nombre):
        etiquetas.append(cont)
        # Leer imagen en escala de grises
        imagen = cv2.imread(nombre + '/' + fileName, 0)
        rostros.append(imagen)
    cont += 1

# -----------------------Creamos y entrenamos el modelo
reconocimiento = cv2.face.LBPHFaceRecognizer_create()
reconocimiento.train(rostros, np.array(etiquetas))

# -----------------------Guardamos el modelo
reconocimiento.write("ModeloEntrenado.xml")
print("Modelo Creado")


