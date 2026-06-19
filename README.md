## Sistema de Diagnóstico para la Detección de Caries Dentales por modelo YOLO

## Integrante
** Vanessa Yazmin Campos Guzmán  6F  23310324

---

## Descripción del Proyecto
Este proyecto implementa un modelo de visión artificial basado en la arquitectura **YOLOv8 (You Only Look Once)** entrenado específicamente para la localización y clasificación automatizada de piezas dentales sanas y lesiones de caries en imágenes intraorales. El objetivo es proporcionar a los profesionales de la salud dental una herramienta de diagnóstico rápido que reduzca el margen de error humano y optimice los tiempos de consulta.

---

## Estructura del Repositorio
* `Entrenamiento_Caries_YOLO.ipynb`: Cuaderno de Google Colab con el pipeline completo de carga de datos, entrenamiento y scripts de inferencia.
* `best.pt`: Pesos finales optimizados del modelo neuronal tras 25 épocas de entrenamiento.
* `results.csv` y `confusion_matrix.png`: Gráficas y métricas de rendimiento del entrenamiento (Precisión, Exhaustividad y Pérdida).
* `requirements.txt`: Archivo de dependencias para la replicación del entorno local.
* `muestras/`: Carpeta de evidencias visuales que contiene imágenes de prueba procesadas por el modelo (prueba2, prueba3, prueba4.jpg).

---
##Ejecución de la Inferencia (Prueba)
Para poner a prueba el modelo con nuevas imágenes intraorales, ejecuta el script de inferencia incluido en el notebook principal:

from ultralytics import YOLO
import cv2
model = YOLO("best.pt")
results = model.predict(source="ruta_de_tu_imagen.jpg", conf=0.4)
for r in results:
    im_bgr = r.plot()
    cv2.imshow("Diagnóstico IA", im_bgr)
    cv2.waitKey(0)
    
## Instrucciones de Ejecución

### Requisitos Previos
Tener instalado Python 3.8 o superior y las dependencias del proyecto. Puedes instalarlas ejecutando:
```bash
pip install -r requirements.txt

