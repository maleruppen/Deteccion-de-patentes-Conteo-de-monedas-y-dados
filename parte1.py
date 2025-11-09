import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PREPROCESAMIENTO ---
img = cv2.imread("monedas.jpg", cv2.IMREAD_GRAYSCALE)
# Para HoughCircles, un desenfoque de mediana suele ser muy bueno para preservar los bordes.
blur = cv2.medianBlur(img, 9)

# --- 2. DETECCIÓN DE CÍRCULOS CON HOUGH ---
# Esta es la parte de prueba y error. Estos parámetros son un buen punto de partida.
#   - dp: Relación inversa de resolución. Siempre 1.
#   - minDist: Distancia mínima entre centros de círculos detectados.
#   - param1: Umbral superior para el detector Canny interno de Hough.
#   - param2: Umbral de "votos". Cuanto más bajo, más círculos (incluso falsos) detectará.
#   - minRadius/maxRadius: Rango de radios de los círculos a buscar.
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=150, param2=40,
                           minRadius=80, maxRadius=250)

# --- 3. VISUALIZACIÓN ---
img_with_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Asegurarse de que se encontraron círculos antes de procesarlos
if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Se encontraron {len(circles[0, :])} monedas con HoughCircles.")
    
    # Dibujar los círculos encontrados
    for i in circles[0, :]:
        # Dibujar el contorno del círculo
        cv2.circle(img_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 3)
        # Dibujar el centro del círculo
        cv2.circle(img_with_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
else:
    print("No se encontraron monedas con los parámetros actuales.")

# Mostrar el resultado
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
plt.title(f'Detección con HoughCircles')
plt.axis('off')
plt.show()