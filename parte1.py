import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- PREPROCESAMIENTO ---
img = cv2.imread("monedas.jpg", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (20, 20), 0)

# --- ¡AQUÍ ESTÁ EL AJUSTE CLAVE! ---
# Bajamos el umbral inferior de 45 a 25 para ser más tolerantes con los bordes débiles.
# Esto debería permitir que el algoritmo de Canny conecte y cierre el círculo.
canny_edges = cv2.Canny(blur, 15, 150)

# --- EL RESTO DEL CÓDIGO FUNCIONA BIEN CON UNA MEJOR ENTRADA ---
# La dilatación ahora trabajará sobre un borde cerrado y lo rellenará correctamente.
kernel = np.ones((15,15), np.uint8)
dilated_edges = cv2.dilate(canny_edges, kernel, iterations=2)

contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# El filtro de área sigue siendo nuestra red de seguridad
min_contour_area = 5000 # Lo ajustamos un poco, ya que las formas serán más grandes
final_contours = []
for cnt in contours:
    if cv2.contourArea(cnt) > min_contour_area:
        final_contours.append(cnt)

# --- VISUALIZACIÓN ---
img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_with_contours, final_contours, -1, (0, 255, 0), 3)

plt.figure(figsize=(15, 5))
plt.subplot(131); plt.imshow(canny_edges, cmap='gray'); plt.title('1. Salida de Canny (Corregido)'); plt.axis('off')
plt.subplot(132); plt.imshow(dilated_edges, cmap='gray'); plt.title('2. Bordes Dilatados (Sólido)'); plt.axis('off')
plt.subplot(133); plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)); plt.title(f'Se encontraron {len(final_contours)} contornos finales'); plt.axis('off')
plt.show()
