import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---  PREPROCESAMIENTO ---
img = cv2.imread("monedas.jpg", cv2.IMREAD_GRAYSCALE)
# Para HoughCircles, un desenfoque de mediana suele ser muy bueno para preservar los bordes.
blur = cv2.medianBlur(img, 9)

# ---  DETECCIÓN DE CÍRCULOS CON HOUGH ---
#   - dp: Relación inversa de resolución. Siempre 1.
#   - minDist: Distancia mínima entre centros de círculos detectados.
#   - param1: Umbral superior para el detector Canny interno de Hough.
#   - param2: Umbral de "votos". Cuanto más bajo, más círculos (incluso falsos) detectará.
#   - minRadius/maxRadius: Rango de radios de los círculos a buscar.
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=150, param2=40,
                           minRadius=80, maxRadius=250)

# ---  VISUALIZACIÓN ---
img_with_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Asegurarse de que se encontraron círculos antes de procesarlos
if circles is not None:
    circles = np.uint16(np.around(circles))
    #print(f"Se encontraron {len(circles[0, :])} monedas con HoughCircles.")
    
    # Dibujar los círculos encontrados
    for i in circles[0, :]:
        # Dibujar el contorno del círculo
        cv2.circle(img_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 3)
        # Dibujar el centro del círculo
        cv2.circle(img_with_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
else:
    print("No se encontraron monedas con los parámetros actuales.")


if circles is not None:
    circles = np.uint16(np.around(circles))
    radios = circles[0, :, 2]
    small_limit = 145   # límite superior para monedas pequeñas (10 cent)
    large_limit = 170  # límite superior para monedas medianas (1 peso)


    categorias = {
        "Pequeñas (10¢)": [],
        "Medianas (1 peso)": [],
        "Grandes (50¢)": []
    }

    for i in circles[0, :]:
        r = i[2]
        if r < small_limit:
            categorias["Pequeñas (10¢)"].append(i)
        elif  small_limit < r < large_limit:
            categorias["Medianas (1 peso)"].append(i)
        else:
            categorias["Grandes (50¢)"].append(i)


    # --- VISUALIZACIÓN CON COLORES POR CLASE ---
    colores = {
        "Pequeñas (10¢)": (255, 0, 0),    # Azul
        "Medianas (1 peso)": (0, 255, 0),    # Verde
        "Grandes (50¢)": (0, 0, 255)         # Rojo
    }

    img_clasif = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for tipo, lista in categorias.items():
        for i in lista:
            cv2.circle(img_clasif, (i[0], i[1]), i[2], colores[tipo], 3)
            cv2.circle(img_clasif, (i[0], i[1]), 2, (0, 0, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_clasif, cv2.COLOR_BGR2RGB))
    plt.title('Clasificación de monedas según tamaño (ajustada)')
    plt.axis('off')
    plt.show()

else:
    print("No se detectaron monedas con los parámetros actuales.")

# Umbral 
thresh_value = 170
_, mask_dados = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

# Buscar contornos
contornos, _ = cv2.findContours(mask_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ---- PARÁMETROS DE FILTRADO ----
min_area_cara = 35000      # Un área mínima más segura que 500
epsilon_perc = 0.04       # CAMBIO 1: 4% (más flexible que 0.02)
min_aspect = 0.7          # CAMBIO 2: Rango más amplio
max_aspect = 1.3          # CAMBIO 2: Rango más amplio


dados_detectados = 0
# Iterar sobre los contornos
for cnt in contornos:
    area = cv2.contourArea(cnt)
    
    # Filtro de área (para ruido)
    if area > min_area_cara:
        
        approx = cv2.approxPolyDP(cnt, epsilon_perc * cv2.arcLength(cnt, True), True)
        num_vertices = len(approx)
        
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = 1.0 # Valor default
        if h > 0: # Evitar división por cero
            aspect_ratio = float(w) / h
            

        # Cuadrilatero
        if num_vertices == 4:
            
            if min_aspect <= aspect_ratio <= max_aspect:
                cv2.drawContours(img_color, [approx], 0, (0, 0, 255), 4)
                dados_detectados += 1
            else:
                continue
        else:
            continue
print(f"Cantidad de dados detectados: {dados_detectados}")


plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Detección con Filtros Flexibles")
plt.show()


## Puntos por cada dado
# Usamos la imagen gris original para crear un umbral que aisle los puntos oscuros.

_, mask_puntos = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
blur_puntos = cv2.medianBlur(mask_puntos, 5) # Peque帽o desenfoque

# re-iterar sobre los contornos originales
# y aplicar los filtros de deteccion de dados nuevamente, para luego contar los puntos.

img_puntos_contados = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
total_puntos_contados = 0
dados_encontrados_info = [] # Lista para guardar los Bounding Boxes de los dados

# Reutilizamos las variables definidas arriba: contornos, min_area_cara, epsilon_perc, min_aspect, max_aspect

for cnt in contornos:
    area = cv2.contourArea(cnt)
    
    if area > min_area_cara:
        approx = cv2.approxPolyDP(cnt, epsilon_perc * cv2.arcLength(cnt, True), True)
        num_vertices = len(approx)
        
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 1.0
            
        # Si cumple los filtros de DADO:
        if num_vertices == 4 and min_aspect <= aspect_ratio <= max_aspect:
            dados_encontrados_info.append((x, y, w, h))

# ---- Contar los puntos dentro de cada dado detectado ----

for idx, (x, y, w, h) in enumerate(dados_encontrados_info):
    
    # definimos el ROI
    roi_puntos = blur_puntos[y:y+h, x:x+w]
    
    # detectamos los circulos
    puntos_circles = cv2.HoughCircles(roi_puntos, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                      param1=50, param2=15, 
                                      minRadius=5, maxRadius=30)
    
    num_puntos = 0
    if puntos_circles is not None:
        puntos_circles = np.uint16(np.around(puntos_circles))
        num_puntos = len(puntos_circles[0, :])
        total_puntos_contados += num_puntos
        
        # Opcional: Dibujar los puntos detectados
        for pt in puntos_circles[0, :]:
            center_x, center_y = pt[0] + x, pt[1] + y 
            cv2.circle(img_puntos_contados, (center_x, center_y), pt[2], (255, 0, 0), 2) # Azul
            cv2.circle(img_puntos_contados, (center_x, center_y), 1, (0, 255, 255), 2)  # Centro
            
    # Dibujar contorno del dado para referencia
    cv2.rectangle(img_puntos_contados, (x, y), (x + w, y + h), (0, 0, 255), 3)
    # Escribir el número de puntos detectados en cada dado
    texto = f"{num_puntos}"
    posicion_texto = (x + 10, y + 50)
    cv2.putText(img_puntos_contados, texto, posicion_texto, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_puntos_contados, cv2.COLOR_BGR2RGB))
plt.title(f"Deteccion de Puntos en Dados (Total: {total_puntos_contados})")
plt.axis('off')
plt.show()



