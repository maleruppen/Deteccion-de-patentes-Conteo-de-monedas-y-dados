import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- LÓGICA DE AGRUPACIÓN (Idéntica) ---
def filtrar_por_agrupacion(candidatos_rects) -> list: 
    if not candidatos_rects: return []
    candidatos_ordenados = sorted(candidatos_rects, key=lambda r: r[0])
    grupos = []
    
    for rect in candidatos_ordenados:
        x, y, w, h = rect
        agregado = False
        for grupo in grupos:
            last_x, last_y, last_w, last_h = grupo[-1]
            diff_h = abs(h - last_h) / float(max(h, last_h))
            centro_y_actual = y + h/2
            centro_y_last = last_y + last_h/2
            diff_y = abs(centro_y_actual - centro_y_last) / float(max(h, last_h))
            distancia_x = x - (last_x + last_w)
            
            if diff_h < 0.6 and diff_y < 0.4 and -5 < distancia_x < (w * 2.3):
                grupo.append(rect)
                agregado = True
                break
        if not agregado: grupos.append([rect])

    grupos_validos = [g for g in grupos if len(g) >= 3]
    if not grupos_validos: return []
    mejor_grupo = sorted(grupos_validos, key=lambda g: (len(g), g[0][2]*g[0][3]), reverse=True)[0]
    return mejor_grupo

# --- PROCESAMIENTO (Genera las 3 imágenes para los gráficos) ---
def procesar_patente_completo(ruta_imagen):
    img_gray = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None, None, None, "Error de carga"

    # --- PASO 1: BINARIZADO (Para Figura 1) ---
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_invertida = cv2.bitwise_not(img_binary)

    # --- PASO 2: CONTOURNOS/CANDIDATOS (Para Figura 2) ---
    contornos, _ = cv2.findContours(img_invertida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidatos_crudos = []
    img_candidatos = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) # Copia a color para dibujar azul
    
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0:
            aspect_ratio = float(h) / w
            area = w * h
            if 0.8 <= aspect_ratio <= 5.0 and 30 < area < 5000: 
                candidatos_crudos.append((x, y, w, h))
                # Dibujamos candidato en AZUL sobre la imagen completa
                cv2.rectangle(img_candidatos, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # --- PASO 3: RECORTE FINAL (Para Figura 3 - Estilo Original) ---
    grupo = filtrar_por_agrupacion(candidatos_crudos)
    
    if grupo:
        # Calculamos crop
        min_x = min([r[0] for r in grupo])
        min_y = min([r[1] for r in grupo])
        max_x = max([r[0]+r[2] for r in grupo])
        max_y = max([r[1]+r[3] for r in grupo])
        
        pad = 15
        h_img, w_img = img_gray.shape
        crop_x = max(0, min_x - pad)
        crop_y = max(0, min_y - pad)
        crop_w = min(w_img, max_x + pad)
        crop_h = min(h_img, max_y + pad)
        
        recorte = img_gray[crop_y:crop_h, crop_x:crop_w]
        recorte_color = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)

        # Dibujamos el grupo final en VERDE sobre el recorte
        grupo_ajustado = [(x - crop_x, y - crop_y, w, h) for (x, y, w, h) in grupo]
        for (x, y, w, h) in grupo_ajustado:
            cv2.rectangle(recorte_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
        estado = f"Detectado ({len(grupo)})"
    else:
        # Si falla, imagen negra
        recorte_color = np.zeros((50, 150, 3), dtype=np.uint8)
        estado = "No Detectado"

    return img_invertida, img_candidatos, recorte_color, estado

# --- ALMACENAMIENTO ---
datos_graficos = []
for i in range(1, 13):
    nombre = f"img{i:02d}.png"
    # Obtenemos las 3 versiones de la imagen
    img_bin, img_cand, img_final, estado = procesar_patente_completo(nombre)
    if img_bin is not None:
        datos_graficos.append((nombre, img_bin, img_cand, img_final, estado))

# --- VISUALIZACIÓN ---
rows, cols = 3, 4

# 1. GRÁFICO DE BINARIZACIÓN (Imagen completa)
fig1, axes1 = plt.subplots(rows, cols, figsize=(16, 8))
fig1.suptitle("Paso 1: Binarización y Thresholding", fontsize=14)
axes1 = axes1.flatten()

for idx, ax in enumerate(axes1):
    if idx < len(datos_graficos):
        nombre, img_bin, _, _, _ = datos_graficos[idx]
        ax.imshow(img_bin, cmap='gray')
        ax.set_title(nombre, fontsize=9)
    ax.axis('off')

# 2. GRÁFICO DE CONTORNOS (Imagen completa con candidatos azules)
fig2, axes2 = plt.subplots(rows, cols, figsize=(16, 8))
fig2.suptitle("Paso 2: Contornos Candidatos (Filtro de Area/Forma)", fontsize=14)
axes2 = axes2.flatten()

for idx, ax in enumerate(axes2):
    if idx < len(datos_graficos):
        nombre, _, img_cand, _, _ = datos_graficos[idx]
        # Convertir BGR a RGB
        ax.imshow(cv2.cvtColor(img_cand, cv2.COLOR_BGR2RGB))
        ax.set_title(nombre, fontsize=9)
    ax.axis('off')

# 3. GRÁFICO FINAL (Tu estilo original: Recorte + Rectángulos Verdes)
fig3, axes3 = plt.subplots(rows, cols, figsize=(16, 6)) # Tamaño original
fig3.suptitle("Paso 3: Resultado Final (Agrupación)", fontsize=14)
axes3 = axes3.flatten()

for idx, ax in enumerate(axes3):
    if idx < len(datos_graficos):
        nombre, _, _, img_final, estado = datos_graficos[idx]
        # Convertir BGR a RGB
        ax.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{nombre}\n{estado}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()