import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- LÓGICA DE AGRUPACIÓN ---
def filtrar_por_agrupacion(candidatos_rects):
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
            
            # Mismas reglas heurísticas
            if diff_h < 0.4 and diff_y < 0.5 and -5 < distancia_x < (w * 2.5):
                grupo.append(rect)
                agregado = True
                break
        if not agregado: grupos.append([rect])

    # Filtramos grupos muy chicos y elegimos el mejor
    grupos_validos = [g for g in grupos if len(g) >= 3]
    if not grupos_validos: return []
    
    mejor_grupo = sorted(grupos_validos, key=lambda g: (len(g), g[0][2]*g[0][3]), reverse=True)[0]
    return mejor_grupo

# --- FUNCIÓN DE RECORTE ---
def obtener_recorte_patente(ruta_imagen):
    img_patente = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img_patente is None: return np.zeros((50,150), dtype=np.uint8)

    _, img_binary = cv2.threshold(img_patente, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_invertida = cv2.bitwise_not(img_binary)
    contornos, _ = cv2.findContours(img_invertida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos_crudos = []
    
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0:
            aspect_ratio = float(h) / w
            area = w * h
            if 0.8 <= aspect_ratio <= 5.0 and 30 < area < 5000: 
                candidatos_crudos.append((x, y, w, h))

    # Obtenemos el grupo ganador
    grupo = filtrar_por_agrupacion(candidatos_crudos)
    
    if grupo:
        # Calculamos los límites del grupo entero
        min_x = min([r[0] for r in grupo])
        min_y = min([r[1] for r in grupo])
        max_x = max([r[0]+r[2] for r in grupo])
        max_y = max([r[1]+r[3] for r in grupo])
        
        # Agregamos un margen (padding) para que se vea prolijo
        pad = 15
        h_img, w_img = img_patente.shape
        crop_x = max(0, min_x - pad)
        crop_y = max(0, min_y - pad)
        crop_w = min(w_img, max_x + pad)
        crop_h = min(h_img, max_y + pad)
        
        # Recortamos la imagen original
        recorte = img_patente[crop_y:crop_h, crop_x:crop_w]
        return recorte, f"Detectado ({len(grupo)} chars)"
    else:
        # Si no detecta nada, devuelve imagen negra
        return np.zeros((50, 150), dtype=np.uint8), "No Detectado"

# --- VISUALIZACIÓN ---
rows = 3
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 6)) # Figura más ancha y baja para ver detalles
axes = axes.flatten()

print("Generando recortes de las patentes detectadas...")

for i in range(1, 13):
    nombre = f"img{i:02d}.png"
    recorte, estado = obtener_recorte_patente(nombre)
    
    ax = axes[i-1]
    ax.imshow(recorte, cmap="gray")
    ax.set_title(f"{nombre}\n{estado}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()


























































