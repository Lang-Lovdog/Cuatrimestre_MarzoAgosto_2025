import numpy as np
from skimage import io, color

def glrlm(image, direction, levels=256):
    """Calcula la matriz GLRLM para una dirección específica."""
    if direction == 0:
        img = image
    elif direction == 45:
        img = np.fliplr(image)
    elif direction == 90:
        img = image.T
    else:
        raise ValueError("Dirección no soportada. Usar 0, 45 o 90.")

    rows, cols = img.shape
    max_run_length = max(rows, cols)
    matrix = np.zeros((levels, max_run_length), dtype=np.int32)

    for row in img:
        run_length = 1
        for i in range(1, len(row)):
            if row[i] == row[i - 1]:
                run_length += 1
            else:
                gray = row[i - 1]
                matrix[gray, run_length - 1] += 1
                run_length = 1
        matrix[row[-1], run_length - 1] += 1
    return matrix[:, :np.max(np.nonzero(matrix)[1])+1]

def glrlm_features(glrlm):
    """Calcula SRE, LRE, GLNU, RLN y RP de una matriz GLRLM."""
    eps = 1e-12
    Ng, Nr = glrlm.shape
    total_runs = np.sum(glrlm)

    i = np.arange(1, Ng + 1).reshape(-1, 1)
    j = np.arange(1, Nr + 1).reshape(1, -1)

    SRE = np.sum(glrlm / (j ** 2 + eps)) / (total_runs + eps)
    LRE = np.sum(glrlm * (j ** 2)) / (total_runs + eps)
    GLNU = np.sum((np.sum(glrlm, axis=1) ** 2)) / (total_runs + eps)
    RLN = np.sum((np.sum(glrlm, axis=0) ** 2)) / (total_runs + eps)
    RP = total_runs / (glrlm.shape[0] * glrlm.shape[1] + eps)

    return [SRE, LRE, GLNU, RLN, RP]

def process_image(image_path):
    image = io.imread(image_path)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    image = (image * 255).astype(np.uint8)

    directions = [0, 45, 90]
    features_all = []

    for d in directions:
        matrix = glrlm(image, d)
        features = glrlm_features(matrix)
        features_all.extend(features)

    return np.array(features_all)

#Cargar imagen
image_path = r"C:\Users\CuentaTemporal\Documents\Maestria IE\Cuarto cuatri\Reconocimiento de patrones\archive\coronary_artery_disease\coronary_artery_disease\coronary_artery  (1).jpe"  
features = process_image(image_path)

#Mostrar características
print("Vector de características GLRLM (15 valores):")
print(features)
print(f"Dimensión: {features.shape}")

