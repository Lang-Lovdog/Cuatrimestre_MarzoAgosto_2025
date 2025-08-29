import os
import sys
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops

encabezados = []
def calculate_entropy(glcm):
    """Calcula la entropía de una matriz GLCM."""
    glcm_norm = glcm / np.sum(glcm)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.nansum(glcm_norm * np.log2(glcm_norm + 1e-12))
    return entropy

def calculate_variance(glcm):
    """Calcula la varianza de una matriz GLCM."""
    glcm_norm = glcm / np.sum(glcm)
    mean = np.sum(glcm_norm * np.arange(glcm.shape[0])[:, None])
    variance = np.sum(glcm_norm * (np.arange(glcm.shape[0])[:, None] - mean) ** 2)
    return variance

def calculate_idf(glcm):
    """Calcula el momento de diferencia inversa (IDF)."""
    glcm_norm = glcm / np.sum(glcm)
    i, j = np.indices(glcm.shape)
    return np.sum(glcm_norm / (1 + (i - j) ** 2))

def extract_features(image, distances=[1, 3, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extrae las 7 características para cada GLCM."""
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    image = (image * 255).astype(np.uint8)

    features = []

    for d in distances:
        for a in angles:
            glcm = graycomatrix(image, [d], [a], levels=256, symmetric=True, normed=True)

            energy = graycoprops(glcm, 'energy')[0, 0]
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            idf = calculate_idf(glcm[:, :, 0, 0])
            entropy = calculate_entropy(glcm[:, :, 0, 0])
            variance = calculate_variance(glcm[:, :, 0, 0])

            features.extend([energy, correlation, contrast, homogeneity, idf, entropy, variance])


            if len(encabezados)<84:
                if a == np.pi/4:
                    a=45
                elif a==np.pi/2:
                    a=90
                elif a==3*np.pi/4:
                    a=135
                encabezados.append(f"d:{d}_a:{a}_energia")
                encabezados.append(f"d:{d}_a:{a}_contraste")
                encabezados.append(f"d:{d}_a:{a}_correlacion")
                encabezados.append(f"d:{d}_a:{a}_homogeneidad")
                encabezados.append(f"d:{d}_a:{a}_idf")
                encabezados.append(f"d:{d}_a:{a}_entropia")
                encabezados.append(f"d:{d}_a:{a}_varianza")

    return np.array(features)


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
    image = image_path#io.imread(image_path)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    image = (image * 255).astype(np.uint8)

    directions = [0, 45, 90]
    features_all = []

    for d in directions:
        matrix = glrlm(image, d)
        features = glrlm_features(matrix)
        features_all.extend(features)

        if len(encabezados)<99:
            encabezados.append(f"GLRL_a:{d}_SRE")
            encabezados.append(f"GLRL_a:{d}_LRE")
            encabezados.append(f"GLRL_a:{d}_GLNU")
            encabezados.append(f"GLRL_a:{d}_RLN")
            encabezados.append(f"GLRL_a:{d}_RP")

    return np.array(features_all)


def search_img_list():
    img_list = "images.lst"
    if len(sys.argv) > 1:
        img_list = sys.argv[1]
    image_list = []
    if os.path.exists(img_list):
        with open(img_list) as f:
            for line in f:
                image_list.append(line.strip())
    else:
        raise FileNotFoundError(f"File {img_list} does not exist")

    return image_list


if __name__ == "__main__":
    #procesamiento de imagenes 
    caracteristicas = []
    nombres=[]
    clases = []
    df = pd.DataFrame()
    for dataPath in search_img_list():
        for root, dirs, files in os.walk(dataPath):
            for file in files:
                falla_path = os.path.join(root, file)
                image = io.imread(falla_path)
                features = extract_features(image)
                features_GLRL = process_image(image)

                caracteristicas.append(np.concatenate((features, features_GLRL)))
                nombres.append(file)
                #Extract class from the folder name 
                clase = os.path.basename(os.path.dirname(root))
                #print(f"Clase {clase}")
                clases.append(clase)


    print(encabezados)
    df = pd.DataFrame(caracteristicas, columns=encabezados)
    df.insert(0, "Nombre_Imagen", nombres)
    df.insert(1, "Clase", clases)
    print(df)
    df.to_csv("featureMatrix.csv", index=False)
