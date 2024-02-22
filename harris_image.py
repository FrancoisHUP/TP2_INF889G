import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def describe_point(im: np.ndarray, pos: list) -> dict:
    """Crée un descripteur de caractéristique pour un point l'image
    Parameters
    ----------
    im: ndarray
        Image source
    pos: (2,) list
        Position (r,c) dans l'image qu'on souhaite décrire
    Returns
    -------
    d: dict
        Descripteur pour cet indice.
    """
    r = 2 # Rayon du voisinage
    
    # Descripteur
    d = dict()
    d["pos"] = pos
    d["n"] = (2*r + 1)**2*im.shape[2] # Nombre de valeurs dans le descripteur
    d["data"] = np.zeros((d["n"],), dtype=float)

    # Valeur du pixel central
    cval = im[pos[0], pos[1], :]

    # Limite du voisinage
    r0 = pos[0] - r if pos[0] - r > 0 else 0
    r1 = pos[0] + r + 1 if pos[0] + r + 1 < im.shape[0] else im.shape[0]-1
    c0 = pos[1] - r if pos[1] - r > 0 else 0
    c1 = pos[1] + r + 1 if pos[1] + r + 1 < im.shape[1] else im.shape[1]-1

    # Extraction et normalisation des valeurs
    values = (im[r0:r1, c0:c1, :].astype(float) - cval).ravel()

    # Intégration dans le descripteur
    d['data'][0:len(values)] = values

    return d

def mark_spot(im: np.ndarray, p: list, color: list = [255,0,255]) -> np.ndarray:
    """ Marque la position d'un point dans l'image.
    Parameters
    ----------
    im: ndarray
        Image à marquer
    p: (2,) list
        Position (r,c) du point
    color: (3,) list
        Couleur de la marque
    Returns
    -------
    im: ndarray
        Image marquée.
    """
    r = p[0]
    c = p[1]

    for i in range(-9,10):
        if r+i < 0 or r+i >= im.shape[0] or c+i < 0 or c+i >= im.shape[1]:
            continue # ce pixel est à l'extérieur de l'image
        im[r+i, c, 0] = color[0]
        im[r+i, c, 1] = color[1]
        im[r+i, c, 2] = color[2]
        im[r, c+i, 0] = color[0]
        im[r, c+i, 1] = color[1]
        im[r, c+i, 2] = color[2]

    return im

def mark_corners(im: np.ndarray, d: list, n: int) -> np.ndarray:
    """ Marks corners denoted by an array of descriptors.
    Parameters
    ----------
    im: ndarray
        Image à marquer
    d: list
        Coins dans l'image
    n: int
        Nombre de descripteurs à marquer
    Returns
    -------
    im: ndarray
        Image marquée
    """
    m = np.copy(im)
    for i in range(n):
        m = mark_spot(m, d[i]['pos'])
    return m

def smooth_image(im: np.ndarray, sigma: float) -> np.ndarray:
    """Lissage d'une image avec un filtre gaussien.
    Parameters
    ----------
    im: ndarray
        Image à traiter
    sigma: float
        Écart-type pour la gaussienne.
    Returns
    -------
    s: ndarray
        Image lissée
    """
    s = gaussian_filter(im, sigma)
    return s

def structure_matrix(im: np.ndarray, sigma: float) -> np.ndarray:
    """Calcul du tenseur de structure d'un image.
    Parameters
    ----------
    im: ndarray
        Image à traiter (tons de gris et normalisée entre 0 et 1).
    sigma: float
        Écart-type pour la somme pondérée
    Returns
    -------
    S: ndarray
        Tenseur de structure. 1er canal est Ix^2, 2e canal est Iy^2
        le 3e canal est IxIy
    """
    # Calcul des dérivées
    Ix = sobel(im, axis=1)
    Iy = sobel(im, axis=0)

    # Lissage des dérivées
    Ix2 = smooth_image(Ix**2, sigma)
    Iy2 = smooth_image(Iy**2, sigma)
    IxIy = smooth_image(Ix*Iy, sigma)

    # Création du tenseur de structure
    S = np.zeros((im.shape[0], im.shape[1], 3))
    S[:,:,0] = Ix2
    S[:,:,1] = Iy2
    S[:,:,2] = IxIy

    return S

def cornerness_response(S: np.ndarray) -> np.ndarray:
    """Estimation du cornerness de chaque pixel avec le tenseur de structure S.
    Parameters
    ----------
    S: ndarray
        Tenseur de structure de l'image
    Returns
    -------
    R: ndarray
        Une carte de réponse de la cornerness
    """
    Sxx = S[:,:,0]
    Syy = S[:,:,1]
    Sxy = S[:,:,2]
    
    # Corner response
    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    alpha=0.06
    
    # On utilise la formulation det(S) - alpha * trace(S)^2, alpha = 0.06
    R = det - alpha *(trace**2)

    return R

def nms_image(im: np.ndarray, w: int) -> np.ndarray:
    """Effectue la supression des non-maximum sur l'image des feature responses.
    Parameters
    ----------
    im: ndarray
        Image 1 canal des réponses de caractéristiques (feature response)
    w: int
        Distance à inspecter pour une grande réponse
    Returns
    -------
    r: ndarray
        Image contenant seulement les maximums locaux pour un voisinage de w pixels.
    """
    r = np.copy(im)
    h, w_ = im.shape

    for i in range(h):
        for j in range(w_):
            max_val = im[i, j]  # On commence par supposer que le pixel actuel est le maximum
            for ki in range(max(0, i-w), min(i+w+1, h)):  # Parcours des voisins en hauteur
                for kj in range(max(0, j-w), min(j+w+1, w_)):  # Parcours des voisins en largeur
                    if im[ki, kj] > max_val:  # Si un voisin a une réponse supérieure
                        r[i, j] = -np.inf  # Le pixel actuel n'est pas un maximum local
                        break  # Pas besoin de chercher plus loin pour ce pixel
                if r[i, j] == -np.inf:
                    break  # Sortie anticipée si le pixel a déjà été marqué

    return r 

def harris_corner_detector(im: np.ndarray, sigma: float, thresh: float, nms: int) -> np.ndarray:
    """ Détecteur de coin de Harris, et extraction des caractéristiques autour des coins.
    Parameters
    ----------
    im: ndarray
        Image à traiter (RGB).
    sigma: float
        Écart-type pour Harris.
    thresh: float
        Seuil pour le cornerness
    nms: int
        Distance maximale à considérer pour la supression des non-maximums
    Returns
    -------
    d: list
        Liste des descripteurs pour chaque coin dans l'image
    """
    img = im.mean(axis=2) # Convert to grayscale
    img = (img.astype(float) - img.min()) / (img.max() - img.min())

    # Calculate structure matrix
    S = structure_matrix(img, sigma)

    # Estimate cornerness
    R = cornerness_response(S)

    # Run NMS on the responses
    Rnms = nms_image(R, nms)

    R_max = np.max(Rnms)
    corner_coordinates = np.where(Rnms >= thresh * R_max)
    corner_coordinates = list(zip(corner_coordinates[0], corner_coordinates[1]))

    d = []
    for c in corner_coordinates:
        d.append(describe_point(im, c))
    
    return d

def detect_and_draw_corners(im: np.ndarray, sigma: float, thresh: float, nms: int) -> np.ndarray:
    """ Trouve et dessine les coins d'une image
    Parameters
    ----------
    im: ndarray
        L'image à traiter (RGB).
    sigma: float
        Écart-type pour Harris.
    thresh: float
        Seuil pour le cornerness.
    nms: int
        Distance maximale à considérer pour la supression des non-maximums
    Returns
    m: ndarray
        Image marqué avec les coins détectés
    """
    d = harris_corner_detector(im, sigma, thresh, nms)
    m = mark_corners(im, d, len(d))
    return m