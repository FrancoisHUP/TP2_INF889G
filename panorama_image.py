from harris_image import harris_corner_detector, mark_corners
import numpy as np
from numpy.linalg import det

def make_translation_homography(dr: float, dc: float) -> np.ndarray:
    """Create a translation homography
    Parameters
    ----------
    dr: float
        Translation along the row axis
    dc: float
        Translation along the column axis
    Returns
    -------
    H: np.ndarray
        Homography as a 3x3 matrix
    """
    H = np.zeros((3,3))
    H[0,0] = 1
    H[1,1] = 1
    H[2,2] = 1
    H[0,2] = dr # Row translation
    H[1,2] = dc # Col translation
    return H

def match_compare(a: float, b: float) -> int:
    """ Comparator for matches
    Parameters
    ----------
    a,b : float
        distance for each match to compare.
    Returns
    -------
    result of comparison, 0 if same, 1 if a > b, -1 if a < b.
    """
    comparison = 0
    if a < b:
        comparison = -1
    elif a > b:
        comparison = 1
    else:
        comparison = 0
    return comparison

def both_images(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Place two images side by side on canvas, for drawing matching pixels.
    Parameters
    ----------
    a,b: ndarray
        Images to place
    Returns
    -------
    c: ndarray
        image with both a and b side-by-side.
    """
    width = a.shape[1] + b.shape[1]
    height = a.shape[0] if a.shape[0] > b.shape[0] else b.shape[0]
    channel = a.shape[2] if a.shape[2] > b.shape[2] else b.shape[2]
    
    both = np.zeros((height,width,channel), dtype=a.dtype)
    both[0:a.shape[0], 0:a.shape[1],0:a.shape[2]] = a
    both[0:b.shape[0], a.shape[1]:a.shape[1]+b.shape[1],0:b.shape[2]] = b
    
    return both

def draw_matches(a: np.ndarray, b: np.ndarray, matches: list, inliers: int) -> np.ndarray:
    """Draws lines between matching pixels in two images.
    Parameters
    ----------
    a, b: ndarray
        two images that have matches.
    matches: list
        array of matches between a and b.
    inliers: int
        number of inliers at beginning of matches, drawn in green.
    Returns
    -------
    c: ndarray
        image with matches drawn between a and b on same canvas.
    """
    both = both_images(a, b)
    n = len(matches)
    for i in range(n):
        r1 = matches[i]['p'][0] # Coordonnée y du point p
        r2 = matches[i]['q'][0] # Coordonnée y du point q
        c1 = matches[i]['p'][1] # Coordonnée x du point p
        c2 = matches[i]['q'][1] # Coordonnée x du point q
        for c in range(c1, c2 + a.shape[1]):
            r = int((c-c1)/(c2 + a.shape[1] - c1)*(r2 - r1) + r1)
            both[r, c, 0] = (0 if i<inliers else 255)
            both[r, c, 1] = (255 if i<inliers else 0)
            both[r, c, 2] = 0
    return both

def draw_inliers(a: np.ndarray, b: np.ndarray, H: np.ndarray, matches: list, thresh: float) -> np.ndarray:
    """ Draw the matches with inliers in green between two images.
    Parameters
    ----------
    a, b: ndarray
        two images to match.
    H: ndarray
        Homography matrix
    matches: list
        array of matches between a and b
    thresh: float
        Threshold to define inliers
    Returns
    -------
    lines: ndarray
        Modified images with inliers
    """
    n_inliers, new_matches = model_inliers(H, matches, thresh)
    lines = draw_matches(a, b, new_matches, n_inliers)
    return lines


def find_and_draw_matches(a: np.ndarray, b: np.ndarray, sigma: float=2, thresh: float=3, nms: int=3) -> np.ndarray:
    """ Find corners, match them, and draw them between two images.
    Parameters
    ----------
    a, b: np.ndarray
         images to match.
    sigma: float
        gaussian for harris corner detector. Typical: 2
    thresh: float
        threshold for corner/no corner. Typical: 1-5
    nms: int
        window to perform nms on. Typical: 3
    Returns
    -------
    lines: np.ndarray
        Images with inliers
    """
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)
    m = match_descriptors(ad, bd)

    a = mark_corners(a, ad, len(ad))
    b = mark_corners(b, bd, len(bd))
    lines = draw_matches(a, b, m, 0)

    return lines

def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates L1 distance between to floating point arrays.
    Parameters
    ----------
    a, b: list or np.ndarray
        arrays to compare.
    Returns
    -------
    l1: float
        l1 distance between arrays (sum of absolute differences).
    """
    return np.sum(np.abs(a - b))


def match_descriptors(a: list, b: list) -> list:
    """Finds best matches between descriptors of two images.
    Parameters
    ----------
    a, b: list
        array of descriptors for pixels in two images.
    Returns
    -------
    matches: list
        best matches found. each descriptor in a should match with at most
        one other descriptor in b.
    """
    an = len(a)
    bn = len(b)
    matches = []
    for j in range(an):
        
        # record ai as the index in a and bi as the index in b.
        min_distance = float('inf')
        bind = 0  
        for i in range(bn):
            distance = l1_distance(a[j]["data"], b[i]["data"]) 
            if distance < min_distance:
                min_distance = distance
                bind = i

        matches.append({})
        matches[j]['ai'] = j
        matches[j]['bi'] = bind 
        matches[j]['p'] = a[j]['pos']
        matches[j]['q'] = b[bind]['pos']
        matches[j]['distance'] = min_distance 

    # Sorting matches based on distance to bring the best matches to the front
    matches.sort(key=lambda x: x['distance']) # Or use match_compare function

    # Ensure matches are one-to-one
    seen_b_indices = set()
    filtered_matches = []
    for match in matches:
        if match['bi'] not in seen_b_indices:
            filtered_matches.append(match)
            seen_b_indices.add(match['bi'])

    return filtered_matches

def project_point(H, p):
    """ Apply a projective transformation to a point.
    Parameters
    ----------
    H: np.ndarray
        homography to project point, of shape 3x3
    p: list
        point to project.
    Returns
    -------
    q: list
        point projected using the homography.
    """
    c = np.zeros((3, 1))

    # Convert point p to homogeneous coordinates
    p_homogeneous = np.array([p[0], p[1], 1])
    
    # Apply homography H to p
    c = H @ p_homogeneous
    
    # Convert back to Cartesian coordinates by dividing by the third component
    q = [c[0] / c[2], c[1] / c[2]]

    return q

def point_distance(p, q):
    """Calculate L2 distance between two points.
    Parameters
    ----------
    p, q: list
        Points.
    Returns
    -------
    l2: float
        L2 distance between them.
    """
    # # Calculate the difference vector and square each component
    # diff = np.array(p) - np.array(q)
    # squared_diff = np.square(diff)
    
    # # Sum the components and take the square root
    # l2 = np.sqrt(squared_diff.sum())
    l2 = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
    return l2

def model_inliers(H: np.ndarray, matches: list, inlier_thresh: float) -> tuple:
    """Count number of inliers in a set of matches. Should also bring inliers to the front of the array.
    Parameters
    ----------
    H: np.ndarray
        homography between coordinate systems.
    matches: list
        matches to compute inlier/outlier.
    thresh: float
        threshold to be an inlier.
    Returns
    -------
    count: int
        number of inliers whose projected point falls within thresh of their match in the other image.
    matches: list
        Should also rearrange matches so that the inliers are first in the array. For drawing.
    """
    count = 0
    new_matches = [] # To reorder the matches
    # TODO: count number of matches that are inliers
    # i.e. distance(H*p, q) < thresh
    # Also, sort the matches m so the inliers are the first 'count' elements.
    
    inliers = []
    outliers = []
    for match in matches:
        # Extract point p from match and convert to homogeneous coordinates (x, y, 1)
        p = np.array([match['p'][0], match['p'][1], 1])
        q = np.array(match['q'])  # Point q in the match
        
        # Project p using the homography H
        p_proj_homog = H @ p
        p_proj = p_proj_homog[:2] / p_proj_homog[2]  # Convert to non-homogeneous coordinates
        
        # Calculate L2 distance between projected point and q
        l2_distance = point_distance(p_proj, q[:2])

        # Check if the match is an inlier based on the threshold
        if l2_distance < inlier_thresh:
            inliers.append(match)
        else:
            outliers.append(match)
    
    # Count of inliers and rearrange matches so inliers are at the beginning
    count = len(inliers)
    # print("Best inlier:", count)
    new_matches = inliers + outliers

    return (count, new_matches)

def randomize_matches(matches: list, n : int) -> list:
    """ Randomly shuffle matches for RANSAC.
    Parameters
    ----------
    matches: list
        matches to shuffle in place
    n: int
        n number of matches to shuffle from the best matches
    Returns
    -------
    shuffled_matches: list
        Shuffled matches
    """
    n_shuffle_matches = matches[:n]
    rest = matches[n:]
    for i in range(len(n_shuffle_matches) - 1, 0, -1):
        j = np.random.randint(0, i)
        n_shuffle_matches[i], n_shuffle_matches[j] = n_shuffle_matches[j], n_shuffle_matches[i]
    return n_shuffle_matches + rest


def compute_homography(matches: list, n: int) -> np.ndarray:
    """Computes homography between two images given matching pixels.
    Parameters
    ----------
    matches: list
        matching points between images.
    n: int
        number of matches to use in calculating homography.
    Returns
    -------
    H: np.ndarray
        matrix representing homography H that maps image a to image b.
    """
    assert n >= 4, "Underdetermined, use n>=4"

    M = np.zeros((n*2,8))
    b = np.zeros((n*2,1))
    for i in range(min(n, len(matches))):
        mx = float(matches[i]['p'][0])
        my = float(matches[i]['p'][1])
        nx = float(matches[i]['q'][0])
        ny = float(matches[i]['q'][1])
        # Fill in the matrices M and b.
        M[2*i] = [mx,  my, 1, 0, 0, 0, -mx*nx, -my*nx] 
        M[2*i + 1] = [0, 0, 0, mx,  my, 1, -mx*ny, -my*ny]
        b[2*i] = nx
        b[2*i + 1] = ny

    # Solve the linear system
    if M.shape[0] == M.shape[1]:
        a = np.linalg.solve(M, b)
    else: # Over-determined, using least-squared
        a = np.linalg.lstsq(M,b,rcond=None)
        a = a[0]
    # If a solution can't be found, return empty matrix;
    if a is None:
        return None
    
    # Fill in the homography H based on the result in a.
    H = np.array([
        [a[0][0], a[1][0], a[2][0]],
        [a[3][0], a[4][0], a[5][0]],
        [a[6][0], a[7][0], 1]
    ])

    return H

def RANSAC(matches: list, thresh: float, k: int, cutoff: int, n: int = 10):
    """Perform RANdom SAmple Consensus to calculate homography for noisy matches.
    Parameters
    ----------
    matches: list
        set of matches.
    thresh: float
        inlier/outlier distance threshold.
    k: int
        number of iterations to run.
    cutoff: int
        inlier cutoff to exit early.
    Returns
    -------
    Hb: np.ndarray
        matrix representing most common homography between matches.
    """
    best = 0
    Hb = make_translation_homography(0, 256) # Initial condition
    for k in range(k): # for k iterations:
        suffle = randomize_matches(matches, n) # shuffle the matches
        H = compute_homography(suffle, n) # compute a homography with a few matches (how many??)
        inlier_count, _ = model_inliers(H, matches, thresh)
        if inlier_count > 0:
            print("Inliers:", inlier_count)
        if inlier_count > best: # if new homography is better than old (how can you tell?):
            best = inlier_count # remember it and how good it is
            Hb = H  # compute updated homography using all inliers
            if inlier_count > cutoff: # if it's better than the cutoff:
                break   # return it immediately

    return Hb  # if we get to the end return the best homography

def combine_images(a, b, H):
    """ Stitches two images together using a projective transformation.
    Parameters
    ----------
    a, b: ndarray
        Images to stitch.
    H: ndarray
        Homography from image a coordinates to image b coordinates.
    Returns
    -------
    c: ndarray
        combined image stitched together.
    """
    Hinv = np.linalg.inv(H)

    # Project the corners of image b into image a coordinates.
    c1 = project_point(Hinv, [0, 0])
    c2 = project_point(Hinv, [b.shape[0], 0])
    c3 = project_point(Hinv, [0, b.shape[1]])
    c4 = project_point(Hinv, [b.shape[0], b.shape[1]])

    # Find top left and bottom right corners of image b warped into image a.
    topleft = [0,0]
    botright = [0,0]
    botright[0] = int(max([c1[0], c2[0], c3[0], c4[0]]))
    botright[1] = int(max([c1[1], c2[1], c3[1], c4[1]]))
    topleft[0]  = int(min([c1[0], c2[0], c3[0], c4[0]]))
    topleft[1]  = int(min([c1[1], c2[1], c3[1], c4[1]]))

    # Find how big our new image should be and the offsets from image a.
    dr = int(min(0, topleft[0]))
    dc = int(min(0, topleft[1]))
    h = int(max(a.shape[0], botright[0]) - dr)
    w = int(max(a.shape[1], botright[1]) - dc)

    # Can disable this if you are making very big panoramas.
    # Usually this means there was an error in calculating H.
    if w > 7000 or h > 7000:
        print("output too big, stopping.")
        return np.copy(a)

    c = np.zeros((h,w,a.shape[2]), dtype=a.dtype)

    for k in range(a.shape[2]):
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                c[i-int(dr),j-int(dc),k] = a[i,j,k]
    
    # Paste in image b
    for i in range(int(h)):
        for j in range(int(w)):
            p_proj = project_point(H, [j+dc, i+dr])  # Project point from c back to b
            if 0 <= p_proj[0] < b.shape[1] and 0 <= p_proj[1] < b.shape[0]:
                # If the projected point falls within the bounds of image b,
                # use bilinear interpolation to estimate the pixel value.
                x, y = p_proj
                x0, y0 = int(x), int(y)
                x1, y1 = min(x0 + 1, b.shape[1] - 1), min(y0 + 1, b.shape[0] - 1)

                # Calculate the bilinear interpolation
                for k in range(b.shape[2]):
                    f00 = b[y0, x0, k]
                    f01 = b[y0, x1, k]
                    f10 = b[y1, x0, k]
                    f11 = b[y1, x1, k]
                    c[i, j, k] = (f00 * (x1 - x) * (y1 - y) +
                                  f10 * (x - x0) * (y1 - y) +
                                  f01 * (x1 - x) * (y - y0) +
                                  f11 * (x - x0) * (y - y0))
    # Paste image a into the new image offset by dr and dc.
    # for k in range(a.shape[2]):
    #     for j in range(a.shape[1]):
    #         for i in range(a.shape[0]):
    #             # TODO: remplir l'image

    #             pass
    # TODO: Paste in image b as well.
    # You should loop over some points in the new image (which? all?)
    # and see if their projection from a coordinates to b coordinates falls
    # inside of the bounds of image b. If so, use bilinear interpolation to
    # estimate the value of b at that projection, then fill in image c.
    
    return c

def panorama_image(a, b, sigma=2, thresh=0.25, nms=3, inlier_thresh=10, iters=1000, cutoff=15):
    """ Create a panoramam between two images.
    Parameters
    ----------
    a, b: ndarray
        images to stitch together.
    sigma: float
        gaussian for harris corner detector. Typical: 2
    thresh: float
        threshold for corner/no corner. Typical: 0.0001-0.0005
    nms: int
        window to perform nms on. Typical: 3
    inlier_thresh: float
        threshold for RANSAC inliers. Typical: 2-5
    iters: int
        number of RANSAC iterations. Typical: 1,000-50,000
    cutoff: int
        RANSAC inlier cutoff. Typical: 10-100
    """
    # Calculate corners and descriptors
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)

    # Find matches
    m = match_descriptors(ad, bd)
    
    # Run RANSAC to find the homography
    H = RANSAC(m, inlier_thresh, iters, cutoff)

    # Stitch the images together with the homography
    comb = combine_images(a, b, H)
    return comb