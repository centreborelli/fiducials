import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares
import math
from itertools import combinations
import cv2 as cv

from scipy.ndimage import maximum_filter

import autocorrelation as autoc
import operations as oprt



def filtring(images_):
    gaus_k = cv.getGaussianKernel(9, 3)
    kernel = (gaus_k * gaus_k.T)
    image_filtred = cv.filter2D(images_, -1,kernel )

    return images_-image_filtred

""" def laplacian (u) : 
    v = 0.0 * u
    for x in range(-1, 2) :
        for y in range(-1, 2) :
            if x == 0 and y == 0: 
                r = -1
            else : r=1/8
        v += r * shift_image(u, x, y) 
    return v """



def masking_fftshift(arg, mask):
    h,w = arg.shape
    arg[:,:w//2]=0
    
    arg[h//2-mask:h//2+mask,w//2-mask:w//2+mask] = 0
    return arg


def autocorrelation_display_contrast(arg,mask):
    autocorr_shifft = autoc.autocorrelation_display(arg,mask)
    x1, y1,x2,y2,x3,y3 = pics_and_shifts(autocorr_shifft, 8)
    min_val = autocorr_shifft[x3, y3]
    max_val = autocorr_shifft[x2, y2]
    return oprt.rescale_contrast(autocorr_shifft, 100, min_val)*(-1)


def autocorr_grid(processed_image, mask, num_cols, num_rows):
    image = np.array(processed_image)
    height, width = image.shape
    # les dimensions de chaque partie
    part_width = width // num_cols
    part_height = height // num_rows
    output = np.zeros((height, width))
    # Division d'image et autocorrélation
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * part_width
            upper = row * part_height
            right = (col + 1) * part_width if col < num_cols - 1 else width
            lower = (row + 1) * part_height if row < num_rows - 1 else height
            part_image = image[upper:lower, left:right]
            autocorr_part = autoc.autocorrelation_display(part_image)
            autocorr_part[(lower-upper)//2-mask:(lower-upper)//2+mask, (right-left)//2-mask:(right-left)//2+mask] = 0
            x1, y1,x2,y2,x3,y3 = pics_and_shifts(autocorr_part, mask, mask//2)
            min_val = output[x2, y2]
            max_val = output[x2, y2]
            output[upper:lower, left:right] = autocorr_part #rescale_contrast(autocorr_part, 100, min_val)*(-1)
    return output



def check_hexagon_like(points):
    if len(points) != 6:
        return False

    def ellipse_distance(params, points):
        xc, yc, a, b, theta = params
        distances = []
        for x, y in points:
            x_rotated = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
            y_rotated = -(x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)
            # Avoid division by zero
            if a == 0 or b == 0:
                return np.inf
            distance = ((x_rotated / a) ** 2 + (y_rotated / b) ** 2 - 1) ** 2
            distances.append(distance)
        return distances

    # Initial guess for ellipse parameters
    x_mean, y_mean = np.mean(points, axis=0)
    x_std, y_std = np.std(points, axis=0)
    # Ensure non-zero values for a and b
    initial_guess = [x_mean, y_mean, max(x_std, 1e-3), max(y_std, 1e-3), 0]

    try:
        # Fit ellipse (On peut utiliser Levenberg-Marquardt optimization algorithm)
        result = least_squares(ellipse_distance, initial_guess, args=(points,), method='lm')

        # Check if the fit is good (1 pixel est juste un seuil de précision)
        if result.cost < 1:
            return True
    except (ValueError, RuntimeError):
        pass
    
    return False




def check_hexagon(list_points, tol):
    if len(list_points) < 3:
        return False  # Pas assez de points pour former un hexagone
    try:
        condition1 = int(np.linalg.norm(np.array(list_points[0]) + np.array(list_points[1]) - np.array(list_points[2])) < tol)
        condition2 = int(np.linalg.norm(np.array(list_points[0]) - np.array(list_points[1]) - np.array(list_points[2])) < tol)
        condition3 = int(-np.linalg.norm(np.array(list_points[0]) - np.array(list_points[1]) + np.array(list_points[2])) < tol)
        condition4 = int(-np.linalg.norm(np.array(list_points[0]) + np.array(list_points[1]) + np.array(list_points[2])) < tol)
        return condition1 or condition2 or condition3 or condition4
    except IndexError:
        return False  # Gestion des erreurs d'index



def check_non_alignment(points, tolerance=200):
    x1, y1, x2, y2, x3, y3 = points
    area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2)
    return area > tolerance

def is_convex_polygon(points):
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0])

    n = len(points)
    if n < 3:
        return False
    
    sign = None
    for i in range(n):
        cp = cross_product(points[i], points[(i+1)%n], points[(i+2)%n])
        if sign is None:
            sign = cp > 0
        elif (cp > 0) != sign:
            return False
    return True

def check_coordinate_relation(points, original_points, tolerance=1):
    x1, y1, x2, y2, x3, y3 = points
    ox1, oy1, ox2, oy2, ox3, oy3 = original_points

    # Calcul des vecteurs pour les points détectés
    v1 = (x2 - x1, y2 - y1)
    v2 = (x3 - x2, y3 - y2)
    v3 = (x1 - x3, y1 - y3)

    # Calcul des vecteurs pour les points originaux
    ov1 = (ox2 - ox1, oy2 - oy1)
    ov2 = (ox3 - ox2, oy3 - oy2)
    ov3 = (ox1 - ox3, oy1 - oy3)

    # Vérification de la condition v1-v2 = (x1,y1) - (x2,y2) = (x3,y3)
    def vector_equal(v1, v2, tolerance):
        return abs(v1[0] - v2[0]) <= tolerance and abs(v1[1] - v2[1]) <= tolerance

    return (vector_equal(v1, ov1, tolerance) and
            vector_equal(v2, ov2, tolerance) and
            vector_equal(v3, ov3, tolerance))


def distance_entre_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculer_distances_trois_points(point1, point2, point3):
    points = [tuple(point1), tuple(point2), tuple(point3)]
    distances = {}
    for (p1, p2) in combinations(points, 2):
        distance = distance_entre_points(p1, p2)
        distances[(p1, p2)] = distance
    return distances

def check_min_distance(points, min_distance):

    
    return all(np.linalg.norm(np.array(p1) - np.array(p2)) >= min_distance 
              for i, p1 in enumerate(points) 
              for p2 in points[i+1:])

def check_region(image, points, min_distance):
    h,w = image.shape

    return all(np.linalg.norm(np.array(p2)) <= int(3*min_distance) 
              for i, p2 in enumerate(points) 
              for p2 in points[i+1:])
    

def regrouper_points_proches(points, min_distance):
    groupes = []
    for point in points:
        for groupe in groupes:
            if any(np.linalg.norm(np.array(point) - np.array(p)) < min_distance for p in groupe):
                groupe.append(point)
                break
        else:
            groupes.append([point])
    return groupes

def calculer_barycentre(groupe):
    return np.mean(groupe, axis=0)



def find_peak(arr, excluded_regions, h, w):
    work_arr = arr.copy()
    
    for (cy, cx, radius) in excluded_regions:
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius + 1)
        work_arr[y_min:y_max, x_min:x_max] = -np.inf
        
        # Masquer la région symétrique
        sy = h - cy
        sx = w - cx
        y_min = max(0, sy - radius)
        y_max = min(h, sy + radius + 1)
        x_min = max(0, sx - radius)
        x_max = min(w, sx + radius + 1)
        work_arr[y_min:y_max, x_min:x_max] = -np.inf
    #Image.fromarray(work_arr).save("/Users/i.bencheikh/Desktop/fp.tif")
    return np.unravel_index(np.argmax(work_arr), work_arr.shape)

def pics_and_shifts(processed_img, mask, min_distance, sigma, gamma):
    #processed_img = cv.Canny(oprt.gray_to_png(processed_img), 50, 150)
    h, w = processed_img.shape
    autocorr = masking_fftshift(autoc.autocorrelation_display(processed_img), mask)
    h, w = autocorr.shape
    
    def find_valid_peaks():
        excluded_regions = [(h//2, w//2, mask)]
        peaks = []
        peak_values = []
        
        # Trouver plus de pics que nécessaire
        num_peaks_to_find = 3  # On cherche plus de pics pour avoir des alternatives
        
        for _ in range(num_peaks_to_find):
            y, x = find_peak(autocorr, excluded_regions, h, w)
            x_shifted = np.abs(x - w//2)
            y_shifted = y - h//2
            peak_values.append(autocorr[y, x])
            peaks.append((x_shifted, y_shifted))
            excluded_regions.append((y, x, min_distance))
        
        # Trier les pics par intensité
        sorted_peaks = [x for _, x in sorted(zip(peak_values, peaks), reverse=True)]
        # Tester toutes les combinaisons de 3 pics
        for i in range(len(sorted_peaks)-2):
            for j in range(i+1, len(sorted_peaks)-1):
                for k in range(j+1, len(sorted_peaks)):
                    points = [sorted_peaks[i], sorted_peaks[j], sorted_peaks[k]]
                    negative_points = [(-x, -y) for x, y in points]
                    coords = [coord for point in points for coord in point]
                    

                    
                    # Vérifier toutes les conditions
                    if (check_min_distance(points[0], points[1], points[2], 
                                        negative_points[0], negative_points[1], negative_points[2],
                                        min_distance) and 
                        
                        #check_region(processed_img, points[0], points[1], points[2], 
                        #           negative_points[0], negative_points[1], negative_points[2],
                        #           min_distance) and
                        check_hexagon(points,sigma) and 
                        check_non_alignment(coords)
                        ):
                        return coords
        
        return None
    valid_peaks = find_valid_peaks()
    if valid_peaks is not None:
            return valid_peaks
    return []





def find_local_maxima(image):
    neighborhood = np.ones((3, 3), bool)  # Voisinage 3x3
    local_max = maximum_filter(image, footprint=neighborhood) == image
    return local_max

def calculate_prominence(image, local_max):
    prominence = np.zeros_like(image, dtype=float)
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            if local_max[i, j]:
                # Trouver le chemin le plus bas autour du maximum
                min_value = np.min(image[max(0, i-1):min(rows, i+2), max(0, j-1):min(cols, j+2)])
                prominence[i, j] = image[i, j] - min_value

    return prominence

def subpixel_peak_position_quadratic(image, x, y):
    x_min = max(0, x-1)
    x_max = min(image.shape[0], x+2)
    y_min = max(0, y-1)
    y_max = min(image.shape[1], y+2)
    
    region = image[x_min:x_max, y_min:y_max]
    pad_x = 3 - region.shape[0]
    pad_y = 3 - region.shape[1]
    if pad_x > 0 or pad_y > 0:
        region = np.pad(region, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=0)

    dx = (region[2, 1] - region[0, 1]) / (2 * (2 * region[1, 1] - region[2, 1] - region[0, 1]))
    dy = (region[1, 2] - region[1, 0]) / (2 * (2 * region[1, 1] - region[1, 2] - region[1, 0]))
    
    return y + dy, x + dx




def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coords
    x0, y0 = float(x0), float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return A * np.exp( - (a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) + offset

def subpixel_peak_position_gaussian_fit(image, x, y):
    x = int(round(x))
    y = int(round(y))
    
    x_min = max(0, x-1)
    x_max = min(image.shape[0], x+2)
    y_min = max(0, y-1)
    y_max = min(image.shape[1], y+2)
    
    region = image[x_min:x_max, y_min:y_max]
    
    if region.shape != (3, 3):
        region = np.pad(region, ((0, 3 - region.shape[0]), (0, 3 - region.shape[1])), mode='constant')

    # Create grid for fitting
    x_grid, y_grid = np.meshgrid(np.arange(3), np.arange(3))
    coords = (x_grid.ravel(), y_grid.ravel())
    values = region.ravel()

    # Initial guess: A, x0, y0, sigma_x, sigma_y, theta, offset
    A0 = np.max(values) - np.min(values)
    offset0 = np.min(values)
    x0 = y0 = 1.0  # center of 3x3
    initial_guess = [A0, x0, y0, 1.0, 1.0, 0.0, offset0]

    def error(params):
        return np.sum((gaussian_2d(coords, *params) - values)**2)

    result = minimize(error, initial_guess, method='L-BFGS-B')
    
    if result.success:
        _, x0_fit, y0_fit, *_ = result.x
        subpixel_x = y + (x0_fit - 1)
        subpixel_y = x + (y0_fit - 1)
        return subpixel_x, subpixel_y
    else:
        # fallback
        return float(y), float(x)




def subpixel_peak_position_centroid(image, x, y):
    x_min = max(0, x-1)
    x_max = min(image.shape[0], x+2)
    y_min = max(0, y-1)
    y_max = min(image.shape[1], y+2)
    
    region = image[x_min:x_max, y_min:y_max]
    region = np.pad(region, ((0, 3 - region.shape[0]), (0, 3 - region.shape[1])), mode='constant')

    total_intensity = np.sum(region)
    if total_intensity == 0:
        return y, x
    
    x_coords, y_coords = np.meshgrid(np.arange(region.shape[1]), np.arange(region.shape[0]))
    x_weighted = np.sum(x_coords * region) / total_intensity
    y_weighted = np.sum(y_coords * region) / total_intensity

    return y + (x_weighted - 1), x + (y_weighted - 1)



def subpixel_peak_position_fit2D(image, x, y):
    from numpy.linalg import lstsq
    
    x_min = max(0, x-1)
    x_max = min(image.shape[0], x+2)
    y_min = max(0, y-1)
    y_max = min(image.shape[1], y+2)
    
    region = image[x_min:x_max, y_min:y_max]
    region = np.pad(region, ((0, 3 - region.shape[0]), (0, 3 - region.shape[1])), mode='constant')

    X, Y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    Z = region.flatten()
    
    A = np.column_stack((X.flatten()**2, Y.flatten()**2, X.flatten()*Y.flatten(), X.flatten(), Y.flatten(), np.ones(9)))
    coeffs, _, _, _ = lstsq(A, Z, rcond=None)
    a, b, c, d, e, f = coeffs

    # Calcul du sommet de la parabole 2D
    denom = 4*a*b - c**2
    if denom == 0:
        return y, x

    x0 = (c*e - 2*b*d) / denom
    y0 = (c*d - 2*a*e) / denom

    return y + x0, x + y0


def subpixel_peak_position_weighted_gaussian(image, x, y):
    import numpy as np
    from scipy.optimize import curve_fit

    # Nouvelle gaussienne 2D générale avec orientation
    def gaussian_2d_general(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
        xg, yg = coords
        x0 = float(x0)
        y0 = float(y0)
        a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
        b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
        c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
        g = amp * np.exp(- (a * (xg - x0)**2 + 2 * b * (xg - x0) * (yg - y0) + c * (yg - y0)**2)) + offset
        return g.ravel()

    # Extraire une région 3x3
    x_min = max(0, x - 1)
    x_max = min(image.shape[0], x + 2)
    y_min = max(0, y - 1)
    y_max = min(image.shape[1], y + 2)
    
    region = image[x_min:x_max, y_min:y_max]
    if region.shape != (3, 3):
        region = np.pad(region, ((0, 3 - region.shape[0]), (0, 3 - region.shape[1])), mode='constant')

    # Coordonnées pour le fitting
    x_coords, y_coords = np.meshgrid(np.arange(3), np.arange(3))
    x_coords = x_coords.astype(float)
    y_coords = y_coords.astype(float)

    # Paramètres initiaux : amplitude, x0, y0, sigma_x, sigma_y, theta, offset
    initial_guess = (
        np.max(region), 1.0, 1.0, 1.0, 1.0, 0.0, np.min(region)
    )

    try:
        popt, _ = curve_fit(
            gaussian_2d_general,
            (x_coords, y_coords),
            region.ravel(),
            p0=initial_guess,
            maxfev=2000
        )
        _, x0, y0, *_ = popt
        return y + (x0 - 1), x + (y0 - 1)  # position sous-pixel
    except Exception as e:
        return float(y), float(x)  # fallback





def peaks_and_shifts_subpixelic(image, sigma, gamma, method):
    local_max = find_local_maxima(image)
    prominence = calculate_prominence(image, local_max)
    flat_prominence = prominence.flatten()
    sorted_indices = np.argsort(flat_prominence)[-7:]
    peak_positions = np.unravel_index(sorted_indices, image.shape)

    method_dict = {
        'quadratic': subpixel_peak_position_quadratic,
        'gaussian': subpixel_peak_position_gaussian_fit,
        'centroid': subpixel_peak_position_centroid,
        'fit2d': subpixel_peak_position_fit2D,
        'weighted_gaussian': subpixel_peak_position_weighted_gaussian
    }

    subpixel_func = method_dict.get(method)
    if not subpixel_func:
        raise ValueError("Méthode inconnue : choisir parmi 'quadratic', 'gaussian', 'centroid', 'fit2d'")

    subpixel_positions = []
    for x, y in zip(peak_positions[0], peak_positions[1]):
        sub_x, sub_y = subpixel_func(image, x, y)
        subpixel_positions.append((sub_x, sub_y))
    
    if check_hexagon(subpixel_positions[:3], sigma):# and check_min_distance(subpixel_positions[:-1], 2):
        return subpixel_positions[:-1]
    
    else:
        return []











