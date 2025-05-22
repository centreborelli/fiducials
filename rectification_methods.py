import time as time
import numpy as np

import shifts_mesure as shm
import autocorrelation as autoc
import operations as oprt
from scipy.interpolate import griddata
import logging
from joblib import Parallel, delayed
import cv2 as cv


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def optim_redressing(image_deforme,homography):
    h_,w_ = image_deforme.shape
    redressed_image = oprt.apply_homography(np.array(image_deforme), homography, (w_,h_))
    return redressed_image


def redressing(U,V, image_originale, image_deforme, sigma, gamma, method):
    h_, w_ = image_deforme.shape
    if combinations(U,V, image_originale, image_deforme, sigma, gamma, method) is None:
        return []
    
    list_affinites, A_ = combinations(U,V, image_originale, image_deforme, sigma, gamma, method)
    list_image_redressed = []

    # Drawing perspectives
    for i in range(len(list_affinites)):
        if np.linalg.det(list_affinites[i].reshape((2, 2))) != 0:
            transf_inv = oprt.transf_clac(image_deforme, A_, list_affinites[i])
            transformed_image_i = oprt.apply_homography(image_deforme, transf_inv, (w_, h_))
            list_image_redressed.append((transformed_image_i, transf_inv))

    return list_image_redressed






def padding(image_1, image_2) :
    h, w = image_1.shape
    h_, w_ = image_2.shape    
    # Calcul des décalages nécessaires pour centrer
    top_pad = (h - h_) // 2
    bottom_pad = h - h_ - top_pad
    left_pad = (w - w_) // 2
    right_pad = w - w_ - left_pad

    # Application du padding
    zone_resized = np.pad(
        image_2, 
        ((top_pad, bottom_pad), (left_pad, right_pad)), 
        mode='constant', 
        constant_values=np.mean(image_2)
    )
    return image_1, zone_resized

def padding_zone_corr(image_ref, zone):
    if image_ref.size >= zone.size :
        return padding(image_ref, zone)
    else :
        return padding(zone,image_ref)



def estimate_affinity(affinities, positions, target_position, method='linear'):
    positions = np.array(positions)
    affinity_components = np.array([aff[:2, :].flatten() for aff in affinities])  
    
    
    estimated_components = np.zeros(6)
    for i in range(6):
        estimated_components[i] = griddata(positions, affinity_components[:, i], target_position, method=method)
    
    
    estimated_affinity = np.array([
        [estimated_components[0], estimated_components[1], estimated_components[2]],
        [estimated_components[3], estimated_components[4], estimated_components[5]],
        [0, 0, 1]
    ])
    
    return estimated_affinity



def best_redressing(image_originale, list_image_redressed):

    if list_image_redressed == [] :
        return None
    h, w = image_originale.shape
    peak_values = []
    shifts = []
    for i in range(len(list_image_redressed)):
        image, image_i_resized = padding_zone_corr(image_originale, list_image_redressed[i])
        correlation_image = autoc.correlation_nomalisee(image, image_i_resized)

        # Calculer la corrélation normalisée
        h_,w_ = correlation_image.shape
        # Calculer la corrélation normalisée
        excluded_regions = [(h_//2, w_//2, 0)]
                
      
        y, x = shm.find_peak(correlation_image, excluded_regions, h_, w_)
        x_shifted = x
        y_shifted = y

        peak_values.append(correlation_image[y, x])
        shifts.append((x_shifted, y_shifted))

    best_index = np.argmax(peak_values)  
    return best_index, shifts, peak_values


def draw_peaks_local(image_deformed, sigma, gamma,num_cols,num_rows) :

    height, width = image_deformed.shape
    part_width = width // num_cols
    part_height = height // num_rows
    output = np.zeros((width,height))
    output = oprt.gray_to_png(output)
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * part_width
            right = width if col == num_cols - 1 else (col + 1) * part_width
            upper = row * part_height
            lower = height if row == num_rows - 1 else (row + 1) * part_height

            # Extraire une sous-image
            part_image = image_deformed[upper:lower, left:right]
            autocorrelation = autoc.autocorrelation_display(part_image)
            if shm.peaks_and_shifts_subpixelic(autocorrelation, sigma, gamma, method='gaussian') == [] :
                return None

            
            deformed_points = shm.peaks_and_shifts_subpixelic(autocorrelation, sigma, gamma, method='gaussian')
            output[upper:lower, left:right] = oprt.draw_peaks(autocorrelation, deformed_points)
    
    cv.imwrite('draw_peaks.jpg',output)
    return output
    


def combinations(U,V, image_originale,image_deforme, sigma, gamma,method) :
    autocorrelation = autoc.autocorrelation_display(image_deforme)
    if shm.peaks_and_shifts_subpixelic(autocorrelation, sigma, gamma, method) == [] :
        return None
    
    x1,y1 = U
    x2,y2 = V
    x3,y3 = x1-x2, y1-y2
    reference_points = oprt.calculate_symmetric_points(image_originale, x1,y1,x2,y2,x3,y3)

    deformed_points = shm.peaks_and_shifts_subpixelic(autocorrelation, sigma, gamma, method)

    # Ordonner les points
    renormal_ref = [oprt.renormalize(image_originale, point) for point in reference_points]
    points_current = [oprt.renormalize(image_deforme, point) for point in deformed_points]
    ref, picked = oprt.find_closest_points_order(points_current, renormal_ref, np.array([0, 0]))
    
    
    A_ = np.array([[ref[0][0],ref[0][1],0,0,
                    0,0,ref[0][0],ref[0][1],
                    ref[1][0],ref[1][1],0,0,
                    0,0,ref[1][0],ref[1][1]]]).reshape((4,4))

    B_1 = np.array([[picked[0][0],picked[0][1],picked[1][0],picked[1][1]]]).reshape((4,1))
    B_2 = np.array([[picked[1][0],picked[1][1],picked[2][0],picked[2][1]]]).reshape((4,1))
    B_3 = np.array([[picked[2][0],picked[2][1],picked[3][0],picked[3][1]]]).reshape((4,1))
    B_4 = np.array([[picked[3][0],picked[3][1],picked[4][0],picked[4][1]]]).reshape((4,1))
    B_5 = np.array([[picked[4][0],picked[4][1],picked[5][0],picked[5][1]]]).reshape((4,1))
    B_6 = np.array([[picked[5][0],picked[5][1],picked[0][0],picked[0][1]]]).reshape((4,1))

    
    list_combinations = [B_1,B_2,B_3,B_4,B_5,B_6]
    return list_combinations, A_


        

def image_redressing(image_deformed, U,V, image_originale, sigma, gama, method):
    h_d,w_d = image_deformed.shape
    h,w = image_originale.shape
    # Calculer les versions corrigées pour la sous-image
    if redressing(U,V, image_originale, image_deformed, sigma, gama, method) == [] :
        return np.eye(3,3)
    list_image_redressed_index = redressing(U,V, image_originale, image_deformed, sigma, gama, method)
    list_image_redressed = [p[0] for p in list_image_redressed_index]
    list_index_redressed = [p[1] for p in list_image_redressed_index]


    # S'assurer que les matrices d'affinités sont au bon format
    list_index_redressed = [
        np.array(j[0]).reshape((2, 2)) if len(j[0]) == 4 else j for j in list_index_redressed
    ]

    # Trouver le meilleur redressement pour cette sous-image
    best_index, shift, peaks_values = best_redressing(image_originale, list_image_redressed)
    x_redressed,y_redressed = shift[best_index]
    local_affinity = oprt.image_centralizer(image_deformed, list_index_redressed[best_index])
    local_affinity[0][2] -= w/2-x_redressed-(w-w_d)/2
    local_affinity[1][2] -= h/2-y_redressed-(h-h_d)/2
    return local_affinity




def process_sub_image(row, col, part_width, part_height, width, height, num_cols, num_rows, image_deformed, U, V, image_originale, sigma, gamma, method):
    left = col * part_width
    right = width if col == num_cols - 1 else (col + 1) * part_width
    upper = row * part_height
    lower = height if row == num_rows - 1 else (row + 1) * part_height
    part_image = image_deformed[upper:lower, left:right]
    
    try:
        affinity_loc = image_redressing(part_image, U, V, image_originale, sigma, gamma, method)

        if affinity_loc is None :
            affinity_loc = np.eye(3, 3)
    except np.linalg.LinAlgError:

        return None  #
    im_best_loc = oprt.apply_homography(part_image, affinity_loc, image_originale.shape)
    
    point_finale = np.array([[(right + left) / 2, (lower + upper) / 2]])
    
    p_reshaped = point_finale.T
    numerator = (affinity_loc[:2, :2] @ p_reshaped) + affinity_loc[:2, 2].reshape(2, 1)
    
    p_with_ones = np.hstack((point_finale, np.array([[1]])))
    p_with_ones_reshaped = p_with_ones.T  
    affinity_last_row_reshaped = affinity_loc[2, :].reshape(1, 3)  
    denominator = affinity_last_row_reshaped @ p_with_ones_reshaped  
    transformation = numerator / denominator
    
    return im_best_loc, affinity_loc, (right + left) / 2, (lower + upper) / 2, transformation

def redressing_zones(image_deformed, U, V, image_originale, num_cols, num_rows, sigma, gamma, method):
    height, width = image_deformed.shape
    part_width = width // num_cols
    part_height = height // num_rows
    
    results = Parallel(n_jobs=-1)(
        delayed(process_sub_image)(row, col, part_width, part_height, width, height, num_cols, num_rows, image_deformed, U, V, image_originale, sigma, gamma, method)
        for row in range(num_rows) for col in range(num_cols)
    )
    
    # Filtrer les résultats None
    results = [res for res in results if res is not None]
    
    if not results:
        return [], [], [], []
    
    list_local_best_images, list_local_affinities, positions_finales_x, positions_finales_y, positions_originales = zip(*results)
    positions_finales = list(zip(positions_finales_x, positions_finales_y))
    
    return list_local_best_images, list_local_affinities, [[i] for i in positions_finales], positions_originales



""" def process_random_patch(x, y, w, h, image_deformed, U, V, image_originale, sigma, gamma, method):
    part_image = image_deformed[y:y+h, x:x+w]
    
    try:
        affinity_loc = image_redressing(part_image, U, V, image_originale, sigma, gamma, method)
        if affinity_loc is None:
            affinity_loc = np.eye(3, 3)
    except np.linalg.LinAlgError:
        return None

    im_best_loc = oprt.apply_homography(part_image, affinity_loc, image_originale.shape)

    point_finale = np.array([[(x + w / 2), (y + h / 2)]])
    p_reshaped = point_finale.T
    numerator = (affinity_loc[:2, :2] @ p_reshaped) + affinity_loc[:2, 2].reshape(2, 1)

    p_with_ones = np.hstack((point_finale, np.array([[1]])))
    p_with_ones_reshaped = p_with_ones.T  
    affinity_last_row_reshaped = affinity_loc[2, :].reshape(1, 3)
    denominator = affinity_last_row_reshaped @ p_with_ones_reshaped

    transformation = numerator / denominator

    return im_best_loc, affinity_loc, x + w / 2, y + h / 2, transformation, part_image


def redressing_random_patches(image_deformed, U, V, image_originale, w, h, N_blocs, sigma, gamma, method, sliding=True, step=20):
    height, width = image_deformed.shape

    positions = []
    if sliding:
        # Générer des positions glissantes avec un pas défini
        for y in range(0, height - h + 1, step):
            for x in range(0, width - w + 1, step):
                positions.append((x, y))
        np.random.shuffle(positions)
        positions = positions[:N_blocs]  # En prendre N_blocs au max
    else:
        # Positions aléatoires
        for _ in range(N_blocs):
            x = np.random.randint(0, width - w + 1)
            y = np.random.randint(0, height - h + 1)
            positions.append((x, y))
    results = Parallel(n_jobs=-1)(
        delayed(process_random_patch)(x, y, w, h, image_deformed, U, V, image_originale, sigma, gamma, method)
        for x, y in positions
    )

    results = [res for res in results if res is not None]

    if not results:
        return [], [], [], [],[]

    list_local_best_images, list_local_affinities, positions_finales_x, positions_finales_y, positions_originales, part_image = zip(*results)
    positions_finales = list(zip(positions_finales_x, positions_finales_y))
    return list_local_best_images, list_local_affinities, [[i] for i in positions_finales], positions_originales,part_image """



def process_random_patch(x, y, w, h, image_deformed, U, V, image_originale, sigma, gamma, method):
    part_image = image_deformed[y:y+h, x:x+w]
    
    try:
        affinity_loc = image_redressing(part_image, U, V, image_originale, sigma, gamma, method)
        if affinity_loc is None:
            affinity_loc = np.eye(3, 3)
    except np.linalg.LinAlgError:
        return None

    return affinity_loc, x + w / 2, y + h / 2


def redressing_random_patches(image_deformed, U, V, image_originale, w, h, N_blocs, sigma, gamma, method, sliding=True, step=20):
    height, width = image_deformed.shape

    positions = []
    if sliding:
        for y in range(0, height - h + 1, step):
            for x in range(0, width - w + 1, step):
                positions.append((x, y))
        np.random.shuffle(positions)
        positions = positions[:N_blocs]
    else:
        for _ in range(N_blocs):
            x = np.random.randint(0, width - w + 1)
            y = np.random.randint(0, height - h + 1)
            positions.append((x, y))

    results = Parallel(n_jobs=-1)(
        delayed(process_random_patch)(x, y, w, h, image_deformed, U, V, image_originale, sigma, gamma, method)
        for x, y in positions
    )

    results = [res for res in results if res is not None]

    if not results:
        return [], [], []

    list_local_affinities, positions_finales_x, positions_finales_y = zip(*results)
    positions_finales = list(zip(positions_finales_x, positions_finales_y))
    positions_finales_adapted = [np.array([[x[0]], [x[1]]]) for x in positions_finales]

    # Compute center positions
    patch_centers = [(x + w / 2, y + h / 2) for x, y in positions[:len(results)]]

    return list_local_affinities, positions_finales_adapted, patch_centers


""" def redressing_zones(image_deformed, U,V, image_originale, num_cols, num_rows, sigma, gamma):
    height, width = image_deformed.shape
    part_width = width // num_cols
    part_height = height // num_rows
    positions_finales = []
    positions_originales = []
    list_local_affinities = []
    list_local_best_images = []

    for row in range(num_rows):
        for col in range(num_cols):
            left = col * part_width
            right = width if col == num_cols - 1 else (col + 1) * part_width
            upper = row * part_height
            lower = height if row == num_rows - 1 else (row + 1) * part_height

            # Extraire une sous-image
            part_image = image_deformed[upper:lower, left:right]
            
            if image_redressing(part_image, U,V, image_originale, sigma, gamma).any()  == None :
                affinity_loc = np.eye(3,3) 
            
            else :
                affinity_loc = image_redressing(part_image, U,V, image_originale, sigma, gamma)

            im_best_loc = oprt.apply_homography(part_image, affinity_loc, image_originale.shape)

            point_finale = np.array([[(right+left)/2,(lower+upper)/2]])
            
            p_reshaped = point_finale.T
            numerator = (affinity_loc[:2, :2] @ p_reshaped) + affinity_loc[:2, 2].reshape(2, 1)

            p_with_ones = np.hstack((point_finale, np.array([[1]])))  
            p_with_ones_reshaped = p_with_ones.T  
            affinity_last_row_reshaped = affinity_loc[2, :].reshape(1, 3)  
            denominator = affinity_last_row_reshaped @ p_with_ones_reshaped  

            # Compute the transformation
            transformation = numerator / denominator



            positions_finales.append(((right+left)/2,(lower+upper)/2))
            positions_originales.append(transformation)
            list_local_best_images.append(im_best_loc)
            list_local_affinities.append(affinity_loc)
    
    return list_local_best_images, list_local_affinities, positions_finales, positions_originales  """




def image_processing_rectification(image_deformation, U,V, original ,sigma, gamma, method) :
    affinity_ = image_redressing(image_deformation, U,V, original, sigma, gamma, method)
    redressed_ = oprt.apply_homography(image_deformation, affinity_, original.shape)
    return redressed_






