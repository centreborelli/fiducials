import time as time
from PIL import Image
import numpy as np
import cv2 as cv
from numpy.fft import fft2





def validate_homography_matrix(H):
    if H.shape != (3, 3):
        raise ValueError("La matrice de transformation doit être de dimension 3x3")
    if H.dtype not in [np.float32, np.float64]:
        H = H.astype(np.float32)
    return H


def apply_homography(src_img, H, shape):
    # Vérifier que l'image source est un numpy array
    if not isinstance(src_img, np.ndarray):
        raise ValueError("L'image source doit être un numpy array")
    
    # Vérifier que l'image est en niveau de gris si nécessaire
    if len(src_img.shape) > 2:
        src_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    
    # Valider et convertir la matrice d'homographie
    if not isinstance(H, np.ndarray):
        H = np.array(H)
    if H.shape != (3, 3):
        raise ValueError("La matrice d'homographie doit être de taille 3x3")
    
    # Appliquer la transformation
    try:
        return cv.warpPerspective(src_img, H, shape, borderValue=int(np.mean(src_img)))
    except Exception as e:
        raise

def circ_mask(h, k, r, x, y):
    return (x-h)**2 + (y-k)**2 <= r**2



def draw_peaks(image, points):
    h,w = image.shape
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image")
    if image.dtype != np.uint8:
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    color = (255, 0, 0)  # Red color in BGR format
    thickness = 1
    radius = 0 # Radius of the circle     
    print("centre" ,points[0][0],points[0][1]) 

#    output = cv.drawMarker(color_image, (int(points[0][0]),int(points[0][1])), radius, color, thickness)
#    for i in range(1, len(points)):
#        output = cv.drawMarker(output, (int(points[i][0]),int(points[i][1])), radius, color, thickness)

    for i in range(len(points)):
        color_image[int(np.round(points[i][1])),int(np.round(points[i][0]))] = color

    return color_image


def order_points_clockwise(points, center):
    # Calculer les angles de chaque point par rapport au centre
    angles = []
    for point in points:
        # Convertir en coordonnées relatives au centre
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        # Calculer l'angle
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    
    # Trier les points selon leurs angles
    points_with_angles = list(zip(points, angles))
    points_ordered = [p[0] for p in sorted(points_with_angles, key=lambda x: x[1])]
    
    return points_ordered


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

def algebraic_distance(point, ref_point):
    # Exemple : distance algébrique basée sur une relation linéaire
    # f(p, r) = (x_p - x_r)^2 + (y_p - y_r)^2
    return (point[0] - ref_point[0])**2 + (point[1] - ref_point[1])**2



def find_closest_points_order(points_unordered, ref_points, center):
    # Convertir les listes en tableaux NumPy
    points_unordered = np.array(points_unordered)
    ref_points = np.array(ref_points)

    # Calculer les angles pour les points de référence
    angles_ref = np.arctan2(ref_points[:, 1] - center[1], ref_points[:, 0] - center[0])
    sorted_ref_indices = np.argsort(angles_ref)  # Indices des points triés dans l'ordre des aiguilles d'une montre
    sorted_ref_points = ref_points[sorted_ref_indices]

    # Calculer la matrice des distances entre points désordonnés et points de référence triés
    distance_matrix = np.linalg.norm(points_unordered[:, np.newaxis, :] - sorted_ref_points[np.newaxis, :, :], axis=2)

    # Trouver le meilleur appariement avec l'algorithme de Hongrie
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Réordonner les points désordonnés selon l'ordre des points de référence triés
    sorted_points_unordered = points_unordered[row_indices[np.argsort(col_indices)]]

    return sorted_ref_points.tolist(), sorted_points_unordered.tolist()



def calculate_symmetric_points(image, x1, y1, x2, y2, x3, y3):
    h, w = image.shape
    points_unordered = [
        (w//2 - x1, h//2 - y1),
        (w//2 - x2, h//2 - y2),
        (w//2 - x3, h//2 - y3),
        (w//2 + x1, h//2 + y1),
        (w//2 + x2, h//2 + y2),
        (w//2 + x3, h//2 + y3),
    ]
    return [(x,y) for x, y in points_unordered]


def calculate_3d_basis(points_current, points_ref, scale=10):
    center_current = np.mean(points_current, axis=0)
    center_ref = np.mean(points_ref, axis=0)
    
    # Calculer les diagonales de l'hexagone déformé
    diag1 = np.array(points_current[0]) - np.array(points_current[3])
    diag2 = np.array(points_current[1]) - np.array(points_current[4])
    
    # Les diagonales de référence
    diag1_ref = np.array(points_ref[0]) - np.array(points_ref[3])
    diag2_ref = np.array(points_ref[1]) - np.array(points_ref[4])
    
    # Le ratio de déformation est le rapport des longueurs des diagonales
    ratio1 = np.linalg.norm(diag1) / np.linalg.norm(diag1_ref)
    ratio2 = np.linalg.norm(diag2) / np.linalg.norm(diag2_ref)
    
    # Estimer la composante z en fonction de la déformation
    z_component = scale * (1 - min(ratio1, ratio2))
    
    # Construire les vecteurs 3D
    diag1_3d = np.array([diag1[0], diag1[1], z_component])
    diag2_3d = np.array([diag2[0], diag2[1], z_component])
    
    # La normale est le produit vectoriel des diagonales 3D
    normal_3d = np.cross(diag1_3d, diag2_3d)
    
    # Normaliser et projeter sur le plan 2D
    if np.linalg.norm(normal_3d) > 1e-6:
        normal_3d = normal_3d / np.linalg.norm(normal_3d) * scale
        
    # Les composantes x et y de la normale projetée
    normal_2d = np.array([normal_3d[0], normal_3d[1]])
    
    # Direction principale de la déformation comme axe x
    x_vector = diag1
    if np.linalg.norm(x_vector) > 1e-6:
        x_vector = x_vector / np.linalg.norm(x_vector) * scale
        
    # y est perpendiculaire à la normale dans le plan image
    y_vector = np.array([-normal_2d[1], normal_2d[0]])
    if np.linalg.norm(y_vector) > 1e-6:
        y_vector = y_vector / np.linalg.norm(y_vector) * scale
        
    return x_vector, y_vector, normal_2d

def draw_peaks_with_normal(image, x1, y1, x2, y2, x3, y3,
                         x1_ref, y1_ref, x2_ref, y2_ref, x3_ref, y3_ref):
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image")
    
    h, w = image.shape
    
    if image.dtype != np.uint8:
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    # Définition des couleurs
    point_color_ref = (33, 140, 15)    # Vert foncé
    point_color = (255, 255, 255)        # Bleu foncé
    line_color_ref = (148, 148, 148)      # Vert
    line_color = (255, 255, 255)           # Rouge
    x_color = (0, 0, 255)              # Rouge pour normale
    y_color = (255, 0, 0)              # Bleu
    z_color = (0, 255, 0)              # Vert
    
    # Centre et points
    center = (w//2, h//2)
    deformed_points = calculate_symmetric_points(image, x1, y1, x2, y2, x3, y3)
    reference_points = calculate_symmetric_points(image, x1_ref, y1_ref, x2_ref, y2_ref, x3_ref, y3_ref)
    

    points_ref,points_current =  find_closest_points_order(deformed_points, reference_points, np.array([w//2, h//2]))

    # Dessiner les hexagones
    for i in range(len(points_current)):
        # Points
        cv.circle(color_image, points_ref[i], 2, point_color_ref, 1)
        cv.circle(color_image, points_current[i], 3, point_color, 1)
        
        # Lignes
        next_point_ref = points_ref[(i + 1) % len(points_ref)]
        next_point = points_current[(i + 1) % len(points_current)]
        cv.line(color_image, points_ref[i], next_point_ref, line_color_ref, 2)
        cv.line(color_image, points_current[i], next_point, line_color, 3)
    
    # Calculer la base
    x_vector, y_vector, normal = calculate_3d_basis(points_current, points_ref)
    
    start_point = tuple(map(int, center))
    
    # La normale en rouge (c'est notre vecteur principal)
    end_point_normal = (int(start_point[0] + normal[0]), 
                       int(start_point[1] + normal[1]))
    end_point_normal_moins = (int(start_point[0] - normal[0]), 
                       int(start_point[1] - normal[1]))
    cv.arrowedLine(color_image, start_point, end_point_normal, (255, 0, 0), 4)
    #cv.arrowedLine(color_image, start_point, end_point_normal_moins, (255, 0, 0), 2)
    
    # Vecteur y en bleu
    end_point_y = (int(start_point[0] + y_vector[0]), 
                  int(start_point[1] + y_vector[1]))
    cv.arrowedLine(color_image, start_point, end_point_y, (0, 0, 255), 4)
    
    
    # Vecteur x en vert
    end_point_x = (int(start_point[0] + x_vector[0]), 
                  int(start_point[1] + x_vector[1]))
    cv.arrowedLine(color_image, start_point, end_point_x, (0, 255, 0), 4)


    points_current = np.array(points_current)
    points_ref = np.array(points_ref)
    # Dessiner les vecteurs de déformation
    for i in range(3):
        start_point = tuple(points_ref[i].astype(int))
        end_point = tuple(points_current[i].astype(int))
        cv.arrowedLine(color_image, start_point, end_point, (0, 0, 0), 1, tipLength=0.2)
    
    
    return color_image



'''centralizer'''
def image_centralizer(M, transf):
    h,w = M.shape
    x,y = w/2,h/2
    rot_centre_i_red = np.array([[x],[y]])@np.array([[x,y]])@([[transf[2][0]],[transf[2][1]]])+np.array([[transf[2][2]-transf[0][0],-transf[0][1]],[-transf[1][0],transf[2][2]-transf[1][1]]])@np.array([[x],[y]])
    transf[0][2] = rot_centre_i_red[0] 
    transf[1][2] = rot_centre_i_red[1] 
    return transf
    
def renormalize(image, point):
    h, w = image.shape
    x_new = point[0] - w // 2
    y_new = point[1] - h // 2
    return (x_new, y_new)
    


def transf_clac(image, matrice, vect) :
    h,w = image.shape
    
    taransf = np.linalg.solve(matrice, vect).reshape((2,2))
    if np.linalg.det(taransf) == 0 :
        return image_centralizer(image, np.array([[1,0,0],[0,1,0], [0,0,1]]))
    matrice_redressment = np.linalg.inv(taransf)
    matrice_redressment = np.pad(matrice_redressment, 1)[1:][:,1:]
    matrice_redressment[2][2] = 1
    transformation = image_centralizer(image, matrice_redressment)
    return transformation



def rescale_contrast(image_np, min_val, max_val):
    rescaled_image = image_np * (max_val - min_val) + min_val
    return rescaled_image


def limits(image_ref_shape, image, shifts):
    height, width = image.shape
    h,w = image_ref_shape
    scale_x = width/w
    scale_y = height/h


    return int(np.floor((width)/2/max(shifts)/scale_x*1/3)), int(np.floor((height)/max(shifts)/scale_y/2/3))


def change_opacity(image, opacity):
    img = Image.fromarray(image).convert("RGBA")

    new_img = Image.new("RGBA", img.size)
    
    new_img.paste(img, (0, 0), img)

    alpha = new_img.getchannel("A")

    alpha = alpha.point(lambda p: p * opacity / 255)
    
    new_img.putalpha(alpha)
    return np.array(new_img)




def convert_to_color(image, color=(255, 255, 0)):
    height, width = image.shape
    colored_image = np.full((height, width, 3), 255, dtype=np.uint8)
    colored_image[image == 0] = color
    return colored_image

def reduce_qr_size(qr_image, scale_factor):
    if len(qr_image.shape) == 2:
        pil_image = Image.fromarray(qr_image)
    else:
        pil_image = Image.fromarray(qr_image[:,:,:3])
    
    new_width = int(qr_image.shape[1] / scale_factor)
    new_height = int(qr_image.shape[0] / scale_factor)    
    resized_image = pil_image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    return np.array(resized_image)

def resize_to_match(image1, image2):
    image1 = image1
    if len(image1.shape) == 2:
        h1, w1 = image1.shape
        ch1 = 1
    else:
        h1, w1, ch1 = image1.shape

    if len(image2.shape) == 2:
        h2, w2 = image2.shape
        ch2 = 1
    else:
        h2, w2, ch2 = image2.shape
    
    new_width = max(w1, w2)
    new_height = max(h1, h2)
    
    channels = max(ch1, ch2)
    if channels == 1:
        result = np.full((new_height, new_width), 0, dtype=np.uint8)
    else:
        result = np.full((new_height, new_width, channels), 0, dtype=np.uint8)
    
    x_center = (new_width - w1) // 2
    y_center = (new_height - h1) // 2
    
    if ch1 == 1 and channels > 1:
        for c in range(channels):
            result[y_center:y_center + h1, x_center:x_center + w1, c] = image1
    else:
        if channels == 1:
            result[y_center:y_center + h1, x_center:x_center + w1] = image1
        else:
            result[y_center:y_center + h1, x_center:x_center + w1] = image1
    h_,w_ = image1.shape
    h,w = image2.shape
    image2[int(h/2 - h_/2): int(h/2 + h_/2), int(w/2 - w_/2): int(w/2 + w_/2)] = 1
    return result, image1, image2

def merge_qr_and_cloud(qr_image, cloud_image, color=(0, 0, 0), qr_scale=1):
    colored_cloud = convert_to_color(cloud_image, color)
    
    if len(qr_image.shape) == 2:
        qr_image = np.stack([qr_image] * 3, axis=-1)
    
    resized_qr, cloud = resize_to_match(qr_image, colored_cloud)
    
    qr_mask = np.all(resized_qr < 128, axis=-1)
    
    result = colored_cloud.copy()
    result[qr_mask] = [0, 0, 0]

    return result

def merge_seal_and_cloud(seal_image, cloud_image, color=(0, 0, 0), qr_scale=1):
    resized_qr, seal, cloud = resize_to_match(seal_image, cloud_image)
    cloud = cloud//np.max(cloud)

    result = resized_qr+cloud
    h,w = result.shape
    h_seal,w_seal = seal.shape

    result[int(h/2 - h_seal/2)-18: int(h/2 - h_seal/2), int(w/2 - w_seal/2)-18: int(w/2 - w_seal/2)] = 1
    result[int(h/2 + h_seal/2): int(h/2 + h_seal/2)+18, int(w/2 + w_seal/2): int(w/2 + w_seal/2)+18] = 1
    return result



    
def reshape_to_match(image1, image2):
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    new_image_width = max(w1, w2)
    new_image_height = max(h1, h2)

    return cv.resize(image1, (new_image_width, new_image_height)),cv.resize(image2, (new_image_width, new_image_height))
    

def gray_to_png(image):
    normalized = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return cv.cvtColor(normalized, cv.COLOR_GRAY2BGR)

def reday_printing(image,threshold):
    image[image<=threshold]=0
    image[image>threshold]=255
    return image




def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        #row,col,ch = image.shape
        s_vs_p = 0.005
        amount = 0.00004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy