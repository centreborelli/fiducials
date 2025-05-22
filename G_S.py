import numpy as np
import cv2 as cv
import sys
import os

import autocorrelation as autoc
import shifts_mesure as shm
import rectification_methods as rectif
import optimization as optmiz
import operations as oprt


def detect_ghostseal(image, sigma=1, gamma=0, method='gaussian', return_peaks=False, verbose=True):
    autocorr_img = autoc.autocorrelation_display(image)
    peaks = shm.peaks_and_shifts_subpixelic(autocorr_img, sigma, gamma, method)

    detected = len(peaks) > 0

    if verbose:
        print(f"[GhostSeal Detection] Peaks detected: {len(peaks)} → {'YES' if detected else 'NO'}")

    if return_peaks:
        return detected, peaks
    return detected


def estimate_ghostseal_homography(reference_image, deformed_image, U, V, iteration,method,w, h, N_blocs, sliding , step):
    total_costs = []
    total_h_estim = np.eye(3,3)

    for i in range(iteration):
        print(f"--- Iteration {i+1} ---")
        local_affinities,positions_finales, patch_centers = rectif.redressing_random_patches(deformed_image, U, V, reference_image, w, h, N_blocs, 1, 0, method, sliding, step)
        H, cost = optmiz.optimize(
            [aff[:2, :2] for aff in local_affinities],
            [vec.T[0] for vec in positions_finales],
            filter_function=lambda x: True
        )

        H_estim = np.vstack((H.T, np.array([[0, 0, 1]]))).T
        H_estim /= H_estim[2, 2]  # Normalisation

        rectified_image = oprt.apply_homography(
            deformed_image,
            oprt.image_centralizer(reference_image, H_estim),
            reference_image.shape
        )
        total_h_estim = H_estim @ total_h_estim 
        print("Cost:", cost)
        total_costs.append(cost)
    
    return rectified_image, total_h_estim






