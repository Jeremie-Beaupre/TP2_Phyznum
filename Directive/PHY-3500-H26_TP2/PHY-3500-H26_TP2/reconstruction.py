#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np
import time
import math as mt
import cmath as cmt
import matplotlib.pyplot as plt
import scipy.interpolate as interp

# local files
import geometry as geo
import util as util
import CTfiltre as CTfilter

## créer l'ensemble de données d'entrée à partir des fichiers
def readInput():
    # lire les angles
    [nbprj, angles] = util.readAngles(geo.anglesFile)

    print("nbprj:",nbprj)
    print("angles min and max (rad):")
    print("["+str(np.min(angles))+", "+str(np.max(angles))+"]")

    # lire le sinogramme
    [nbprj2, nbpix2, sinogram] = util.readSinogram(geo.sinogramFile)

    if nbprj != nbprj2:
        print("angles file and sinogram file conflict, aborting!")
        exit(0)

    if geo.nbpix != nbpix2:
        print("geo description and sinogram file conflict, aborting!")
        exit(0)

    return [nbprj, angles, sinogram]


## reconstruire une image TDM en mode retroprojection
def laminogram():
    
    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    # "etaler" les projections sur l'image
    # ceci sera fait de façon "voxel-driven"
    # pour chaque voxel, trouver la contribution du signal reçu
    # voxel‑driven laminogram
    # paramètres du détecteur
    L = geo.nbpix * geo.voxsize          # largeur physique du détecteur
    tmin = -L/2                           # début du détecteur
    dt = L / geo.nbpix                    # taille d’un pixel détecteur
    coords = (np.arange(geo.nbvox) - geo.nbvox/2 + 0.5) * geo.voxsize
    X, Y = np.meshgrid(coords, coords)   # shape: (nbvox, nbvox)

    for a, th in enumerate(angles):

        print(f"Processing angle {a+1}/{len(angles)}")

        # IMPORTANT: ensure angles are in radians
        # th = np.deg2rad(th)  # Uncomment if needed

        # Project all voxels at once
        S = X * np.cos(th) + Y * np.sin(th)

        # Convert to detector index
        K = np.round((S - tmin) / dt).astype(int)

        # Mask valid indices
        valid = (K >= 0) & (K < geo.nbpix)

        # Add sinogram contribution
        image[valid] += sinogram[a, K[valid]]

    # Flip image (if required by your convention)
    image = np.fliplr(image)
    CTfilter.filterSinogram(image)
    image = np.fft.fft2(image)
    image = np.fft.fftshift(image)


    # Apply filter if needed
    #

    util.saveImage(np.abs(image), "laminogram_fantome_192_04_image_droite_test_question3")



## reconstruire une image TDM en mode retroprojection filtrée
def backproject():

    [nbprj, angles, sinogram] = readInput()

    # filtrer le sinogramme (question 3)
    CTfilter.filterSinogram(sinogram)

    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    # paramètres du détecteur
    L = geo.nbpix * geo.voxsize      # largeur physique du détecteur
    tmin = -L/2                       # début du détecteur
    dt = L / geo.nbpix                # taille d’un pixel détecteur

    # rétroprojection filtrée voxel-driven

    # --- grille des voxels ---
    coords = (np.arange(geo.nbvox) - geo.nbvox/2 + 0.5) * geo.voxsize
    X, Y = np.meshgrid(coords, coords)

    image = np.zeros((geo.nbvox, geo.nbvox))

    for a, th in enumerate(angles):

        cos_th = np.cos(th)
        sin_th = np.sin(th)

        # projection de tous les voxels
        S = X * cos_th + Y * sin_th

        Kfloat = (S - tmin) / dt

        K1 = np.floor(Kfloat).astype(int)
        K2 = np.ceil(Kfloat).astype(int)

        # choix du plus proche (équivalent à ton if abs(...))
        choose_k1 = np.abs(K1 - Kfloat) <= np.abs(K2 - Kfloat)
        K = np.where(choose_k1, K1, K2)

        # masque validité détecteur
        valid = (K >= 0) & (K < geo.nbpix)

        # accumulation
        image[valid] += sinogram[a, K[valid]]

    # remettre l'image à l'endroit
    image = np.fliplr(image)

    util.saveImage(image, "fbp_test1")

## reconstruire une image TDM en mode retroprojection filtrée + interpolation
def backproject2():

    [nbprj, angles, sinogram] = readInput()

    # filtrer le sinogramme (question 3)
    CTfilter.filterSinogram(sinogram)

    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    # paramètres du détecteur
    L = geo.nbpix * geo.voxsize      # largeur physique du détecteur
    tmin = -L/2                       # début du détecteur
    dt = L / geo.nbpix                # taille d’un pixel détecteur

    # rétroprojection filtrée voxel-driven


# Création de la grille des voxels avec meshgrid
    coords = (np.arange(geo.nbvox) - geo.nbvox/2 + 0.5) * geo.voxsize
    X, Y = np.meshgrid(coords, coords)   # X -> colonnes, Y -> lignes

    image = np.zeros((geo.nbvox, geo.nbvox))

    # rétroprojection filtrée voxel-driven (vectorisée sur les voxels)
    for a, th in enumerate(angles):

        # projection de tous les voxels en même temps
        S = X * np.cos(th) + Y * np.sin(th)

        K = (S - tmin) / dt
        K1 = np.floor(K).astype(int)
        K2 = np.ceil(K).astype(int)

        # masque des indices valides
        valid = (K1 >= 0) & (K1 < geo.nbpix) & (K2 != K1)

        contrib = np.zeros_like(S)

        # interpolation linéaire
        contrib[valid] = (
            sinogram[a, K1[valid]] +
            (K[valid] - K1[valid]) / (K2[valid] - K1[valid]) *
            (sinogram[a, K2[valid]] - sinogram[a, K1[valid]])
        )

        # cas sans interpolation (ou bord)
        valid_simple = (K1 >= 0) & (K1 < geo.nbpix) & (~valid)
        contrib[valid_simple] = sinogram[a, K1[valid_simple]]

        image += contrib
    image = np.fliplr(image)
    util.saveImage(image, "fbp_test1")




## reconstruire une image TDM en mode retroprojection
def reconFourierSlice():
    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite, complexe
    # pour qu'elle puisse contenir sa version FFT d'abord
    IMAGE = np.zeros((geo.nbvox, geo.nbvox), 'complex')
    # conteneur pour la FFT du sinogramme
    SINOGRAM = np.zeros((336, 336), 'complex')
    print(SINOGRAM.shape)
    #image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))
    #votre code ici

    N = geo.nbpix
    sinoX = np.linspace(-N/2, N/2, N)
    #sinoX = np.fft.fftshift(np.fft.fftfreq(N, d=geo.voxsize))

    Hx = []
    Hy = []
    Hz = []
    for a, th in enumerate(angles):
        print(f"angle {th}")
        sinoTF = np.fft.fft(sinogram[a,:])
        sinoTF = np.fft.fftshift(sinoTF)
        #sinoTF = sinoTF/max(abs(sinoTF))
        sinoX_peak = sinoX[np.argmax(np.abs(sinoTF))]
        sinoX_center = sinoX - sinoX_peak

        for b, w in enumerate(sinoX_center):

            kx = (336/2-1) + w*np.cos(-th)
            ky = (336/2-1) + w*np.sin(-th)

            

            if 0 <= ky < SINOGRAM.shape[0] and 0 <= kx < SINOGRAM.shape[1]:
                
                #SINOGRAM[kx, ky] = sinoTF[b]
                Hx.append(kx)
                Hy.append(ky)
                Hz.append(sinoTF[b])

                


                #print(SINOGRAM[kx, ky])
                # plt.plot(sinoX_center, sinoTF)
                # plt.show()
       

            #print(f"w = {w}, k_x = {kx}, k_y = {ky}, angle = {th}")
        #print(np.real(SINOGRAM)) 

    # print("Hx range:", np.min(Hx), np.max(Hx))
    # print("Hy range:", np.min(Hy), np.max(Hy))
    # plt.scatter(Hx, Hy, c=np.real(Hz), cmap='viridis', s=1)
    # plt.colorbar(label='Value')
    # plt.xlim(0, 336)
    # plt.ylim(0, 336)
    # plt.xlabel('kx')
    # plt.ylabel('ky')
    # plt.title('Fourier Slice')
    # plt.show()

    # Interpolate values onto the grid
    # 1️⃣ Create a regular grid for interpolation


    Hx = np.array(Hx)
    Hy = np.array(Hy)
    Hz = np.array(Hz)

    grid_size = 300  # adjust resolution
    grid_x = np.linspace(Hx.min(), Hx.max(), grid_size)
    grid_y = np.linspace(Hy.min(), Hy.max(), grid_size)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

    # 2️⃣ Interpolate scattered data onto the grid
    Z_grid = interp.griddata(
        points=(Hx, Hy),
        values=np.real(Hz),  # in case your values are complex
        xi=(X_grid, Y_grid),
        method= 'linear'      # 'linear', 'nearest', 'cubic'
    )

    Z_grid = np.nan_to_num(Z_grid, nan=0.0)

    # 3️⃣ Plot interpolated heatmap
    plt.figure(figsize=(8,6))
    plt.imshow(
        Z_grid,
        extent=(Hx.min(), Hx.max(), Hy.min(), Hy.max()),
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )
    plt.colorbar(label='Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interpolated Heatmap')
    plt.show()

   
    Z_reconstructed = np.fft.ifft2(Z_grid)
    Z_reconstructed = np.real(Z_reconstructed)
    plt.imshow(Z_reconstructed, cmap='gray')
    plt.colorbar(label='Reconstructed value')
    plt.title("Inverse FFT")
    plt.show() 

    util.saveImage(abs(Z_grid), "fft")



## reconstruire une image TDM en mode retroprojection
def reconFourierSlice2():
    [nbprj, angles, sinogram] = readInput()

    # conteneur pour la FFT du sinogramme
    SINOGRAM = np.zeros((geo.nbpix, geo.nbpix), 'complex')
    print(SINOGRAM.shape)
    #image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))
    #votre code ici

    N = geo.nbpix
    ##sinoX = np.arange(N) - N/2
    ##print(sinoX)
    sinoX = np.fft.fftfreq(N) * N
    # 1. Préparer une grille de comptage pour normaliser
    counts = np.zeros_like(SINOGRAM, dtype=float)

    for a, th in enumerate(angles):
        sinoTF = np.fft.fft(sinogram[a,:])
        
        for b, w in enumerate(sinoX):
            # Utilisation de th en radians si ce n'est pas déjà le cas
            kx = round((N)//2 + w * np.cos(th))
            ky = round((N)//2 + w * np.sin(th))

            if 0 <= ky < N and 0 <= kx < N:
                SINOGRAM[ky, kx] += sinoTF[b]
                counts[ky, kx] += 1


    # 2. Normalisation pour éviter l'accumulation au centre
    SINOGRAM[counts > 0] /= counts[counts > 0]

    # 3. Retour dans l'espace spatial
    SINOGRAM = np.fft.ifftshift(SINOGRAM)

    #image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(image)))
    #image = np.fft.ifftshift(np.fft.ifft2(SINOGRAM))
    image = np.fft.ifft2(SINOGRAM)
    util.saveImage(np.abs(image), "fft")

# def showFilteredSinogram():

#     [nbprj, angles, sinogram] = readInput()

#     # filtrer le sinogramme (question 3)
#     CTfilter.filterSinogram(sinogram)

#     # sauvegarder l’image du sinogramme filtré
#     util.saveImage(sinogram, "sinogram_filtre_q3")


## main ##
start_time = time.time()
#laminogram()
#showFilteredSinogram()
#backproject()
#backproject2()

reconFourierSlice2()

print("--- %s seconds ---" % (time.time() - start_time))

