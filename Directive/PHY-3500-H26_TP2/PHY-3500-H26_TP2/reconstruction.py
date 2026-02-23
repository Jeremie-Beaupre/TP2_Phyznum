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

    for j in range(geo.nbvox):            # boucle colonnes
        print(f"working on image column: {j+1}/{geo.nbvox}")
        x =  (j - geo.nbvox/2 + 0.5) * geo.voxsize   # coordonnée x du voxel

        for i in range(geo.nbvox):        # boucle lignes
            y = (i - geo.nbvox/2 + 0.5) * geo.voxsize   # coordonnée y du voxel
            total = 0.0                   # accumulation des contributions

            for a, th in enumerate(angles):           # boucle angles
                s = x*np.cos(th) + y*np.sin(th)       # projection du voxel sur le détecteur
                k = int(round((s - tmin) / dt))       # conversion en index détecteur

                if 0 <= k < geo.nbpix:                # si dans les bornes
                    total += sinogram[a, k]           # ajouter la valeur du sinogramme

            image[i, j] = total           # assigner la valeur finale du voxel
            

    image = np.fliplr(image) # mettre l'image à l'endroit
    CTfilter.filterSinogram(image)
    util.saveImage(image, "laminogram_fantome_192_04_image_droite_test_question3")


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
        print(f"angle {a}/720")
        sinoTF = np.fft.fft(sinogram[a,:])
        sinoTF = np.fft.fftshift(sinoTF)
        sinoX_peak = sinoX[np.argmax(np.abs(sinoTF))]
        sinoX_center = sinoX - sinoX_peak

        for b, w in enumerate(sinoX_center):

            kx = (336/2-1) + w*np.cos(th)
            ky = (336/2-1) + w*np.sin(th)

            if 0 <= ky < SINOGRAM.shape[0] and 0 <= kx < SINOGRAM.shape[1]:
                
                #SINOGRAM[kx, ky] = sinoTF[b]

                Hx.append(kx)
                Hy.append(ky)
                Hz.append(sinoTF[b])


                #print(SINOGRAM[kx, ky])
                #plt.plot(sinoX_center, sinoTF)
                #plt.show()
        
            #print(f"w = {w}, k_x = {kx}, k_y = {ky}, angle = {th}")
        #print(np.real(SINOGRAM)) 



    # Interpolate values onto the grid
    # 1️⃣ Create a regular grid for interpolation


    Hx = np.array(Hx)
    Hy = np.array(Hy)
    Hz = np.array(Hz)

    grid_size = 100  # adjust resolution
    grid_x = np.linspace(Hx.min(), Hx.max(), grid_size)
    grid_y = np.linspace(Hy.min(), Hy.max(), grid_size)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

    # 2️⃣ Interpolate scattered data onto the grid
    Z_grid = interp.griddata(
        points=(Hx, Hy),
        values=np.real(Hz),  # in case your values are complex
        xi=(X_grid, Y_grid),
        method='linear'      # 'linear', 'nearest', 'cubic'
    )

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

    # plt.figure(figsize=(8,6))
    # plt.contourf(X_grid, Y_grid, Z_grid, levels=50, cmap='viridis')
    # plt.colorbar(label='Value')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Interpolated Contour Plot')
    # plt.show()

    # #util.saveImage(image, "fft")


    
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
reconFourierSlice()
print("--- %s seconds ---" % (time.time() - start_time))

