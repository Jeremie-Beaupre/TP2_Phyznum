#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np
import time
import math as mt

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
    SINOGRAM = np.zeros(sinogram.shape, 'complex')

    #image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))
    #votre code ici

    
    util.saveImage(image, "fft")

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
backproject()
#backproject2()
#reconFourierSlice()
print("--- %s seconds ---" % (time.time() - start_time))

