#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# fichier contenant la description de la géométrie
# d'acquisition
# et de reconstruction

import numpy as np

### VARIABLES ###

### paramètres d'acquisition ###

## largeur d'un élément de détecteur (cm)
pixsize = 0.165

## taille du détecteur (nombre d'échantillons)
nbpix = 336

### paramètres de reconstruction ###

## taille de la grille d'image (carrée)
nbvox = 384 # options: 96, 192

## taille du voxel (carré) (cm)
voxsize = 0.2# option: 0.4, 0.2

## fichiers d'entrée
dataDir = "./data/"
anglesFile = r"C:\École\Université\Session 8 (H26)\PhysNum\TP_2\TP2_Phyznum\Directive\PHY-3500-H26_TP2\PHY-3500-H26_TP2\angles.txt"
sinogramFile = r"C:\École\Université\Session 8 (H26)\PhysNum\TP_2\TP2_Phyznum\Directive\PHY-3500-H26_TP2\PHY-3500-H26_TP2\sinogram-password.txt"

