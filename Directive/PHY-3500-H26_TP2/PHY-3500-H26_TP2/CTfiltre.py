#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np

## filtrer le sinogramme
## ligne par ligne
def filterSinogram(sinogram):
    for i in range(sinogram.shape[0]):
        sinogram[i] = filterLine(sinogram[i])

## filter une ligne (projection) via FFT
def filterLine(projection):

    # FFT de la projection
    P = np.fft.fft(projection)

    # fréquences associées à chaque coefficient FFT
    freqs = np.fft.fftfreq(len(projection))

    # filtre passe-haut demandé : |u|
    H = np.abs(freqs)

    # appliquer le filtre dans le domaine fréquentiel
    Pf = P * H

    # retour dans le domaine spatial (partie réelle)
    filtered = np.fft.ifft(Pf).real

    return filtered

