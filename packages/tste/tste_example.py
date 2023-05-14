#!/usr/bin/env python
import tste
import scipy.io

tste.USING_THEANO=False

music_triplets = scipy.io.loadmat("./STE_Release/data/music_triplets.mat")['triplets']
# Python variables are 0-indexed:
music_triplets -= 1

# Learn the embedding from the triplets:
embedding = tste.tste(music_triplets, no_dims=2)
# embedding is now a 412x2 array