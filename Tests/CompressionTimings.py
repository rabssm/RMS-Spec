# RPi Meteor Station
# Copyright (C) 2015  Dario Zubovic
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Timings of compression algorithm with various cases.
"""

from __future__ import print_function, division, absolute_import

from RMS.Compression import Compressor
import RMS.ConfigReader as cr
import numpy as np
import time
import sys

config = cr.parse(".config")
comp = Compressor(None, None, None, None, None, config)


# IMAGE SIZE
WIDTH = 1280
HEIGHT = 720

ITERATIONS = 5

def timing(img):
    t = time.time()
    comp.compress(img)
    return time.time() - t

def timingOptimized(img):
    t = time.time()
    comp.compressOptimized(img)
    return time.time() - t
   
def create(f):

    arr = np.empty((256, HEIGHT, WIDTH), np.uint8)

    for i in range(256):
        arr[i] = f()

    return arr


def black():
    return np.zeros((HEIGHT, WIDTH), np.uint8)

def white():
    return np.full((HEIGHT, WIDTH), 255, np.uint8)

def uniform():
    return np.random.uniform(0, 256, (HEIGHT, WIDTH))

def gauss():
    return np.random.normal(128, 2, (HEIGHT, WIDTH))


def test():

    func_list = [black, white, uniform, gauss]
    
    t = [[0, 0, 0, 0], [0, 0, 0, 0]]
    
    for i in range(4):

        arr = create(func_list[i])
        timing(arr) # warmup
        timingOptimized(arr) # warmup

        for n in range(ITERATIONS):
            t[0][i] += timing(arr)
            t[1][i] += timingOptimized(arr)

    print("Non-optimized:")
    print("Black:", t[0][0]/ITERATIONS)
    print("White:", t[0][1]/ITERATIONS)
    print("Uniform noise:", t[0][2]/ITERATIONS)
    print("Gaussian noise:", t[0][3]/ITERATIONS)
    print("Average:", (t[0][0]+t[0][1]+t[0][2]+t[0][3])/ITERATIONS/4)

    print("Optimized:")
    print("Black:", t[1][0]/ITERATIONS)
    print("White:", t[1][1]/ITERATIONS)
    print("Uniform noise:", t[1][2]/ITERATIONS)
    print("Gaussian noise:", t[1][3]/ITERATIONS)
    print("Average:", (t[1][0]+t[1][1]+t[1][2]+t[1][3])/ITERATIONS/4)
    

if __name__ == "__main__":
    
    test()
