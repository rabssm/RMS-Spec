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

array_pad = 1 if (256*config.width*config.height)%(512*1024) == 0 else 0

ITERATIONS = 2
INNER_ITERATIONS = 2

def timing(img):
    t = time.time()
    comp.compress(img)
    return time.time() - t

def timingOptimized(img):
    t = time.time()
    comp.compressOptimized(img)
    return time.time() - t
   
def create(f):
    arr = np.empty((256, HEIGHT+array_pad, WIDTH+array_pad), np.uint8)

    for i in range(256):
        arr[i] = f()

    return arr


def black():
    return np.zeros((HEIGHT+array_pad, WIDTH+array_pad), np.uint8)

def white():
    return np.full((HEIGHT+array_pad, WIDTH+array_pad), 255, np.uint8)

def uniform():
    return np.random.uniform(0, 256, (HEIGHT+array_pad, WIDTH+array_pad))

def gauss():
    return np.random.normal(128, 2, (HEIGHT+array_pad, WIDTH+array_pad))


def test():

    func_list = [black, white, uniform, gauss]
    
    t = np.zeros((2, 4), dtype=np.float)
    
    for iteration in range(ITERATIONS):
        for i in range(4):
            arr = create(func_list[i])
            timing(arr) # warmup
            timingOptimized(arr) # warmup

            for n in range(INNER_ITERATIONS):
                t[0][i] += timing(arr)
                t[1][i] += timingOptimized(arr)

    t /= (ITERATIONS * INNER_ITERATIONS)

    print("Non-optimized:")
    print("Black:", t[0][0])
    print("White:", t[0][1])
    print("Uniform noise:", t[0][2])
    print("Gaussian noise:", t[0][3])
    print("Average:", np.mean(t[0]))

    print("Optimized:")
    print("Black:", t[1][0])
    print("White:", t[1][1])
    print("Uniform noise:", t[1][2])
    print("Gaussian noise:", t[1][3])
    print("Average:", np.mean(t[1]))
    

if __name__ == "__main__":
    
    test()
