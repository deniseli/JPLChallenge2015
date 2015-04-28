# LS (Landing Spot) Finder
# author: Denise Li
#
# Reads in a low resolution DEM (digital elevation map) in .ply format and a
# high resolution DI (descent image) in .pgm format to find valid landing spots
# in the given terrain. A valid landing spot has:
# 1. Lander angle to vertical < 10 degrees for all orientations about vertical
#    at that site.
# 2. No part of the surface is > 0.39 meters above any of the 4 planes defined
#    by triplets of landing footpads.
# The lander has:
# 1. 3.4 meter diameter base plate
# 2. 4 * 0.5 meter footpads at 90 degrees around the base plate
# 3. 0.39 meters space between bottom of footpads and base plate

from bisect import bisect_left
from matplotlib import pyplot as plt

import bisect
import numpy as np
import numpy.linalg as la
import random
import re

def binary_search(a, x, lo=0, hi=None):   # can't use a to specify default for hi
    hi = hi if hi is not None else len(a) # hi defaults to len(a)   
    pos = bisect_left(a,x,lo,hi)          # find insertion position
    return (pos if pos != hi and a[pos] == x else -1) # don't walk off the end

def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def solution_to_pgm(dir, solution):
    hdrstr = "P5\n1000\n1000\n255\n"
    pixels = []
    for i in range(1000):
        for j in range(1000):
            pixels.append(solution[i, j])
    f = open('../solutions/' + dir + '.pgm', 'w')
    f.write(hdrstr + ''.join(map(chr, pixels)))
    f.close()

def get_rot_matrix(theta):
    mult = theta / abs(theta)
    return np.array([[np.cos(theta), -mult * np.sin(theta)], [mult * np.sin(theta), np.cos(theta)]])

def point_shift(point, do_compress):
    # Translate point for rotating about the origin
    point_n = point - center

    # Rotate point about the z-axis
    rot_matrix1 = get_rot_matrix(np.arctan(n[1] / n[0]))
    point_2 = rot_matrix1.dot(np.array([point_n[0], point_n[1]])) # temp var: rotate about z-axis
    point_xz = np.array([point_2[0], point_2[1], point_n[2]]) # point rotated into the xz plane
    new_n = rot_matrix1.dot(np.array([n[0], n[1]])) # temp var: rotate n(x, y) about z-axis

    # Rotate point about the y-axis
    rot_matrix2 = get_rot_matrix(np.arctan(new_n[0] / n[2]))
    point_3 = rot_matrix2.dot(np.array([point_xz[0], point_xz[2]])) # temp var: rotate about y-axis
    point_aligned = np.array([point_3[0], point_xz[1], point_3[1]])

    # Flip y and z
    point_aligned[1] *= -1
    point_aligned[2] *= -1

    # Stretch/compress to fit within bounds of image dimensions
    if do_compress:
        point_aligned[0] -= minx
        point_aligned[0] *= 500 / (maxx - minx + 1)
        point_aligned[1] -= miny
        point_aligned[1] *= 500 / (maxy - miny + 1)
        point_aligned[2] -= minz

    return point_aligned

def dist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

np.set_printoptions(suppress=True)

# The directories holding the training data
dirs = ['terrainS0C0R10_100', 'terrainS4C0R10_100', 'terrainS4C4R10_100', 'terrainS4C4R20_100']
dir = dirs[0]

# Read DI (.pgm) file
print 'Reading DI file...'
DIfname = '.pgm'
DI = read_pgm('../data/' + dir + '/' + dir + DIfname)
plt.imshow(DI)
plt.show()

# Read DEM (.ply) file
print 'Reading DEM file...'
DEMfname = '_500by500.ply'
with open('../data/' + dir + '/' + dir + DEMfname) as f:
    DEMfcontent = f.readlines()

# Parse DEMfcontent and save vertices in dictionary, with x and y as keys
# Break into 10x10 meter chunks to speed up nearest-neighbor search
print 'Parsing DEM file...'
DEM_raw = {}
buckets = []
center = np.zeros((3,))
for i in range(18, len(DEMfcontent)):
    vals = DEMfcontent[i].split()
    bucket = (10 * int(float(vals[0]) / 10), 10 * int(float(vals[1]) / 10))
    center += np.array([float(vals[0]), float(vals[1]), float(vals[2])]) / (len(DEMfcontent) - 18)
    if binary_search(buckets, bucket) == -1:
        DEM_raw[bucket] = {}
        bisect.insort_left(buckets, bucket)
    DEM_raw[bucket][(float(vals[0]), float(vals[1]))] = float(vals[2])

# Estimate the average normal (vertical) for this DEM by finding the average
# across X random planes
print 'Estimating the average normal (vertical) for this DEM (using X = 2000)...'
X = 2000
vertical = np.zeros((3,))
for i in range(X):
    rand_buckets = random.sample(buckets, 3)
    plane_points = []
    for bucket in rand_buckets:
        key = random.choice(DEM_raw[bucket].keys())
        plane_points.append(np.array([key[0], key[1], DEM_raw[bucket][key]]))
    vecA = plane_points[1] - plane_points[0]
    vecB = plane_points[2] - plane_points[0]
    n = np.cross(vecA, vecB)
    n = n / la.norm(n) # normalize n
    if n[0] < 0:
        n *= -1 # correct direction of n
    vertical += n
vertical /= la.norm(vertical)

# Find mins and maxes of the rotated points
print 'Finding mins and maxes to do final adjustments for DEM...'
minx = 0
maxx = 0
miny = 0
maxy = 0
minz = 0
for bucket in buckets:
    for key in DEM_raw[bucket].keys():
        vals = point_shift(np.array([key[0], key[1], DEM_raw[bucket][key]]), False)
        if vals[0] < minx:
            minx = float(vals[0])
        if vals[0] > maxx:
            maxx = float(vals[0])
        if vals[1] < miny:
            miny = float(vals[1])
        if vals[1] > maxy:
            maxy = float(vals[1])
        if vals[2] < minz:
            minz = vals[2]

# Moving adjusted DEM data into bucketed structure
print 'Calculating final values for DEM...'
DEM_points = {}
for i in range(100):
    for j in range(100):
        DEM_points[(i * 10, j * 10)] = {}
for bucket in buckets:
    for key in DEM_raw[bucket].keys():
        p = point_shift(np.array([key[0], key[1], DEM_raw[bucket][key]]), True)
        new_bucket = (10 * int(p[0] / 10), 10 * int(p[1] / 10))
        DEM_points[new_bucket][(p[0], p[1])] = p[2]

# Build final estimated surface using nearest neighbor search on DEM_points
print 'Building 500x500 grid DEM using nearest neighbor search on finalized DEM points...'
DEM = np.zeros((500, 500))
for i in range(500):
    for j in range(500):
        bucket = (10 * int(i / 10), 10 * int(j / 10))
        # Find nearest neighbor to i, j
        nearest_x = 0
        nearest_y = 0 
        nearest_dist = 99999999
        found = False
        for key in DEM_points[bucket].keys():
            if dist(key, (i, j)) < nearest_dist:
                nearest_dist = dist(key, (i, j))
                nearest_x = key[0]
                nearest_y = key[1]
                found = True
        if found:
            DEM[i,j] = DEM_points[bucket][(nearest_x, nearest_y)]
        else:
            DEM[i,j] = minz

plt.imshow(DEM)
plt.show()
