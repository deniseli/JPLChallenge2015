# LS (Landing Spot) Finder
# author: Denise Li
#
# Reads in a low resolution DEM (digital elevation map) in .ply format to find
# valid landing spots in the given terrain. A valid landing spot has:
# 1. Lander angle to vertical < 10 degrees for all orientations about vertical
#    at that site.
# 2. No part of the surface is > 0.39 meters above any of the 4 planes defined
#    by triplets of landing footpads.
# The lander has:
# 1. 3.4 meter diameter base plate
# 2. 4 * 0.5 meter footpads at 90 degrees around the base plate
# 3. 0.39 meters space between bottom of footpads and base plate

from matplotlib import pyplot as plt
from multiprocessing import Pool

import math
import numpy as np
import numpy.linalg as la
import re
import time

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
    for i in xrange(1000):
        for j in xrange(1000):
            pixels.append(solution[i, j])
    f = open('../data/' + dir + '/solution' + dir[7:] + '.pgm', 'w')
    f.write(hdrstr + ''.join(map(chr, pixels)))
    f.close()

def euclideanDist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def getCircleIndices(c, r, res, minval, maxval):
    points = []
    r_new = int(math.ceil(r/res))
    for i in xrange(min(maxval, max(minval, c[0]-r_new)), min(maxval, max(minval, c[0]+r_new))):
        for j in xrange(min(maxval, max(minval, c[1]-r_new)), min(maxval, max(minval, c[1]+r_new))):
            p = np.array([i, j])
            if sol[i, j] == 255 and euclideanDist(p, c + np.array([0.5, 0.5])) <= r/res:
                points.append(p)
    return points

def countErrors():
    falsepos = 0 # sol labels a pixel 255 but it should be 0
    falseneg = 0 # sol labels a pixel 0 but it should be 255
    for i in xrange(1000):
        for j in xrange(1000):
            if sol[i,j] == 255 and realSol[i,j] == 0:
                falsepos += 1
            if sol[i,j] == 0 and realSol[i,j] == 255:
                falseneg += 1
    nonzeros = float(np.count_nonzero(realSol)) / 100
    print 'False positives: ' + str(falsepos) + ' = ' + str(falsepos / (10000. - nonzeros)) + ' %'
    print 'False negatives: ' + str(falseneg) + ' = ' + str(falseneg / nonzeros) + ' %'
    print 'Total error: ' + str(falsepos + falseneg) + ' = ' + str((falsepos + falseneg) / 10000.) + ' %'

def slope(i, cols):
    sAngle = 0.0364 # 10 degrees translated to radians and this resolution, tweeked for accuracy
    sRadius = 9
    sRadius45 = 6
    sRadius30 = 4
    sRadius60 = 8
    for j in xrange(10, 490):
        edgepoints = [np.array([i, j + sRadius]), np.array([i - sRadius30, j + sRadius60]), \
                          np.array([i - sRadius45, j + sRadius45]), np.array([i - sRadius60, j + sRadius30]), \
                          np.array([i - sRadius, j]), np.array([i - sRadius60, j - sRadius30]), \
                          np.array([i - sRadius45, j - sRadius45]), np.array([i - sRadius30, j - sRadius60]), \
                          np.array([i, j - sRadius]), np.array([i + sRadius30, j - sRadius60]), \
                          np.array([i + sRadius45, j - sRadius45]), np.array([i + sRadius60, j - sRadius30]), \
                          np.array([i + sRadius, j]), np.array([i + sRadius60, j + sRadius30]), \
                          np.array([i + sRadius45, j + sRadius45]), np.array([i + sRadius30, j + sRadius60])]
        for pIndex in xrange(16):
            # Build triangle starting from edgepoints[pIndex]
            # If the slope of the plane exceeds sAngle, block out the square and break
            p1 = edgepoints[pIndex]
            p2 = edgepoints[(pIndex + 4) % 16]
            p3 = edgepoints[(pIndex + 8) % 16]
            vecA = np.array([p2[0] - p1[0], p2[1] - p1[1], DEM[p2[0],p2[1]] - DEM[p1[0], p1[1]]])
            vecB = np.array([p2[0] - p3[0], p2[1] - p3[1], DEM[p2[0],p2[1]] - DEM[p3[0], p3[1]]])
            n = np.cross(vecA, vecB)
            angle = math.acos(n.T.dot(np.array([0, 0, -1])) / la.norm(n))
            if angle >= sAngle:
                for x in xrange(2):
                    for y in xrange(2*j, 2*j + 2):
                        cols[x,y] = 0
                break
    return cols

np.set_printoptions(suppress=True)

# The directories holding the training data
dirs = ['terrainS0C0R10_100', 'terrainS4C0R10_100', 'terrainS4C4R10_100', 'terrainS4C4R20_100']
dir = dirs[2]

# Read solution file
# 0.1 m/pixel
print 'Reading real solution file...'
realSolfname = '.invHazard.pgm'
realSol = read_pgm('../data/' + dir + '/' + dir + realSolfname)

# Read raw DEM (.raw) file
# 0.2 m/pixel
print 'Reading raw DEM file...'
DEMfname = '_500by500_dem.raw'
DEM = np.fromfile('../data/' + dir + '/' + dir + DEMfname, dtype='>f4', sep="")
DEM = DEM.reshape([500, 500])

# Initialize solution array
# 0.1 m/pixel
sol = np.zeros((1000, 1000), dtype=np.int)
sol[20:979,20:979].fill(255) # edges always disallowed

# Filter for slope
# If the maximum slope at a point is more than 10 degrees, we disallow all
# points in a 3.4x3.4 square around it
print 'Searching for points of extreme slope...'
startSlope = time.time()
pool = Pool()
sResults = [pool.apply_async(slope, args=(i, sol[i*2:i*2+2, :])) for i in xrange(10, 490)]
sOutput = [p.get() for p in sResults]
sol[20:980,:] = np.concatenate(sOutput, axis=0)

print 'Time to find points of extreme slope: ' + str(time.time() - startSlope)

# Filter for roughness
# If a point is more than 0.39 m above the points above, below, and to either
# side of it, we disallow all points within 3.4 m of it.
print 'Searching for points of roughness...'
startRoughness = time.time()
roughC = 0.36
rRadius = 11
for i in xrange(1, 499):
    for j in xrange(1, 499):
        istart = min(max(i - rRadius + 1, 0), 499)
        iend = min(max(i + rRadius, 0), 499)
        jstart = min(max(j - rRadius + 1, 0), 499)
        jend = min(max(j + rRadius, 0), 499)

        irange = max(1, min(i - istart, iend - i))
        jrange = max(1, min(j - jstart, jend - j))
        imult = np.arange(1., irange + 1.) / (irange + 1.)
        jmult = np.arange(1., jrange + 1.) / (jrange + 1.)
        
        ileft = DEM[istart:istart + irange, j]
        iright = DEM[istart + rRadius:istart + rRadius + irange, j]
        imaxheights = np.multiply(imult, ileft) + np.multiply(imult[::-1], iright)

        jtop = DEM[i, jstart:jstart + jrange]
        jbot = DEM[i, jstart + rRadius:jstart + rRadius + jrange]
        jmaxheights = np.multiply(jmult, jtop) + np.multiply(jmult[::-1], jbot)

        if DEM[i,j] - roughC > np.min(imaxheights) or DEM[i,j] - roughC > np.min(jmaxheights):
            # Set circle of points to 0
            for p in getCircleIndices(np.array([i*2, j*2]), 1.9, 0.1, 20, 979):
                sol[p[0], p[1]] = 0

print 'Time to find points of roughness: ' + str(time.time() - startRoughness)

countErrors()
solution_to_pgm(dir, sol)

# Check correctness
plt.imshow(sol)
plt.show()
plt.imshow(sol.astype('int8') - realSol.astype('int8'))
plt.show()
