# -*- coding: Latin-1 -*-
# TP optim : maximisation of the area
# par l'algorithme PSO
# Peio Loubiere & Rachid Chelouah pour l'EISTI
# septembre 2017
# usa : python yourMethod.py 
from scipy import *
from math import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
import pyclipper

# Visualization of the parcel
fig = plt.figure()
canv = fig.add_subplot(1,1,1)
canv.set_xlim(0,500)
canv.set_ylim(0,500)

# ************ Parameters of metaheuristics *************
Nb_Cycles = 
Nb_Indiv =
# ***********************************************************

# ***************** Problem settings ******************
#  Different proposals of parcels: 
polygon = ((10,10),(10,400),(400,400),(400,10)) 
# polygon = ((10,10),(10,300),(250,300),(350,130),(200,10)) 
# polygon = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
# polygon = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))

# ***********************************************************

# Transform the  polygon in list for display.
def poly2list(polygon):
	polygonfig = list(polygon)
	polygonfig.append(polygonfig[0])
	return polygonfig

# Drawable polygon variable
polygonfig = poly2list(polygon)

# Display window
def draw(polyfig,rectfig):
	global canv, codes
	canv.clear()
	# Draw of the polygon
	codes = [Path.MOVETO]
	for i in range(len(polyfig)-2):
	  codes.append(Path.LINETO)   
	codes.append(Path.CLOSEPOLY)
	path = Path(polyfig, codes)
	patch = patches.PathPatch(path, facecolor='orange', lw=2)
	canv.add_patch(patch)

	# Rectangle drawing
	codes = [Path.MOVETO]
	for i in range(len(rectfig)-2):
	  codes.append(Path.LINETO)   
	codes.append(Path.CLOSEPOLY)
	path = Path(rectfig, codes)
	patch = patches.PathPatch(path, facecolor='grey', lw=2)
	canv.add_patch(patch)

	# Title display (rectangle area)
	plt.title("area : {}".format(round(area(rectfig[:-1]),2)))

	plt.draw()
	plt.pause(0.1)


# Collect bounding box bounds around the parcel
def getBounds(polygon):

# Transformation of a problem solution into a rectangle for clipping
# Return the rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
def pos2rect(pos):

# Distance between two points (x1,y1), (x2,y2)
def distance(p1,p2):

# Area of the rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
# 	= distance AB * distance BC
def area((pa, pb, pc, pd)):

# Clipping
# Predicate that verifies that the rectangle is in the polygon
# Test if 
# 	- there is an intersection (!=[]) between the figures and
#	- both lists with the same length
# 	- all the points of the rectangle belong to the result of clipping 
# If error (~flat angle), return false
def verifConstraint(rect, polygon):

# Creates a feasible particle (solution)
# ua particle is described by your metaheuristics : 
# 	- pos : solution list of variables
#	- eval :  rectangle area
#	- ... : other components of the solution
def initOne(polygon):

# Init of the population
def initPop(nb,polygon):
	return [initOne(polygon) for i in range(nb)]

# Returns the best particle depends on the metaheuristic
def bestPartic(p1,p2):

# Return a copy of the best particle of the population
def getBest(population):
	return dict(reduce(lambda acc, e: bestPartic(acc,e),population[1:],population[0]))


# *************************************** optimization algorithm***********************************
# Bounds calculating for initialization
xmin,xmax,ymin,ymax = getBounds(polygon)
# initialization of the population (of the agent if simulated annealing) and the best individual.
pop = initPop(Nb_Indiv,polygon)
best = getBest(pop)

# main loop (to be refined according to the metaheuristic / the chosen convergence criterion)
for i in range(Nb_Cycles):
	#displacement

	# Update of the best solution and display
	

# END : dis^play
