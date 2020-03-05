#!/usr/bin/python
# -*- coding: Latin-1 -*-
# TP optim : maximisation de surface
# par l'algorithme PSO
# Peio Loubiere pour l'EISTI
# septembre 2017
# usage : python surface.corr1.py 
# Modified by: GUSTAVO FLEURY SOARES & INDURAJ R.
# Heuristic - ADEO2 2019-2020 - EISTI
# 2019.12.05

from scipy import *
from math import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
import pyclipper
from functools import *
from shapely.geometry import Point, Polygon

# Figure de visualisation de la parcelle
fig = plt.figure()
canv = fig.add_subplot(1,1,1)
canv.set_xlim(0,500)
canv.set_ylim(0,500)

# ************ Paramètres de la métaheuristique *** S. Annealing Parameters
T0 = 10 # initial temperature
Tmin = 1e-3 # final temperature
tau = 1e4 # constant for temperature decay
Alpha = 0.9 # constant for geometric decay
Step = 7 # number of iterations on a temperature level
IterMax = 15000 # 15000 max number of iterations of the algorithm

sizeNeigh = 0.4 # Size of the Xmax-Xmin to randomize
# ***********************************************************

anglemin = 30
anglemax = 60

# ***************** Paramètres du problème ******************
# Différentes propositions de parcelles : 
# polygone = ((10,10),(10,400),(400,400),(400,10)) 
# polygone = ((10,10),(10,300),(250,300),(350,130),(200,10)) 
polygone = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
# polygone = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))

# ***********************************************************

poly = Polygon(polygone)

# Draw the figure of the graphs of:
def drawStats(Htime, Henergy, Hbest, HT, parametersName):
    # display des courbes d'evolution
    fig2 = plt.figure(2, figsize=(18,6))   
    plt.subplot(1,3,1)
    plt.semilogy(Htime, Henergy)
    plt.title("Evolution of the total energy of the system")
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.subplot(1,3,2)
    plt.semilogy(Htime, Hbest)
    plt.title('Evolution of the best distance')
    plt.xlabel('time')
    plt.ylabel('Distance')
    plt.subplot(1,3,3)
    plt.semilogy(Htime, HT, label=parametersName)
    plt.legend()
    plt.title('Evolution of the temperature of the system')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    fig2.suptitle('ChangeParameters')
    fig2.savefig( parametersName + '.pdf')
    # plt.show()

# Transforme le polygone en liste pour l'affichage.
def poly2list(polygone):
    polygonefig = list(polygone)
    polygonefig.append(polygonefig[0])
    return polygonefig

# Constante polygone dessinable
polygonefig = poly2list(polygone)

# Fenètre d'affichage
def dessine(polyfig,rectfig):
    global canv, codes
    canv.clear()
    # Dessin du polygone
    codes = [Path.MOVETO]
    for i in range(len(polyfig)-2):
      codes.append(Path.LINETO)   
    codes.append(Path.CLOSEPOLY)
    path = Path(polyfig, codes)
    patch = patches.PathPatch(path, facecolor='orange', lw=2)
    canv.add_patch(patch)
    canv.autoscale_view()


    # Dessin du rectangle
    codes = [Path.MOVETO]
    for i in range(len(rectfig)-2):
      codes.append(Path.LINETO)   
    codes.append(Path.CLOSEPOLY)
    path = Path(rectfig, codes)
    patch = patches.PathPatch(path, facecolor='grey', lw=2)
    canv.add_patch(patch)

    # Affichage du titre (aire du rectangle)
    plt.title("Aire : {}".format(round(aire(rectfig[:-1]),2)))

    
    plt.draw()
    plt.pause(0.1)

# Récupère les bornes de la bounding box autour de la parcelle
def getBornes(polygone):
    lpoly = list(polygone) #tansformation en liste pour parcours avec reduce
    #return reduce(lambda (xmin,xmax,ymin,ymax),(xe,ye): (min(xe,xmin),max(xe,xmax),min(ye,ymin),max(ye,ymax)),lpoly[1:],(lpoly[0][0],lpoly[0][0],lpoly[0][1],lpoly[0][1]))
    return reduce(lambda acc,e: (min(e[0],acc[0]),max(e[0],acc[1]),min(e[1],acc[2]),max(e[1],acc[3])),lpoly[1:],(lpoly[0][0],lpoly[0][0],lpoly[0][1],lpoly[0][1]))
# Transformation d'une solution du pb (centre/coin/angle) en rectangle pour le clipping 
# Retourne un rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
def pos2rect(pos):
    # coin : point A
    xa, ya = pos[0], pos[1]
    # centre du rectangle : point O
    xo, yo = pos[2], pos[3]
    # angle  AÔD
    angle = pos[4]

    # point D : rotation de centre O, d'angle alpha
    alpha = pi * angle / 180 # degre en radian
    xd = cos(alpha)*(xa-xo) - sin(alpha)*(ya-yo) + xo 
    yd = sin(alpha)*(xa-xo) + cos(alpha)*(ya-yo) + yo
    # point C : symétrique de A, de centre O
    xc, yc = 2*xo - xa, 2*yo - ya 
    # point B : symétrique de D, de centre O
    xb, yb = 2*xo - xd, 2*yo - yd

    # round pour le clipping
    return ((round(xa),round(ya)),(round(xb),round(yb)),(round(xc),round(yc)),(round(xd),round(yd)))

    
# Distance entre deux points (x1,y1), (x2,y2)
def distance(p1,p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Aire du rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
#     = distance AB * distance BC
def aire(p1):
    return distance(p1[0],p1[1])* distance(p1[2],p1[3])
#def aire((pa, pb, pc, pd)):
#    return distance(pa,pb)*distance(pb,pc)

# Clipping
# Prédicat qui vérifie que le rectangle est bien dans le polygone
# Teste si 
#     - il y a bien une intersection (!=[]) entre les figures et
#    - les deux listes ont la même taille et
#     - tous les points du rectangle appartiennent au résultat du clipping 
# Si erreur (~angle plat), retourne faux
def verifcontrainte(rect, polygone):
    try:
        # Config
        pc = pyclipper.Pyclipper()
        pc.AddPath(polygone, pyclipper.PT_SUBJECT, True)
        pc.AddPath(rect, pyclipper.PT_CLIP, True)
        # Clipping
        clip = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        #all(iterable) return True if all elements of the iterable are true (or if the iterable is empty)        
        return (clip!=[]) and (len(clip[0])==len(rect)) and all(list(map(lambda e:list(e) in clip[0], rect)))
    except pyclipper.ClipperException:
        # print rect
        return False

# Crée un individu (centre/coin/angle) FAISABLE
# un individu est décrit par votre metaheuristique contenant au moins: 
#     - pos : solution (centre/coin/angle) liste des variables
#    - eval :  aire du rectangle
#    - ... : autres composantes de l'individu
def initUn(polygone):
    global xmin,xmax,ymin,ymax
    boolOK = False
    while not boolOK: # tant que non faisable
        xo=random.uniform(xmin,xmax)
        yo=random.uniform(ymin,ymax)

        xa=xo+pow(-1,random.randint(0,1))*random.uniform(10,min(xo-xmin,xmax-xo))
        ya=yo+pow(-1,random.randint(0,1))*random.uniform(10,min(yo-ymin,ymax-yo))

        angle = random.uniform(anglemin,anglemax)

        pos = [round(xa),round(ya),round(xo),round(yo),angle]
        rect = pos2rect(pos)
        # calcul du clipping
        boolOK = verifcontrainte(rect,polygone)
    ev = aire(pos2rect(pos))
    return {'pos':pos, 'eval':ev}

# Init de la population
def initPop(nb,polygone):
    return [initUn(polygone) for i in range(nb)]

# Retourne la meilleure particule entre deux : dépend de la métaheuristique
def bestPartic(p1,p2):
    if (p1["eval"] > p2["eval"]):
        return p1 
    else:
        return p2
    
# Retourne une copie de la meilleure particule de la population
def getBest(population):
    return dict(reduce(lambda acc, e: bestPartic(acc,e),population[1:],population[0]))

def dispRes(best_route, best_dist, num_improvements):
    print("Pos = {}".format(best_route))
    print("Area = {}".format(best_dist))
    print("Number of improvements = {}".format(num_improvements))

def randomNeighbor(rectSol, polygone, pointVar):
    xa = rectSol['pos'][0]
    ya = rectSol['pos'][1]
    xo = rectSol['pos'][2]
    yo = rectSol['pos'][3]
    angle = rectSol['pos'][4]
    xSN = sizeNeigh*(xmax-xmin)
    ySN = sizeNeigh*(ymax-ymin)
    angleSN = sizeNeigh*(anglemax-anglemin)

    boolOK = False
    iii=0
    while not boolOK: # tant que non faisable
        iii+=1
        if pointVar in ['O','A']:
            # Try to Discover A in the Border.
            p=point(xMinTest, yMaxTest)
            y=0
            if yMaxTest == ymin or xMinTest == xmax:
                while ~(poly.contains(p)):
                    if y==0:
                        yMaxTest-=1
                        y=1
                    else:
                        xMinTest+=1
                        y=0
                    p=point(xMinTest, yMaxTest)
                xa=xMinTest
                ya=yMaxTest

            else: 
                xa=rectSol['pos'][0] + random.uniform(-xSN,xSN)
                ya=rectSol['pos'][1] + random.uniform(-ySN,ySN)
                
            xo=xa+pow(-1,random.randint(0,1))*random.uniform(10,max(xa-xmin,xmax-xa))
            yo=ya+pow(-1,random.randint(0,1))*random.uniform(10,max(ya-ymin,ymax-ya))

        if pointVar=='Angle':
            angle = rectSol['pos'][4] + random.uniform(-angleSN,angleSN)

        pos = [round(xa),round(ya),round(xo),round(yo),angle]
        # print(pos)
        pos = verifyBoundaries(pos)
        # print(pos)

        if iii>100:
            pos=rectSol['pos']

        rect = pos2rect(pos)
        # calcul du clipping
        boolOK = verifcontrainte(rect,polygone)
    ev = aire(pos2rect(pos))
    return {'pos':pos, 'eval':ev}

def verifyBoundaries(pos):
    newpos=pos
    #For position A  (Put in Boundary)
    for i in [0,2]:
        if pos[i] < xmin:
            newpos[i]=xmin
        elif pos[i] > xmax:
            newpos[i]=xmax


    #For position 0 (Return Old position)
    for i in [1,3]:
        if pos[i] < ymin:
            newpos[i]=ymin
        elif pos[i] > ymax:
            newpos[i]=ymax
    
    #For Alpha
    if pos[4] > anglemax :
        newpos[4]=anglemax
    elif pos[4] < anglemin:
        newpos[4]=anglemin

    return newpos


def metropolis(ch1,ch2,T, pointVar):
    global  num_improvements, best
    delta = ch1['eval'] - ch2['eval'] # calculating the differential
    if delta >= 0: # if improving,
        if ch1['eval'] > best['eval']: # comparison to the best, if better, save and refresh the figure
            updateBest(ch1,pointVar)
            # dessine(polygonefig, poly2list(pos2rect(best["pos"])))
            num_improvements = num_improvements + 1
        return ch1 # the fluctuation is retained, returns the neighbor
    else:
        delta1=10*delta/maxArea #to control the size of delta and avoid overflow
        if random.uniform() > exp(-delta1/T): # the fluctuation is not retained according to the proba
            return ch2              # initial path
        else:
            return ch1

def updateBest(ch1,pointVar):
    global best
    best= ch1
    # best['eval']=ch1['eval']
    # if pointVar=='A':
    #     best['pos'][0]=ch1['pos'][0]
    #     best['pos'][1]=ch1['pos'][1]
    # elif pointVar=='O':
    #     best['pos'][2]=ch1['pos'][2]
    #     best['pos'][3]=ch1['pos'][3]
    # elif pointVar=='Angle':
    #     best['pos'][4]=ch1['pos'][4]
    # else:
    #     best = ch1         

# *************************************** ALGO D'OPTIM ***********************************
# calcul des bornes pour l'initialisation
global xmin,xmax,ymin,ymax, maxArea, best, yMaxTest, xMinTest
xmin,xmax,ymin,ymax = getBornes(polygone)
yMaxTest = ymax
xMinTest = xmin

maxArea=(xmax-xmin)*(ymax-ymin)

lstVar=['A','Angle','O']
# lstVar=['O','Angle']

# initialisation de element
rectSol = initUn(polygone)
area = rectSol['eval']

# boucle principale (à affiner selon la métaheuristique / le critère de convergence choisi)
# initializing history lists for the final graph
Henergy = []     # energy
Htime = []       # time
HT = []           # temperature
Hbest = []        # distance

#initialization of the best route
best=rectSol
num_improvements = 0

# we trace the path of departure
dessine(polygonefig, poly2list(pos2rect(best['pos'])))

# main loop of the annealing algorithm
t = 0
T = T0
iterStep = Step

# ############################################ PRINCIPAL LOOP OF THE ALGORITHM ###### ######################

# Convergence loop on criteria of number of iteration (to test the parameters)
for i in range(IterMax):

    # Modify point A, O or Angle each time
    for pointVar in lstVar:

        # cooling law enforcement
        while (iterStep > 0): 
            # Create random Point A, O or Angle:            
            neighbor = randomNeighbor(rectSol, polygone, pointVar)
            # neighbor = initUn(polygone)

            # application of the Metropolis criterion to determine the persisted fulctuation
            rectSol = metropolis(neighbor,rectSol,T, pointVar)
            dessine(polygonefig, poly2list(pos2rect(best["pos"])))
            iterStep -= 1

        # cooling law enforcement
        t += 1
        # rules of temperature decreases
        #T = T0*exp(-t/tau)
        #T = T*Alpha
        iterStep = Step

    #historization of data
    if t % 5 == 0:
        Henergy.append(rectSol['eval'])
        Htime.append(t)
        HT.append(T)
        Hbest.append(best['eval'])
    
    #Plot actual solution
    # if i % 5 == 0:
    #     dessine(polygonefig, poly2list(pos2rect(rectSol["pos"])))        
    
############################################## END OF ALGORITHM - DISPLAY RESULTS ### #########################
parametersName="T0-" + str(T) + "Num_Improvements-" + str(num_improvements)
# display result in console
dispRes(rectSol['pos'], rectSol['eval'], num_improvements)
# graphic of stats
drawStats(Htime, Henergy, Hbest, HT, parametersName)


    

# FIN : affichages
dessine(polygonefig, poly2list(pos2rect(best['pos'])))
plt.show()

