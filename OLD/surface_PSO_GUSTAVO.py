#!/usr/bin/python
# -*- coding: Latin-1 -*-
# TP optim : maximisation de surface
# par l'algorithme PSO
# Peio Loubiere pour l'EISTI
# septembre 2017
# usage : python surface.corr1.py 
# Adapted: Gustavo FLEURY SOARES && INDU R.
from scipy import *
from math import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
import pyclipper
from functools import *

# Figure de visualisation de la parcelle
fig = plt.figure()
canv = fig.add_subplot(1,1,1)
canv.set_xlim(0,500)
canv.set_ylim(0,500)

# ************ Paramètres de la métaheuristique ***PSO=10000 DE=1500*********NB indiv 20*
Nb_Cycles = 1500
Nb_Indiv = 20

# Nb_Cycles = 150
# Nb_Indiv = 6

psi,cmax = (0.8, 1.62)
# psi,cmax = (0.5, 1.22)
# psi,cmax = (0.2, 1.0)
# ***********************************************************

# ***************** Paramètres du problème ******************
# Différentes propositions de parcelles : 
polygone = ((10,10),(10,400),(400,400),(400,10)) 
# polygone = ((10,10),(10,300),(250,300),(350,130),(200,10)) 
# polygone = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
# polygone = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))

# ***********************************************************
minX=min([x[0] for x in polygone])
maxX=max([x[0] for x in polygone])
minY=min([x[1] for x in polygone])
maxY=max([x[1] for x in polygone])
anglemin = 80
anglemax = 100


# Draw the figure of the graphs of:
def drawStats(Htemps, Hbest):
    # afFILEhage des courbes d'evolution
    fig2 = plt.figure(2)
    plt.subplot(1,1,1)
    # semilogy(Htemps, Hbest)
    plt.plot(Htemps, Hbest)
    plt.title('Evolution of the best Area')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.show()

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
    # return distance(p1[0],p1[1])* distance(p2[0],p2[1])
# def aire(pa, pb, pc, pd):
#     return distance(pa,pb)*distance(pb,pc)

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
    return {'pos':pos, 'eval':ev, 'bestEval':ev, 
            'vit':[0,0,0,0,0], 'bestpos':pos, 'bestvois':[0,0,0,0,0] }

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

# Retourne une copie de la meilleure particule de la population
def getNbBetters(population, nb):
    return [getBest(population) for i in range(nb)]

def update(particle,bestParticle):
    nv = dict(particle)
    if(particle["eval"] > particle["bestEval"]):
        nv['bestpos'] = particle["pos"][:]
        nv['bestEval'] = particle["eval"]

    if (particle["eval"] > bestParticle["bestEval"]):
        nv['bestvois'] = bestParticle["pos"][:]
    else:
        nv['bestvois'] = bestParticle["bestpos"][:]
    return nv

#FUNCTIONS OF PSO (Particle Swarm Opt.)
# Calculate the velocity and move a paticule
def move(particle, polygone, ii):
    # global ksi,c1,c2,psi,cmax

    nv = dict(particle)
    # print(nv)
    
    #1st Element-> psi*Vel
    el1 = funcTimes(psi, particle["vit"])    
    # print(particle["vit"])

    #2nd Element-> Xpb - Xp
    el2 = funcTimes(cmax*random.uniform(), funcMinus(particle["bestpos"], particle["pos"]) ) 
    # print(particle["bestpos"])
    # print(particle["pos"])

    #3rd Element-> Xb - Xp
    el3 = funcTimes(cmax*random.uniform(), funcMinus(particle["bestvois"], particle["pos"]) )     
    # print(particle["bestvois"])
    # print(particle["pos"])

    el1et2 = funcAdd(el1,el2)
    velocity = funcAdd(el1et2,el3)
    # print(velocity)

    #Update Position:
    position = particle['pos']
    newposition = position
    positionAux = funcAdd(position, velocity)
    #Update A,O,angle each time
    # if ii%30<10 :
    #     newposition[0] = positionAux[0]
    #     newposition[1] = positionAux[1]
    # elif ii%30>=10 and ii%30<20 :
    #     newposition[2] = positionAux[2]
    #     newposition[3] = positionAux[3]
    # else:
    #     newposition[4] = positionAux[4]

    # newposition = funcAdd(position, velocity)
    
    #Verify Boudaries
    newposition2 = verifyBoundaries(newposition, position)
    # print(newposition)
 
    rect=pos2rect(newposition2)
    # print(rect)
    nv['vit'] = velocity
    if verifcontrainte(rect,polygone):
        #Update Position        
        # nv['vit'] = velocity
        nv['pos'] = newposition2
        nv['eval'] = aire(pos2rect(newposition2))

    # print(nv)
    return nv

def verifyBoundaries(pos, posOld):
    newpos=[]
    #For position A  (Put in Boundary)
    for i in [0,2]:
        if pos[i] < minX:
            newpos.append(minX)
        elif pos[i] > maxX:
            newpos.append(maxX)
        else:
            newpos.append(pos[i])

    #For position 0 (Return Old position)
    for i in [1,3]:
        if pos[i] < minX:
            newpos.append(posOld[i])
        elif pos[i] > maxX:
            newpos.append(posOld[i])
        # elif pos[i] > ((maxX - pos[i-2])/2):
        #     newpos.append(((maxX - pos[i-2])/2))
        else:
            newpos.append(pos[i])
    
    #For Alpha
    if pos[4] > anglemax :
        newpos.append(anglemax)
    elif pos[4] < anglemin:
        newpos.append(anglemin)
    else:
        newpos.append(pos[4])
    # print(pos)
    # print(newpos)
    return newpos

def funcAdd(vel1, vel2):
    vel=[]
    for i in range(len(vel1)):
        vel.append(round(vel1[i]-vel2[i]))
    return vel

# k% of the list
def funcTimes(k, vel):
    return [round(i*k) for i in vel]

#particle - particle = velocity 
def funcMinus(vel1, vel2):
    vel=[]
    for i in range(len(vel1)):
        vel.append(round(vel1[i]-vel2[i]))
    return vel

def printPartcileArea(sw):
    print( '*' + str(round(sw[0]['bestEval'])) + '*-', end='' )
    for p in sw:
        print( str(round(p['eval'])) + '(' + 
               str(round(p['vit'][0]))  + ',' +
               str(round(p['vit'][1]))  + ',' +
               str(round(p['vit'][2]))  + ',' + 
               str(round(p['vit'][3]))  + ',' +
               str(round(p['vit'][4]))  + 
                 ')' + ' - ', end='' )
    print('')

def dispRes(best_route, best_dist):
    print("Pos = {}".format(best_route))
    print("Area = {}".format(best_dist))

# *************************************** ALGO D'OPTIM ***********************************
# calcul des bornes pour l'initialisation
xmin,xmax,ymin,ymax = getBornes(polygone)

# initialisation de la population (de l'agent si recuit simulé) et du meilleur individu.
global best, best_cycle
pop = initPop(Nb_Indiv,polygone)
best = getBest(pop)
best_cycle = best

printPartcileArea(pop)

# boucle principale (à affiner selon la métaheuristique / le critère de convergence choisi)
Htemps = []       # temps
Hbest = []        # distance

ii=0
noImprov=0
for i in range(Nb_Cycles):
    ii += 1
    #Update informations
    pop = [update(e,best_cycle) for e in pop]
    # velocity calculations and displacement
    pop = [move(e, polygone,ii) for e in pop]
    # Update of the best solution
    best_cycle = getBest(pop)
    if (best_cycle["bestEval"] > best["bestEval"]):
        # print(str(best_cycle["bestEval"]) + '----' + str(best["bestEval"]))
        # print(str(best_cycle["pos"]) + '----' + str(best["pos"]))
        best = best_cycle
        best["bestpos"]=best["pos"]
    else:
        noImprov += 1
        
    #Change if no Improvement
    if noImprov >= 20:
        # Randomize the half worst Eval 
        # nbNewPart=int(Nb_Indiv/2)
        nbNewPart=len(pop)-1
        PopBetters = getNbBetters(pop, len(pop)-nbNewPart )
        PopRandom = initPop(nbNewPart, polygone)
        pop = PopBetters + PopRandom
        pop = [update(e,best) for e in pop]

    # historization of data
    if i % 5 == 0:
        Htemps.append(i)
        Hbest.append(best['bestEval'])

    # swarm display
    if i % 5 == 0:
        dessine(polygonefig, poly2list(pos2rect(best["bestpos"])))        
        printPartcileArea(pop)
        # if i==0: 
        #     wait = input("PRESS ENTER TO CONTINUE.")

# END, displaying results
Htemps.append(ii)
Hbest.append(best['bestEval'])
# draw(best['route'], best['bestfit'], x, y, ii)
dessine(polygonefig, poly2list(pos2rect(best["bestpos"])))

        
#displaying result on the console
dispRes(best['bestpos'], best['bestEval'])
drawStats(Htemps, Hbest)

# FIN : affichages
# dessine(polygonefig, poly2list(pos2rect(best["pos"])))
plt.show()

