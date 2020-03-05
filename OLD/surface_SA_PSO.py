from scipy import *
from math import *
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
import pyclipper
from functools import *
import time

fig = plt.figure()
canv = fig.add_subplot(1,1,1)
canv.set_xlim(0,500)
canv.set_ylim(0,500)

## PARAMETERS:
IterMax_LST = [1500]        # 15000 max number of iterations of the algorithm
iterStep_LST = [10]         # number of iterations on a temperature level
T_LST = [10]                # initial temperature
alpha_LST = [1] # [0.9]          # constant for geometric decay  
sizeNeigh = 1               # Size of the Xmax-Xmin to randomize

numTests = 1            # Number of test to Statistics Analysis.

( polygone, maxPolArea ) = ( ((10,10),(10,400),(400,400),(400,10)) , 152100)
# ( polygone, maxPolArea ) = ( ((10,10),(10,300),(250,300),(350,130),(200,10)) , 81100) 
# ( polygone, maxPolArea ) =( ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200)) , 46000) 
# ( polygone, maxPolArea ) =( ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50)) , 167000)

anglemin = 0.01
anglemax = 89.99

def poly2list(polygone):
    polygonefig = list(polygone)
    polygonefig.append(polygonefig[0])
    return polygonefig

def dessine(polyfig, rectfig , iteration, pause=0.01):
    global canv, codes
    canv.clear()
    codes = [Path.MOVETO]
    for i in range(len(polyfig) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(polyfig, codes)
    patch = patches.PathPatch(path, facecolor='orange', lw=2)
    canv.add_patch(patch)
    canv.autoscale_view()
    codes = [Path.MOVETO]
    for i in range(len(rectfig) - 2):
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(rectfig, codes)
    patch = patches.PathPatch(path, facecolor='grey', lw=2)
    canv.add_patch(patch)
    plt.title("Aire: {} Aire/Max: {}% Iteration: {}".format(round(aire(rectfig[:-1]), 2), round((aire(rectfig[:-1])/maxPolArea)*100,1), iteration) )
    plt.draw()
    plt.pause(pause)

def dispRes(best_route, best_dist):
    print("Pos = {}".format(best_route))
    print("Area = {}".format(best_dist))

def drawStats(Htemps, Hbest, parametersSTR, titleName):
    fig2 = plt.figure(2,figsize=(9,6))
    plt.subplot(1,1,1)
    plt.plot(Htemps, Hbest, label=parametersSTR)   #plt.semilogy
    plt.title(titleName)
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.legend()

def drawBoxPlot(BoxPlotData, xValues, xLabel, titleName): 
    fig3 = plt.figure(3,figsize=(6,6))
    plt.subplot(1,1,1)
    sns.boxplot( x=xValues, y=BoxPlotData, palette="Blues")
    # plt.boxplot(BoxPlotData)
    # plt.xticks(range(1, len(BoxPlotData)+1), xLables)
    plt.title(titleName)
    plt.xlabel(xLabel)
    plt.ylabel('Percentage Area Best Rectangule')
    plt.legend()

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def getBornes(polygone):
	lpoly = list(polygone)
	return reduce(lambda acc,e: (min(e[0],acc[0]),max(e[0],acc[1]),min(e[1],acc[2]),max(e[1],acc[3])),lpoly[1:],(lpoly[0][0],lpoly[0][0],lpoly[0][1],lpoly[0][1]))

def initPop(nb,polygone):
	return [initUn(polygone) for i in range(nb)]

def initUn(polygone):
    global xmin,xmax,ymin,ymax;
    anglemin = 1
    anglemax = 89
    boolOK = False
    while boolOK==False:
        xo=random.uniform(xmin,xmax)                                           # centre point x coord
        yo=random.uniform(ymin,ymax)                                           # centre point y coord
        xa=xo+pow(-1,random.randint(0,1))*random.uniform(10,min(xo-xmin,xmax-xo))    # corner A - x coord
        ya=yo+pow(-1,random.randint(0,1))*random.uniform(10,min(yo-ymin,ymax-yo))    # corner A - y coord
        angle = random.uniform(anglemin,anglemax)                              # angle
        pos=[round(xa),round(ya),round(xo),round(yo),angle]                    # all above are put together
        rect=post2rect(pos)                                                    # using all above, we generate rectangle
        boolOK = verifcontrainte(rect, polygone)                               # to verify if all these 4 points are within polygone
    ev = aire(post2rect(pos))                                                  # finding area of the rect formed
    return {'vit': [0,0,0,0,0], 'pos': pos, 'fit': ev, 'bestfit': ev, 'bestpos': pos, 'bestvois': [0,0,0,0,0]}

def post2rect(pos):
    xa, ya = pos[0], pos[1]                                                    # point A
    xo, yo = pos[2], pos[3]                                                    # point o
    angle = pos[-1]                                                            # angle in degree
    alpha = pi * angle / 180                                                   # angle in radian
    xd = cos(alpha)*(xa-xo) - sin(alpha)*(ya-yo) + xo                          # generating point d
    yd = sin(alpha)*(xa-xo) + cos(alpha)*(ya-yo) + yo
    xc, yc = 2*xo - xa, 2*yo - ya                                              # generating point c
    xb, yb = 2*xo - xd, 2*yo - yd                                              # generating point b
    return ((round(xa),round(ya)),(round(xb),round(yb)),(round(xc),round(yc)),(round(xd),round(yd)))

def verifcontrainte(rect, polygone):
	try:
		pc = pyclipper.Pyclipper()
		pc.AddPath(polygone, pyclipper.PT_SUBJECT, True)
		pc.AddPath(rect, pyclipper.PT_CLIP, True)
		clip = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
		return (clip!=[]) and (len(clip[0])==len(rect)) and all(list(map(lambda e:list(e) in clip[0], rect)))
	except pyclipper.ClipperException:
		return False

def aire(p1):
    l= length(p1[0],p1[1])
    b= breath(p1[1],p1[2])
    return l*b

def length(a,b):
    return abs(a[1]-b[1])

def breath(b,c):
    return abs(b[0]-c[0])

def getBest(population):
	return dict(reduce(lambda acc, e: bestPartic(acc,e),population[1:],population[0]))

def bestPartic(p1,p2):
    if (p1["fit"] > p2["fit"]):
        return p1
    else:
        return p2

def update(particle, bestParticle):
    nv = dict(particle)
    if (particle["fit"] > particle["bestfit"]):
        nv['bestpos'] = particle["pos"][:]
        nv['bestfit'] = particle["fit"]
    nv['bestvois'] = bestParticle["bestpos"][:]
    return nv

def move(particle,psi,c1,c2):
    nv= dict(particle)
    tu_velocity1= particle['vit']
    tu_velocity1= times(psi,tu_velocity1)
    tu_loc_velocity= minus(particle['bestpos'],particle['pos'])
    tu_loc_velocity= times(c1*random.uniform(0,1),tu_loc_velocity)
    tu_gro_velocity= minus(particle['bestvois'],particle['pos'])
    tu_gro_velocity= times(c1*random.uniform(0,1),tu_gro_velocity)
    velocity1_comb = add_list(tu_velocity1,tu_loc_velocity)
    final_velocity = add_list(tu_gro_velocity, velocity1_comb)
    final_move = add_list(final_velocity,particle['pos'])
    pos = [round(final_move[0]), round(final_move[1]), round(final_move[2]), round(final_move[3]), final_move[4]]
    rect = post2rect(pos)
    boolOK = verifcontrainte(rect, polygone)
    if boolOK==True:
        nv['vit'] = final_velocity
        nv['pos'] = pos
        nv['fit'] = aire(post2rect(pos))
        return nv
    else:
        return nv

def times(parameter,tu_velocity):
    new_velocity=[parameter*element for element in tu_velocity]
    return new_velocity

def minus(vel1,vel2):
    vel = []
    for i in range(len(vel1)):
        vel.append(round(vel1[i] - vel2[i]))
    return vel

def add_list(v1,v2):
    new=[]
    for i in range(len(v1)):
        new.append(v1[i]+v2[i])
    return new

def metropolis(ch1,dist1,ch2,dist2,T):
    global best_route, best_dist, x, y
    delta = dist1 - dist2
    if delta <= 0:
        if dist1 <= best['fit']:
            best['bestfit'] = dist1
            best['fit'] = dist1
            best['bestpos'] = ch1[:]
            best['pos'] = ch1[:]
        return (ch1, dist1)
    else:
        if random.uniform(0,1) > exp(-delta/T):
            # return (ch2, dist2)
            return (ch1, dist1)
        else:
            return (ch1, dist1)

def randomNeighbor(rectSol, polygone, pointVar):
    (xa, ya, xo, yo, angle ) = rectSol['pos']
    xSN = sizeNeigh*(xmax-xmin)
    ySN = sizeNeigh*(ymax-ymin)
    angleSN = sizeNeigh*(anglemax-anglemin)

    boolOK = False
    iii=0
    while not boolOK: # tant que non faisable
        iii+=1
        if pointVar=='O':
            xo = xo + random.uniform(-xSN,xSN)
            yo = yo + random.uniform(-ySN,ySN)            
            xa=xo+pow(-1,random.randint(0,1))*random.uniform(10,min(xo-ymin,ymax-xo))
            ya=yo+pow(-1,random.randint(0,1))*random.uniform(10,min(yo-ymin,ymax-yo))

        if pointVar=='A':
            xa = xa + random.uniform(-xSN,xSN)
            ya = ya + random.uniform(-ySN,ySN)
            xo=xa+pow(-1,random.randint(0,1))*random.uniform(10,max(xa-xmin,xmax-xa))
            yo=ya+pow(-1,random.randint(0,1))*random.uniform(10,max(ya-ymin,ymax-ya))

        if pointVar=='Angle':
            angle = random.uniform(anglemin,anglemax)

        pos = [round(xa),round(ya),round(xo),round(yo),angle]

        if iii>100:
            pos=rectSol['pos']
        boolOK = verifcontrainte(post2rect(pos),polygone)
    ev = aire(post2rect(pos))
    return {'vit': [0,0,0,0,0], 'pos': pos, 'fit': ev, 'bestfit': ev, 'bestpos': pos, 'bestvois': [0,0,0,0,0]}

def main_SA(IterMax, iterStep, T, alpha):
    global best
    Htemps = []
    Hbest = [] 
    rectSol = initUn(polygone)
    polygonefig = poly2list(polygone)
    best=rectSol
    lstVar=['Angle','O','A']

    for t in range(IterMax):
        for pointVar in lstVar:
            for j in range(iterStep):
                neighbor = randomNeighbor(rectSol, polygone, pointVar)
                if neighbor['fit']>best['fit']:
                    best=neighbor
                #(rectSol['pos'], rectSol['fit']) = metropolis(neighbor['pos'], neighbor['fit'], best['pos'], best['fit'], T)
            T = T*alpha
        if t%10 == 0:
            dessine(polygonefig, poly2list(post2rect(best["pos"])), t)
            Htemps.append(t)
            Hbest.append(best['fit'])

    dessine(polygonefig, poly2list(post2rect(best["pos"])), t)
    Htemps.append(t)
    Hbest.append(best['fit'])
    dispRes(best["pos"], best['fit'])


# MAIN PART
xmin,xmax,ymin,ymax = getBornes(polygone)
HLastBest = []
for IterMax in IterMax_LST:
    for iterStep in iterStep_LST:
        for T in T_LST:
            for alpha in alpha_LST:
                main_SA(IterMax, iterStep, T, alpha)


# drawTimeVS(Nb_particleLST, timeSolutionMEAMLST, parameterName )
titleName = 'IterMax: ' + str(IterMax) + ' iterStep: ' + str(iterStep) + ' T: ' + str(T) + ' alpha: ' + str(alpha) 
# drawBoxPlot(BoxPlotData, Nb_Indiv_LST, 'NbParticles', titleName)
plt.show()




