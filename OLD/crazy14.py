from scipy import *
from math import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
import pyclipper
from functools import *
fig = plt.figure()
canv = fig.add_subplot(1,1,1)
canv.set_xlim(0,500)
canv.set_ylim(0,500)
Nb_Cycles = 1500
Nb_Indiv = 20                    # for 5 works good for 3rd type, for 20 works good for 4th
#psi,cmax = (0.7, 0.2)
psi,cmax = (0.7, 1.47)
Htemps = []
Hbest = []
( polygone, maxPolArea ) = ( ((10,10),(10,400),(400,400),(400,10)) , 152100)
# polygone = ((10,10),(10,300),(250,300),(350,130),(200,10))
# polygone = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
# polygone = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))


anglemin = 1
anglemax = 89

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

def drawStats(Htemps, Hbest):
    fig2 = plt.figure(2)
    plt.subplot(1, 1, 1)
    plt.plot(Htemps, Hbest)
    plt.title('Evolution of the best Area')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.show()

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
        xo=random.uniform(xmin,xmax)                                                                                    # centre point x coord
        yo=random.uniform(ymin,ymax)                                                                                    # centre point y coord
        xa=xo+pow(-1,random.randint(0,1))*random.uniform(10,min(xo-xmin,xmax-xo))                                       # corner A - x coord
        ya=yo+pow(-1,random.randint(0,1))*random.uniform(10,min(yo-ymin,ymax-yo))                                       # corner A - y coord
        angle = random.uniform(anglemin,anglemax)                                                                       # angle
        pos=[round(xa),round(ya),round(xo),round(yo),angle]                                                             # all above are put together
        rect=post2rect(pos)                                                                                             # using all above, we generate rectangle
        boolOK = verifcontrainte(rect, polygone)                                                                        # to verify if all these 4 points are within polygone
    ev = aire(post2rect(pos))                                                                                           # finding area of the rect formed
    return {'vit': [0,0,0,0,0], 'pos': pos, 'fit': ev, 'bestfit': ev, 'bestpos': pos, 'bestvois': [0,0,0,0,0]}

def post2rect(pos):
    xa, ya = pos[0], pos[1]                                                                                             # point A
    xo, yo = pos[2], pos[3]                                                                                             # point o
    angle = pos[-1]                                                                                                     # angle in degree
    alpha = pi * angle / 180                                                                                            # angle in radian
    xd = cos(alpha)*(xa-xo) - sin(alpha)*(ya-yo) + xo                                                                   # generating point d
    yd = sin(alpha)*(xa-xo) + cos(alpha)*(ya-yo) + yo
    xc, yc = 2*xo - xa, 2*yo - ya                                                                                       # generating point c
    xb, yb = 2*xo - xd, 2*yo - yd                                                                                       # generating point b
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

def move(particle):
    global ksi,c1,c2,psi,cmax;
    nv= dict(particle)
    tu_velocity1= particle['vit']
    tu_velocity1= times(psi,tu_velocity1)
    tu_loc_velocity= minus(particle['bestpos'],particle['pos'])
    tu_loc_velocity= times(cmax*random.uniform(0,1),tu_loc_velocity)
    tu_gro_velocity= minus(particle['bestvois'],particle['pos'])
    tu_gro_velocity= times(cmax*random.uniform(0,1),tu_gro_velocity)
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
            return (ch2, dist2)
        else:
            return (ch1, dist1)

# MAIN PART

xmin,xmax,ymin,ymax = getBornes(polygone)
pos = initPop(Nb_Indiv,polygone)
best = getBest(pos)
best_cycle = best
polygonefig = poly2list(polygone)
k=0
z=0
no_iteration=0
ltaboo=10
best_of_best=[0]
best_of_best[0]=best.copy()
for z in range(Nb_Cycles):
    pos=[update(e,best_cycle) for e in pos]
    pos=[move(e) for e in pos]
    best_cycle=getBest(pos)
    print("{} + ' '+{}".format(best_cycle['bestfit'],best_of_best[0]['bestfit']))
    print(best_cycle['bestfit'] > best_of_best[0]['bestfit'])
    if best_cycle['bestfit']>best_of_best[0]['bestfit']:
        best_of_best[0]=best_cycle.copy()
        print(best_of_best[0]['bestfit'])
    if(best_cycle['bestfit']>best['bestfit']):
        best=best_cycle
    else:
        no_iteration=0
        rand_pos = initUn(polygone)
        (best_cycle['bestpos'], best_cycle['fit']) = metropolis(rand_pos['pos'], rand_pos['fit'], best['bestpos'], best['bestfit'],0.1) #10000000
    if z % 10 == 0:
        # dessine(polygonefig, poly2list(post2rect(best_cycle["bestpos"])), z)
        dessine(polygonefig, poly2list(post2rect(best_of_best[0]["bestpos"])), z)
        Htemps.append(z)
        Hbest.append(best_of_best[0]['bestfit'])

Htemps.append(z)
Hbest.append(best_of_best[0]['bestfit'])
dessine(polygonefig, poly2list(post2rect(best_of_best[0]["bestpos"])), z,1)
dispRes(best_of_best[0]['bestpos'], best_of_best[0]['bestfit'])
drawStats(Htemps, Hbest)
plt.show()




