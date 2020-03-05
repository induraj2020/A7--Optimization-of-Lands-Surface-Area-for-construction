# import matplotlib.pyplot as plt
   
# fig1 = plt.figure(1,figsize=(6,6))
# plt.subplot(1,1,1)
# HLastBest=[[83.8, 78.7, 78.0], [88.3, 87.6, 83.0], [84.7, 82.3, 88.3], [86.0, 85.2, 83.6], [85.7, 89.0, 83.2]]
# plt.boxplot(HLastBest)
# xlabels=[0.2, 0.8, 1.0, 1.5, 2.0]
# plt.xticks(range(1, len(HLastBest)+1), xlabels)
# plt.title("titleName")
# plt.xlabel('C2')
# plt.ylabel('Area Best Rectangule')
# # plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# HLastBest=[[83.8, 78.7, 78.0], [88.3, 87.6, 83.0], [84.7, 82.3, 88.3], [86.0, 85.2, 83.6], [85.7, 89.0, 83.2]]
# xlabels=[0.2, 0.8, 1.0, 1.5, 2.0]
# sns.boxplot( x=xlabels, y=HLastBest, palette="Blues")
# plt.show()

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# corn=((10,10),(10,400),(400,400),(400,10))
# corn=((10,10),(10,300),(250,300),(350,130),(200,10))
# corn=((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
corn=((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))
corn=((221, 98), (315, 280), (397, 238), (303, 56))
print(PolygonArea(corn))