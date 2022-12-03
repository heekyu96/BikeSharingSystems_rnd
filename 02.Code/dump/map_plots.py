# -*- coding: utf-8 -*-

import csv
from datetime import datetime
import math
import networkx as net
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import entropy
import math

#function for degree entropy
def degree_entropy(G, normed=False):
    P = np.bincount(list(dict(net.degree(G)).values()))
    NP = np.nonzero(P)
    N = NP[0].size
    H = entropy(P[NP])
    if normed and N > 1:
        return H / math.log(N)
    else:
        return H


def graphHistogram(g1, g2):

    # Histogram of density function between time of day being morning and evening

    morning = dict(net.degree(g1)).values()
    evening = dict(net.degree(g2)).values()

    plt.subplot(211)
    plt.hist(morning,density=True)
    plt.title('Morning Degree Histogram')
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.subplot(212)
    plt.hist(evening,density=True)
    plt.title('Evening Degree Histogram')
    plt.xlabel('Degree')
    plt.ylabel('Probability')


    plt.show()
    plt.close()



def graphRank(g,graphTitle):

    #rank of vertex degrees
    D = sorted(dict(net.degree(g)).values(),reverse=True)
    plt.suptitle(graphTitle)
    plt.subplot(221)
    plt.plot(D,'b-',marker='o')
    plt.xlabel('rank')
    plt.ylabel('degree')
    plt.subplot(222)
    plt.xlabel('rank')
    plt.ylabel('degree')
    plt.semilogx(D,'b-',marker='o')
    plt.subplot(223)
    plt.xlabel('rank')
    plt.ylabel('degree')
    plt.semilogy(D, 'b-', marker='o')
    plt.subplot(224)
    plt.xlabel('rank')
    plt.ylabel('degree')
    plt.loglog(D, 'b-', marker='o')
    plt.show()
    plt.close()

def withInBoundry(compNum, lowerBound, upperBound):
    if(compNum > lowerBound and compNum < upperBound):
        return True
    else:
        return False
    
def getXY(lat,lon):
     #start position of the graph, 36, -86
     defaultLat = 39.706446
     defaultLon = -86.249760
     #increment of graph
     inc = .02
     
     x = (lat - defaultLat)/inc
     y = (lon - defaultLon)/inc
     
     if(x >=0):
         x = math.floor(x)
     else:
         x= math.ceil(x)
     if(y >=0):
         y = math.floor(y)
     else:
         y = math.ceil(y)
     return (x,y) 
  
def createGraphFromCsv(fileName,monthLowerBound = None, monthUpperBound = None,
                       hourLowerBound = None, hourUpperBound = None):

    G = net.DiGraph()
    
    with open('purr_scooter_data.csv', mode = 'r') as file:
        csvFile = csv.reader(file)
        #read header
        for line in csvFile:
            break
        
        #read data from csv
        for line in csvFile:

            #ignore lines not complete
            if "NA" in line:
                continue
            
            #read positions and times
            startTime = datetime.strptime(line[2], "%Y-%m-%dT%H:%M:%SZ")
            endTime = datetime.strptime(line[3], "%Y-%m-%dT%H:%M:%SZ")

            startLat = float(line[4])
            endLat = float(line[6])
            startLon = float(line[5])
            endLon = float(line[7])

            if not 40.1 >= startLat >=39.65:
                continue
            if not 40.1 >= endLat >=39.65:
                continue
            if not -86.00 >= startLon >= -86.27:
                continue
            if not -86.00 >= endLon >= -86.27:
                continue

            startPos = str(getXY(float(line[4]),float(line[5])))
            endPos = str(getXY(float(line[6]),float(line[7])))

            
            #check if line is within month 
            if((monthLowerBound is not None) and (monthUpperBound is not None)):  
                if(not withInBoundry(startTime.month, monthLowerBound, monthUpperBound)):
                    continue
        
            #check if start time is between time bound
            if((hourLowerBound is not None) and (hourUpperBound is not None)):    
                if(not withInBoundry(startTime.hour, hourLowerBound, hourUpperBound)):
                    continue
            
            
            if startPos in G.nodes():
                G.nodes()[startPos]['weight'] += 1
            else:
                G.add_node(startPos,weight = 1)
                
            if endPos in G.nodes():
                G.nodes()[endPos]['weight'] += 1
            else:
                G.add_node(endPos,weight = 1)
            
            #check if the positions equal eachother
            if startPos == endPos:
                continue
            
            if ((startPos,endPos) in G.edges()):
                G[startPos][endPos]['weight'] += 1
            else:
                G.add_edge(startPos, endPos, weight = 1)
                
            
            
            #csvData.append(data)
        return G
     
def main():
  g = createGraphFromCsv('purr_scooter_data.csv', monthLowerBound=None, monthUpperBound=None,
                    hourLowerBound=13, hourUpperBound=21)
  #create graph that only consider months may through sep
  # g = createGraphFromCsv('purr_scooter_data.csv', monthLowerBound= 5,
  #                        monthUpperBound = 8)
  
  #create graph that only considers times between 5am though 4pm
  # g = createGraphFromCsv('purr_scooter_data.csv', hourLowerBound= 9,
  #                        hourUpperBound = 16)
  
  # create graph that conciders all data
  # g = createGraphFromCsv('purr_scooter_data.csv')
  
  nodesToRemove = []
  #remove nodes with no edges
  for node in g.nodes():
      if(g.degree(node) == 0):
          nodesToRemove.append(node)
          
  g.remove_nodes_from(nodesToRemove)
  nodes = g.nodes()

  xyPos = []

  for node in nodes:
      xyPos.append(tuple(map(int, node.strip("(").strip(")").split(", "))))

  layout = dict(zip(g, xyPos))
  D = dict(net.degree(g))
  print(D)
  labels = {n: g.nodes[n]['weight'] for n in g.nodes}
  deg = net.degree_centrality(g)

  net.draw_networkx(g, pos=layout,node_size=200,font_size=10, width = .5, with_labels=False)
  plt.title('Evening Travel')
  plt.show()
  plt.close()
  net.write_gml(g,"graph_evening.gml")


main()

#read both morning and evening graph files
gMorning = net.read_gml("graph_morning.gml")
gEvening = net.read_gml("graph_evening.gml")

#check the assortativity of both graphs
morningAssort = net.degree_pearson_correlation_coefficient(gMorning)
eveningAssort = net.degree_pearson_correlation_coefficient(gEvening)
print("Morning assortativity: ", morningAssort)
print("Evening assortativity: ", eveningAssort)

#check entropy of both graphs
morningEntropy = degree_entropy(gMorning, normed=True)
eveningEntropy = degree_entropy(gEvening,normed=True)
print("Morning entropy: ", morningEntropy)
print("Evening entropy: ", eveningEntropy)

#plot probablity density histograms
graphHistogram(gMorning,gEvening)

#calling rank function to plot degree ranks between morning and evening graph data
graphRank(gMorning, "Morning Degree Rank")
graphRank(gEvening,"Evening Degree Rank")
