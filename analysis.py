import zen
import matplotlib.pyplot as plt
plt.ioff()
import numpy
from math import e
from numpy import *
from numpy.linalg import eig,norm
import sys
sys.path.append('../zend3js/')
import d3js
from time import sleep
import colorsys
import numpy.linalg as la

G = zen.io.gml.read('tennis.gml',weight_fxn = lambda x: x['weight'])
R = G.matrix()
R = R.transpose()
def print_top(G, v, num=10):
    idx_list = [(i,v[i]) for i in range(len(v))]
    idx_list = sorted(idx_list, key = lambda x: x[1], reverse=True)
    for i in range(min(num,len(idx_list))):
        nidx, score = idx_list[i]
        print ("%i. %s (%1.4f)" % (i+1,G.node_object(nidx),score))

def index_of_max(v):
    return numpy.where(v == max(v))[0]

def cocitation(G):
    G1=zen.Graph()
    nn=G.nodes()
    w=0
    P=0
    e=0
    j=0
    y=0
    for i in range(0,len(nn)):
        G1.add_node(nn[i])
    for j in range(0,len(nn)):
        u=G.neighbors(nn[j])
        r=len(u)
        if r==0:
            r=r+1
        for y in range(0,len(nn)):
            w=[]
            v=G.neighbors(nn[y])
            p=len(v)
            if p==0:
                p=p+1
            for s in range(0,r):
                if len(u)==0:
                    continue
                w=G.weight(u[s],nn[j])
                for e in range(0,p):
                    if len(v)==0:
                        continue
                    P=G.weight(v[e],nn[y])
                    if len(u)==0:
                        continue
                    if len(v)==0:
                        continue
                    if (v[e]==u[s]):
                        w.append(w*P)
                        count=numpy.sum(w)
                        if G1.has_edge(nn[j],nn[y])==True:
                            G1.set_weight(nn[j],nn[y],count)
                            continue
                        if nn[j]==nn[y]:
                            continue
                        G1.add_edge(nn[j],nn[y],weight=count)
    #X=G1.matrix()
    #numpy.fill_diagonal(X,0)
    #print X.sum().sum()
    return G1

def calc_powerlaw(G, kmin):
    N = G.num_nodes
    ddist = zen.degree.ddist(G,normalize=False)
    cdist = zen.degree.cddist(G,inverse=True)
    k = numpy.arange(len(ddist))

    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.bar(k,ddist, width=0.8, bottom=0, color='b')
    plt.xlabel("Degree")
    plt.ylabel("Degree Distribution")

    plt.subplot(212)
    plt.loglog(k,cdist)
    plt.xlabel("Degree")
    plt.ylabel("Cumulative Degree Distribution")

    sub = 0
    for z in range(0,len(ddist) - 1):
        if z < kmin:
            sub = sub + ddist[z]
    N = G.num_nodes - sub

    sum = 0
#     print ddist
#     print len(ddist)
    for k_i in range(kmin, len(ddist) - 1):
        fraction = k_i / (kmin - 0.5)
        iLn = ddist[k_i] * math.log(fraction,e)
        sum = sum + iLn
        if (sum != 0):
            sum_inv = 1/sum
    alpha = 1 + N * sum_inv # calculate using formula!
    print ('alpha is %1.2f' %alpha)
    plt.show()
def index_of_max(v):
	return where(v == max(v))[0]
def check_friendship_paradox(G):
    G_nodes = G.num_nodes
    G_edges = G.num_edges
    k_sum = 0
    k_sqr_sum = 0
    for i in range(0, G_nodes):
        i_deg = G.degree_(i)
        k_sum = k_sum + i_deg
        k_sqr_sum = k_sqr_sum + i_deg * i_deg
    k = k_sum / G_nodes
    k_sqr = k_sqr_sum / G_nodes
    print '\n <k^2> = %i' %k_sqr
    print '\n <k> = %i' %k
    neighbor_average = float(k_sqr)/float(k)
    node_average = float(k)
    if(float(k_sqr)/float(k) > float(k)):
        print 'Neighbor of a node has more neighbors (%1.3f) than the node itself (%1.3f) - Friendship Paradox is observed' %(neighbor_average, node_average)
    else:
        print 'Friendship Paradox is not observed'
        
    plt.figure(figsize=(8,15))
    plt.plot(k,node_average) # change x and y to your "x" and "y" values
    plt.plot([0.0001,1],[(log10(100)),(log10(100))]) # plot the limit
    plt.xlabel('probability of paradox')
    plt.ylabel('neighbor_average')
    plt.show()
        
    
def kats(alpha) :
#	alpha = 0.99
	ones = numpy.ones((G.num_nodes,1))
	iden =numpy.eye(G.num_nodes)
	temp1 = la.inv(iden - (alpha*R))
	ex = numpy.dot(temp1,ones)
	print_top(G,ex)

def main():
    G = zen.io.gml.read('tennis.gml',weight_fxn = lambda x: x['weight'])
    A = G.matrix()
    N = G.num_nodes

    print ('\n=============================================')
    print ('\nDegree Centrality:')
    ei = [0] * N
    for i in range(N):
            st = G.neighbors_(i)
            sum = 0
            for j in range(len(st)):
                    sum += G.weight(G.node_object(i),G.node_object(st[j]))
            ei[i] = sum
    print_top(G,ei)

    print ('\n=============================================')
    # Eigenvector Centrality
    print ('\nEigenvector Centrality (by Zen):')
    ec = zen.algorithms.centrality.eigenvector_centrality_(G,weighted = True)
    print_top(G,ec)

    print ('\n=============================================')
    # Betweenness Centrality
    print ('\nBetweenness Centrality')
    bc= zen.algorithms.centrality.betweenness_centrality_(G)
    print_top(G,bc)


    print ('\n==============================================')
    print ('\nPOWER LAW')
    calc_powerlaw(G,3)    # need to change kmin appropriately


    print ('\n==============================================')
    print ('\nClustering Coefficient')
    c = zen.algorithms.clustering.gcc(G)
    print ('Clustering: %s' % c)
    check_friendship_paradox(G)
    print '\n============================================='
    # Katz Centrality
    print '\nKatz Centrality:'

    print 'katz for alpha = 1.37'
    kats(1.37)








if __name__ == '__main__':
    main()
