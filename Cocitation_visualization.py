from zen.constants import *
from zen.exceptions import *
import zen
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from numpy import *
from numpy.linalg import eig,norm
import sys
sys.path.append('../zend3js/')
import d3js
from time import sleep
import colorsys
import numpy.linalg as la
from zen.algorithms.community import louvain


# handle the keyword arguments
#node_obj_fxn = kwargs.pop('node_obj_fxn',str)
#directed = kwargs.pop('directed',False)
#check_for_duplicates = kwargs.pop('ignore_duplicate_edges',False)
#if len(kwargs) > 0:
#  raise ZenException, 'Unknown keyword arguments: %s' % ', '.join(kwargs.keys())
#if check_for_duplicates and directed:
#  raise ZenException, 'ignore_duplicate_edges can only be set when directed = False'


G = zen.io.gml.read('tennis0.gml', weight_fxn = lambda x: x['weight']) 
A1 = G.matrix()
A1 = A1.transpose()
C1 = np.dot(A1.transpose(), A1)
print 'C1 matrix is:\n', C1

def propagate(G,d3,x,steps,slp=0.5,keep_highlights=True,update_at_end=True):
   interactive = d3.interactive
   d3.set_interactive(False)
   d3.highlight_nodes_(list(where(x>0)[0]))
   d3.update()
   sleep(slp)
   cum_highlighted = sign(x)
   for i in range(steps): # the brains
      x = sign(dot(C1,x)) # the brains
      cum_highlighted = sign(cum_highlighted+x)
      if not update_at_end:
        if not keep_highlights:
          d3.clear_highlights()
          d3.highlight_nodes_(list(where(x>0)[0]))
          d3.update()
          sleep(slp)
   if update_at_end:
     if not keep_highlights:
       d3.clear_highlights()
       d3.highlight_nodes_(list(where(x>0)[0]))
     else:
         d3.highlight_nodes_(list(where(cum_highlighted>0)[0]))
         d3.update()
   d3.set_interactive(interactive)
   if keep_highlights:
     return cum_highlighted
   else:
       return x
               

def cocitation(G): 
 G_cocitation = zen.DiGraph()
 print "Loaded a graph with " + str(G.num_nodes) + " nodes and " + str(G.num_edges) + " edges."
 n = G.num_nodes

 for x in range(0,n): # adding nodes
  G_cocitation.add_node(G.node_object(x))
 
 for i in range(0,n):
  for j in range(i+1, n):
   neigh_i = G.in_neighbors_(i)
   neigh_j = G.in_neighbors_(j)
   y = set(neigh_i).intersection(set(neigh_j)) # The intersections between neighbors of i and j
   if len(y) > 0:
    cumsum = 0
    for k in y:
     cumsum += G.weight_(G.edge_idx_(k,i)) * G.weight_(G.edge_idx_(k,j))
    G_cocitation.add_edge_(i,j, weight=cumsum)
 return G_cocitation   
  



#C2 = cocitation(G).matrix()
#print 'C2 matrix is:\n', C2


       
# Set up visualizer
#G = zen.Graph()
#d3 = d3js.D3jsRenderer(G,event_delay=0.03, canvas_size=(4000,2000), interactive=True, autolaunch=False)

# Set up visualizer
G1 = zen.io.gml.read('tennis0.gml', weight_fxn = lambda x: x['weight'])
C3 = cocitation(G1)
d3.clear()
d3.set_graph(C3)
d3.update()


# code to propagate
x=zeros(C3.num_nodes)
x[0]=1
propagate(C3,d3,x,10,slp=1)
#sleep(3)

#cset = louvain(C3)
#print 'The Number of Communities are: %d'%(cset.communities().__len__())
#comm_sizes = []
#for comm in cset.communities().__iter__():
#   comm_sizes.append(comm.__len__())
#plt.hist(comm_sizes,20)
#plt.xlabel('community sizes')
#plt.show()
d3.stop_server()

