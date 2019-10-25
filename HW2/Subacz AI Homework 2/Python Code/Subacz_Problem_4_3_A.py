# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:15:34 2019

@author: psubacz
"""

""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.
"""
import sys

class Node:
    '''
    Simple node class for the agent to keep track of self in graph
    '''
    def __init__(self, parent = None, node = None, dist = 0,all_visited = False):
        self.parent = parent
        self.node = node
        self.h = 0
        self.dist = dist
        self.all_visited = all_visited
        
    def get_parent(self):
        return self.parent

    def get_node(self):
        return self.node
     
    def get_dist(self):
        return self.dist

    def get_all_visited(self):
        return self.all_visited
    
    def set_all_visited(self):
        self.all_visited = True
        
class Graph(object):
    '''
    Graph class source: https://www.python-course.eu/graphs_python.php
    '''
    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]

    def __generate_edges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {list(neighbour)[0], vertex} not in edges:
                    edges.append({vertex, list(neighbour)[0]})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

def hueristic():
    pass

def agent_hill_climbing(_map,origin,frontier,expanded = None,max_itar = 30):
    '''
    Implementation of a hill climbing agent to solve a TSP problem
    '''
    #Create Origin Node
    origin_node = Node(parent = None, node = origin)
    #Set Current node to Origin
    current_node = origin_node
    #Create a list of frontier nodes
    frontier_nodes = []
    
    curr_iter = 0
    
    #While not all cities have been visited
    while (True):
        all_visited = False
        at_origin = False
        all_cities = list(_map)

        print('Iteration: ', curr_iter,'\nCurrently in City: ',current_node.get_node(), 'Distance traveled: ',current_node.get_dist())
        #1. Check for gaol conditions - See if all cities have been visited 
        #  and if at origin
        cities_visited = [current_node.get_node()]
        solution =  current_node
        
        #This loop traces the nodes back to the orgin and checks off visited 
        #  cities if all cities have been checked then set the goal all_visited
        #  to true
        while (solution.get_parent() != None):
            #Grab the parent node
            solution = solution.get_parent()
            #Add to visited cities list
            cities_visited.append(solution.get_node())
            #Check off the visited city and remove from list
            if (all_cities.count(solution.get_node())>=1):
                all_cities.remove(solution.get_node())
            #if all cities have been checked then set the goal to true 
            if (len(all_cities) == 0):
                current_node.set_all_visited()
                all_visited = True
                print('ALL VISITED')

        #Check to see if the agent is at the origin node origin
        if (current_node.get_node() == origin):
            print('AT ORIGIN')
            at_origin = True

        #If both conditions have been met, then break and return the solution
        if at_origin and all_visited:
            print('Jobs Done!')
            cities_visited.reverse()
            print('Travel Solution: ',cities_visited)
            print('Travel Distance: ',current_node.get_dist())
            break

        #1. Look at neighbor cities        
        neighbors = list(_map[current_node.get_node()])
        
        #2 - Generate nodes at next level
        total_poss_dist = 0
        for neighbor in neighbors:
#            Generate new frontier nodes and encode historic paths 
            new_node = Node(parent = current_node, node = list(neighbor)[0], 
                            dist = current_node.get_dist()+neighbor[list(neighbor)[0]],
                            all_visited = current_node.get_all_visited())
            frontier_nodes.append(new_node)
            #Calculate the total possible distance of the nodes 
            total_poss_dist = total_poss_dist+current_node.get_dist()+neighbor[list(neighbor)[0]]
        sol = []
        #Select the lowest node 
        for temp_node in frontier_nodes:
            sol.append(temp_node.get_node())
            if (temp_node.get_dist() < total_poss_dist):
                total_poss_dist = temp_node.get_dist()
                current_node = temp_node

        frontier_nodes.remove(current_node)
        
        if (current_node.get_node() == origin):
            at_origin = True
            
        curr_iter += 1
        
if __name__ == "__main__":
    romania_map = { 'Oradea' :[{'Zerind':71},{'Sibiu':151}],
                   'Zerind':[{'Oradea':71},{'Arad':75}],
                   'Arad':[{'Zerind':75},{'Sibiu':140},{'Timisoara':118}],
                   'Timisoara':[{'Arad':118},{'Lugoj':111}],
                   'Lugoj':[{'Mehadia':70},{'Timisoara':111}],
                   'Mehadia':[{'Drobeta':75},{'Lugoj':70}],
                   'Drobeta':[{'Mehadia':75},{'Craiova':120}],
                   'Craiova':[{'Drobeta':120},{'Rimnicu Vilcea':146},{'Pitesti':138}],
                   'Rimnicu Vilcea':[{'Craiova':146},{'Pitesti':97},{'Sibiu':80}],
                   'Sibiu':[{'Oradea':151},{'Arad':140},{'Fagaras':99},{'Rimnicu Vilcea':80}],
                   'Fagaras':[{'Sibiu':99},{'Bucharest':211}],
                   'Pitesti':[{'Rimnicu Vilcea':97},{'Craiova':138},{'Bucharest':101}],
                   'Bucharest':[{'Pitesti':101},{'Giurgiu':90},{'Fagaras':211},{'Urziceni':85}],
                   'Giurgiu':[{'Bucharest':90}],
                   'Urziceni':[{'Bucharest':85},{'Hirsova':98},{'Vaslui':142}],
                   'Hirsova':[{'Urziceni':98},{'Eforie':86}],
                   'Eforie':[{'Hirsova':86}],
                   'Vaslui':[{'Urziceni':142},{'Iasi':92}],
                   'Iasi':[{'Vaslui':92},{'Neamt':87}],
                   'Neamt':[{'Iasi':87}]}
    
    romania_map = {'Oradea' :[{'Zerind':71},{'Sibiu':151}],
                   'Zerind':[{'Oradea':71},{'Arad':51}],
                   'Arad':[{'Zerind':75},{'Sibiu':140}],
                   'Sibiu':[{'Oradea':151},{'Arad':140},{'Fagaras':99},{'Rimnicu Vilcea':80}],
                   'Fagaras':[{'Sibiu':99},{'Bucharest':211}],
                   'Bucharest':[{'Pitesti':101},{'Fagaras':211}],
                   'Pitesti':[{'Rimnicu Vilcea':97},{'Bucharest':101}],
                   'Rimnicu Vilcea':[{'Pitesti':97},{'Sibiu':80}]}
    
#    romania_map = {'Oradea' :[{'Zerind':71},{'Sibiu':151}],
#               'Zerind':[{'Oradea':71},{'Arad':51}],
#               'Arad':[{'Zerind':75},{'Sibiu':140}],
#               'Sibiu':[{'Arad':140},{'Oradea':151}]}
    
    graph = Graph(romania_map)
    
#    print("Vertices of graph:")
#    print(graph.vertices())
#
#    print("Edges of graph:")
#    print(graph.edges())

    frontiers = graph.vertices()
    agent_hill_climbing(romania_map,'Oradea',frontiers)