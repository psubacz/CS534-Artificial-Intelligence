# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:08:19 2019

@author: psubacz
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:15:34 2019

@author: psubacz
"""
import random as rd

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



def agent_genetic(graph,_map,origin,breakpoint = 10,pop_mult = 2, mutation_rate = 25, cull_pop_size = 2**3):
    '''
    Implementation of a genetic agent to solve a TSP problem
    The cities are incoded as [Oradea ‣ 1, Zerind ‣ 2, Arad ‣ 3, Sibiu ‣ 4,
     Fagaras ‣ 5, Bucharest ‣ 6, Pitesti ‣7, Rimnicu Vilcea ‣ 8]
    '''

    def fitness_fuction(population):
    #    1. What is the total distance traveled 
    #    2. Are all cities accounted for
    #    3. Are we at the origin
    #    4. Can we travel to all
        population[1] = 0 # Recalcualte the cost
        org_vrtx = population[0][0] #Get the previous city
        vrtx = population[0][1] # Get the current city
#            Assume path is bad
        is_not_traversable = False 
        for vrtx in population[0]:
            #if we are not moving
            if (graph.vertices()[org_vrtx] != graph.vertices()[vrtx]):
                #If we can move from Point A to point B with the graph or vise versa
                if ((list(graph.edges()[org_vrtx]).count(graph.vertices()[vrtx])>=1)or
                    (list(graph.edges()[vrtx]).count(graph.vertices()[org_vrtx])>=1)):
                    # Look at the verticies and set the cost of the encoding calcualte the total cost
                    for neighbor in _map[graph.vertices()[org_vrtx]]:
                        if(list(neighbor)[0] == graph.vertices()[vrtx]):
                            population[1] = population[1] + neighbor[graph.vertices()[vrtx]]
                else:
                    is_not_traversable = True
            #if we are at the same city (have not moved), do not count the cost
            elif(graph.vertices()[org_vrtx] == graph.vertices()[vrtx]):
                    pass
            #Set to new vrtx
            org_vrtx = vrtx
        if (is_not_traversable):
            population[1] = -1
        #calculate the squashed cost of each function
        if (population[1] != 0):
            population[2] = len(graph.vertices())**(1/population[1])
#        print(population)
        return population

    def random_selection(population):
        '''
        Randomly selects a pop from the total possible population and pass 
        in the fitness function
        '''
        rand = rd.randint(0,len(population)-1)
        population[rand]=fitness_fuction(population[rand])
        return population[rand]
    
    def reproduce(pop_1,pop_2):
        '''
        Reproduce the population using the partially-mapped crossover 
        path encoding 
        '''
        pop_3 = pop_1
        pop_1[0][1:len(pop_1[0])-1]
        pop_2[0][1:len(pop_1[0])-1]
        pop_3[0][1:len(pop_1[0])-1] = pop_2[0][1:len(pop_1[0])-1]
        return pop_3
        
    def mutate(child):
        '''
        Mutate the child by randomly inserting city into the encoding. 
        The city and location of the mutation are random
        '''
        z = rd.randint(1,len(child[0])-2)
        q = rd.randint(0,len(graph.vertices())-1)
        child[0][z] = q
        return child

    def cull_pop(population,surivors_bias=len(graph.vertices())**10 ):
        '''
        Culls the population based on cost of the encoding. Optional survivors 
        bias can be passed as a parameter to increase the difficulty to survive 
        the culling.
        '''
        new_bias = 0
        if (surivors_bias <= 1):
            surivors_bias=len(graph.vertices())**10 
        elif(surivors_bias>1):
            new_bias=surivors_bias
        for i in range(len(population)):
            if (len(population)<=8):
                break
            #randomly pick a pop
            i = rd.randint(0,len(population)-1)
            #We want to maximize this number, the higher this number is the 
            #  closer the agent is to a solution. As the survivors_bias shinks,  
            #  the minimum check to surivve will require better scores will only get lower
            if (population[i][1]<=surivors_bias):
                if (population[i][1]>=new_bias):
                    new_bias =population[i][1]
                population.pop()
        return population,new_bias
    
    def init_population(num_population,pop_mult=1):
        '''
        Initialize the population based on the number of cities passed to the 
        algorithm. Optional population multipler incase the encoding cannot 
        produce an optimal solution without making return trips
        '''
        population = [[],0,0]
        for i in range(pop_mult):
            for i in range(0,num_population):
                population[0].append(i)
            rd.shuffle(population[0])
        population[0][0] = 0
        population[0][-1] = 0
        return population
    
    def decode_pop(population):
        '''
        Decodes the population to strings
        '''
        path = [[],0]
        for city in population[0]:
            path[0].append(graph.vertices()[city])
        path[1] = population[1]
        return path
    
    def goal_test(population):
        '''
        Tests to see if population has meet the goal
        '''
        results = []
        counter = 0
        for pop in population:
            if (pop[1]>=0):
                results.append(decode_pop(pop))
                counter += 1
            else:
                pass
        if (len(results) >= 1):
            return results
        else:   
            return None
    
    population = [init_population(len(graph.vertices()),pop_mult),
                  init_population(len(graph.vertices()),pop_mult)]
    
    iteration = 0
    bias = 0
    generation = 0
    while True:
        #Goal test!
        if iteration >= breakpoint:
            results = goal_test(population)
            if (results is not None):
                #return solutions
                return results
            else:
                #keep going, result the interations and increment the
                #generateion counter
                iteration = 0
                generation = generation+breakpoint
           
        print('Iteration: ',iteration+generation)
        new_population = []
        #loop through the reproduction, fitness test and mutation chance for 
        # each population.
        for i in range(len(population)):
            #randomly select pops and reproduce
            pop_1 = random_selection(population)
            pop_2 = random_selection(population)
            child = reproduce(pop_1,pop_2)
            #mutation based on random chance
            if (rd.random() <= mutation_rate):
                mutate(child)
                child = mutate(child)
            new_population.append(child)        
        for pop in new_population:
            population.append(pop)
        
        #Reduce memory required and reduce the population
        if (((cull_pop_size)%len(population))>=1):
            population,bias = cull_pop(population,bias)
        #Count the number of iterations
        iteration+=1
        
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
                   'Bucharest':[{'Pitesti':101},{'Fagaras':211},{'Urziceni':85}],
                   'Pitesti':[{'Rimnicu Vilcea':97},{'Bucharest':101}],
                   'Rimnicu Vilcea':[{'Pitesti':97},{'Sibiu':80}]}
    
    romania_map = {'Oradea' :[{'Zerind':71},{'Sibiu':151}],
               'Zerind':[{'Oradea':71},{'Arad':51}],
               'Arad':[{'Zerind':75},{'Sibiu':140}],
               'Sibiu':[{'Arad':140},{'Oradea':151}]}
    graph = Graph(romania_map)
    
    print('Something need doing?')
    population = agent_genetic(graph,romania_map,'Oradea',pop_mult=1,breakpoint = 1000,cull_pop_size = 2**8)
    print('Jobs Done!')
    print('Possible paths:')
    print(population)
        