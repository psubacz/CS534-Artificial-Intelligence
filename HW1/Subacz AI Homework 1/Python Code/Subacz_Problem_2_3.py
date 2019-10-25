# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:11:16 2019

@author: psubacz

As stated on page 38:
Consider the simple vacuum-cleaner agent that cleans a square if it is dirty and moves to the
other square if not; this is the agent function tabulated in Figure 2.3. Is this a rational agent?
That depends! First, we need to say what the performance measure is, what is known about
the environment, and what sensors and actuators the agent has. Let us assume the following:
    
• The performance measure awards one point for each clean square at each time step,
over a “lifetime” of 1000 time steps.

• The “geography” of the environment is known a priori (Figure 2.2) but the dirt distribution
and the initial location of the agent are not. Clean squares stay clean and sucking
cleans the current square. The Left and Right actions move the agent left and right
except when this would take the agent outside the environment, in which case the agent
remains where it is.

• The only available actions are Left , Right, and Suck.

• The agent correctly perceives its location and whether that location contains dirt.

• The dirt distrubtion has been abstracted in this enviroment to be a square that is dirty or clean



NOTE: There is a slight variance with the performance measure due to the random generation of which 
    floor the robot spawn in! Therefore if the robot spawns on a clean floor then moves to a dirty floor 
    then the performance measure should read 999. If the robot spawn on a dirty floor the performance
    measure should read 1000
"""


import numpy as np
import random

class Floor:
    '''
    Class to contain floor object with attributes
    '''
    def __init__(self,location,isDirty = True):
        '''
        Initialize Floor class for each instance of floor object
        '''
        self.location = location
        self.isDirty = isDirty
        self.dirtDistribution = None
    
    def get_location(self):
        '''
        Returns name of floor space
        '''
        return self.location
    
    def is_dirty(self):
        '''
        Returns True if floor is dirty and False is floor is clean.
        '''
        return self.isDirty
    
    def set_dirty(self):
        '''
        Sets isDirty attribute to True, indacates floor has been made dirty
        '''
        self.isDirty = True
        
    def set_clean(self):
        '''
        Sets isDirty attribute to False, indacates floor has been made clean
        '''
        self.isDirty = False

class Robot:
    def __init__(self, position = None, debug = False,preset_clean_floor = None):
        '''
        Initialize robot with random position if None is given, and Floor ennviroment
        '''
        #Assumption is that the enviroment is a known priori  but the dirt distribution
        #and the initial location of the agent are not. 
        self.position = position
        self.debug = debug
        #Performance Measures are incremented when a floor has been detected as clean
        self.performance_measure = 0
        self.performed_actions = 0
        self.percept_sequence = []
        #There are two known floor Locations, left = 0 and right = 1
#        if (preset_clean_floor == None):
#            #Both floors are dirty
#            self.floor_locations = [Floor('left'),Floor('right')]
        if(preset_clean_floor==0):
            self.floor_locations = [Floor('left',False),Floor('right')]
        elif(preset_clean_floor==1):
            self.floor_locations = [Floor('left'),Floor('right',False)]
        elif(preset_clean_floor==2):
            self.floor_locations = [Floor('left',False),Floor('right',False)]
        else:
            self.floor_locations = [Floor('left'),Floor('right')]
        #Generate floor Location
        self.get_robot_position()
    
    def get_robot_position(self):
        '''
        '''
        #If robot position is unknown, randomyl generate a location from possible floor locations
        if (self.position == None):
            self.position = random.randint(0, len(self.floor_locations)-1)
        return self.position
    
    def move_left(self):
        '''
        '''
        #The Left actions move the agent left except when this would take the agent 
        #outside the environment, in which case the agent remains where it is.
        self.position = 0
    
    def move_right(self):
        '''
        '''
        #The Right actions move the agent right except when this would take the agent 
        #outside the environment, in which case the agent remains where it is.
        self.position = 1
    
    def clean_floor(self,floor_location):
        '''
        '''
        #Set floor to clean(false) and
        while(self.percieve_dirt()):
            self.floor_locations[self.position].set_clean()
        
    def percieve_dirt(self):
        '''
        Percieve if the floor is dirty, return true if dirty and false if clean
        '''
        if (self.floor_locations[self.position].is_dirty()==True):
            return True
        else:
            return False
    
    def reflex_agent(self):
        '''
        A Simple Reflex agent as stated on page 49:
        function SIMPLE-REFLEX-AGENT(percept ) returns an action
            persistent: rules, a set of condition–action rules
            state←INTERPRET-INPUT(percept )
            rule←RULE-MATCH(state, rules)
            action ←rule.ACTION
            return action
        '''
        #Percepts
        state = self.percieve_dirt()
        just_cleaned = False
        
        #Log the percept sequence
        self.percept_sequence.append((self.position,state))
        
        #Rule 1 - Clean a dirty floor
        if state:
            #Action
            self.clean_floor(self.floor_locations[self.position])
            self.performed_actions += 1
#            print('Floor Cleaned')
            self.performance_measure += 1
            just_cleaned =True
            
        #Rule 2 -  Floor clean, Move to next area
        elif (self.position == 1):
            #Action
            self.move_left()
            self.performed_actions += 1
#            print('Moving left...')
            
        #Rule 3 -  Floor clean, Move to next area
        elif (self.position == 0):
            #Action
            self.move_right()
            self.performed_actions += 1
#            print('Move right...')
            
        #Reward - Reward 1 point if floor is clean dont double count points!
        if (self.floor_locations[self.position].is_dirty() == False) and (just_cleaned == False):
            self.performance_measure += 1
            
        #Debug print statements
        if (self.debug == True):
            print("self.performance_measure: ",self.performance_measure)
            print("self.performed_actions: ",self.performed_actions)
        
    def get_stats(self):
        '''
        Returns the a tuple of actions, performance measure, percept sequence 
        '''
        return (self.performed_actions,self.performance_measure,self.percept_sequence)
    
    
def run_enviroment(MAX_TIME_STEPS = 1000):
    '''
    '''
    #Robot - Both floors dirty
    cleaningRobot = Robot()
    #Run Enviroment for MAX_TIME_STEPS
    for step in range(0,MAX_TIME_STEPS):
        cleaningRobot.reflex_agent()
    stats0 = cleaningRobot.get_stats()
    print('\nRobot - Both floors dirty')
    print('performed_actions: ',stats0[0])
    print('performance_measure: ',stats0[1])
    print('Avg performance_measure: ',stats0[1])
    #print('percept_sequence: ', stats0[2])
    #clean up
    del cleaningRobot
    
    #Robot - left floor dirty
    cleaningRobot = Robot(preset_clean_floor = 0)
    #Run Enviroment for MAX_TIME_STEPS
    for step in range(0,MAX_TIME_STEPS):
        cleaningRobot.reflex_agent()
    stats1 = cleaningRobot.get_stats()
    print('\nRobot - left floor dirty')
    print('performed_actions: ',stats1[0])
    print('performance_measure: ',stats1[1])
    #print('percept_sequence: ', stats1[2])
    #clean up
    del cleaningRobot
    
    #Robot - right floor dirty
    cleaningRobot = Robot(preset_clean_floor = 1)
    #Run Enviroment for MAX_TIME_STEPS
    for step in range(0,MAX_TIME_STEPS):
        cleaningRobot.reflex_agent()
    stats2 = cleaningRobot.get_stats()
    print('\nRobot - right floor dirty')
    print('performed_actions: ',stats2[0])
    print('performance_measure: ',stats2[1])
    #print('percept_sequence: ', stats2[2])
    #clean up
    del cleaningRobot
    
    #Robot - Both floors clean
    cleaningRobot = Robot(preset_clean_floor = 2)
    #Run Enviroment for MAX_TIME_STEPS
    for step in range(0,MAX_TIME_STEPS):
        cleaningRobot.reflex_agent()
    stats3 = cleaningRobot.get_stats()
    print('\nRobot - Both floors clean')
    print('performed_actions: ',stats3[0])
    print('performance_measure: ',stats3[1])
    #print('percept_sequence: ', stats3[2])
    #clean up
    del cleaningRobot
    APM = (stats0[1]+stats1[1]+stats2[1]+stats3[1])/4
    print('\nAvg Performance Measure: ', APM)
    
if __name__ == "__main__":
    '''
    Run enviroment and display performance_measure
    '''
    run_enviroment()