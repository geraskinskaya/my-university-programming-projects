# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:27:29 2021

@author: doist
"""

import numpy

# implements controller structure for player
class player_controller(object):
	
    def control(self, inputs, controller = None):
        '''
        Parameters
        ----------
        inputs : array of integers of size 20
            inputs from environment
        controller : Controller class, optional
            Defines the algorithm that takes an action based on the inputs. The default is None.

        Returns
        -------
        list
            List of booleans indicating whether an action is taken or not: ['walk left', 'walk right', 'jump', 'shoot', 'release jump'].

        '''
        
        # if no controller is defined, take random actions
        if controller == None:
            action1 = numpy.random.choice([1,0])
            action2 = numpy.random.choice([1,0])
            action3 = numpy.random.choice([1,0])
            action4 = numpy.random.choice([1,0])
            action5 = numpy.random.choice([1,0])
            action6 = numpy.random.choice([1,0])
        else:
            action = controller.activate(inputs)
            action1 = 1 if action[0] > 0 else 0 
            action2 = 1 if action[1] > 0 else 0
            action3 = 1 if action[2] > 0 else 0
            action4 = 1 if action[3] > 0 else 0
            action5 = 1 if action[4] > 0 else 0
            action6 = 1 if action[5] > 0 else 0

        return [action1, action2, action3, action4, action5, action6]