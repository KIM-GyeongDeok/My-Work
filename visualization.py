# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:48:48 2020

@author: 김병철
""" 
import matplotlib.pyplot as plt
from medpy.io import load

def showImange_From_File(DIR, Class = "None"):
    data, header = load(DIR)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(data, edgecolor='k')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z") 
    plt.suptitle("Predicted Class : %s"%Class, fontsize = 16)
    plt.show()      
 
def showImange_From_Array(data, Class = "None"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(data, edgecolor='k')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z") 
    plt.suptitle("Predicted Class : %s"%Class, fontsize = 16)
    plt.show()    
    

# =============================================================================
#  DIR = "testdata\\2f55d919-f3a5-601b-f5cd-36f060bcf8f9.mhd"
#  showImange_From_File(DIR)
#  data, header = load(DIR)
#  showImange_From_Array(data)
# =============================================================================

