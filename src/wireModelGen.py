import os
# import sys
# sys.path.append(os.path.realpath('./src/'))
from mesher import *
import numpy as np

# Define the path to script directory
script_dir = os.getcwd()

def getWireModel(params=None):
    # lets set some params
    
    if params is None:
        params = {'base': 12.7,'height':12.7,'phi':0.0,'delta':0.0, 'nx': 3, 'ny':3, 'nz':1,
                    'Name':'C4','Shape':'Square'}
    
    p = 2
    params['beta'] = 1.0 # keep 1 as default
    params['xy_squeeze'] = 1.0 # keep 1 as default
    mf = MeshFrame()
    
    nodeXY,connectivity,radiiElemIndex = mf.generateCombined3DLattice(params)
    xyzV = np.concatenate((nodeXY[connectivity[:,0],:],nodeXY[connectivity[:,1],:]),axis=1)
    return xyzV
    

#  test fun

if __name__ == "__main__":
    params = {'base': 2.,'height':5.,'phi':0.0,'delta':0.0, 'nx': 3, 'ny':3, 'nz':1,
                        'Name':'C12','Shape':'Square', 'beta':1.0} # Square, Hexagon
    scale = 25.4 # inch to mm 
    H = 0.5*scale# 0.5 inch per row height

    W = 1*scale  # 1 inch
    B = 1*scale  # 1 inch

    params['height'] =  H
    params['base'] =  H

    xyzALL = getWireModel(params)

    filename = 'Geometry/wireModel' + params['Name'] + params['Shape'][0]  + str(params['nx'])+ str(params['ny']) +  '.npy'
    np.save(filename, xyzALL)


# np.save(filename, xyzALL)
# print("wire model saved to "+filename)
# print("Size of wire model:", xyzALL.shape)