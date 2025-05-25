import sys
import os
sys.path.append(os.path.realpath('./src/'))
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=20,linewidth=100000000000000)
np.set_printoptions(precision=20,suppress=True,linewidth=np.inf)
from src.utilFuncs import *
from src.optimizeFrame3 import OptimizeFrame
from src.mesher import MeshFrame

####################################
# check readme
####################################
params = {'base': 2.,'height':5.,'phi':0.0,'delta':0.0, 'nx': 1, 'ny':1, 'nz':1,
            'Name':'C4','Shape':'Square'}

scale = 25.4 # inch to mm 
H = 0.5*scale# 0.5 inch per row height

W = 1*scale  # 1 inch
B = 1*scale  # 1 inch

params['height'] =  H
params['base'] =  H

params['Name']='C4'
p = 2

mf = MeshFrame()

nodeXY,connectivity,radiiElemIndex = mf.generateCombined3DLattice(params)
# radiiNodIndex,radiiElemIndex help to map radii of one unit cell to multiple unit cell
numUnitLatElem = max(radiiElemIndex) + 1        

E = torch.tensor([17.0])
elemSize = 0.8 #0.8

midpointsZ = (nodeXY[connectivity[:,0],-1] 
             + nodeXY[connectivity[:,1],-1]) / 2
midpointsZ = np.round(midpointsZ,decimals=3)
midpointsUnique = np.unique(midpointsZ)

alpha = []
if p == 2:
    alphaTop = (midpointsZ>max(nodeXY[:,-1])/2)*1.0 + (midpointsZ == max(nodeXY[:,-1])/2)*0.5
    alphaBot = (midpointsZ<max(nodeXY[:,-1])/2)*1.0 + (midpointsZ == max(nodeXY[:,-1])/2)*0.5
    alpha.append(alphaBot);
    alpha.append(alphaTop)
else:       
    if len(midpointsUnique) > p:
        listComb = [1,2,2,3,2,2,1] # group few of the beams together to apply same radius
        count = 0;
        for i in range(len(listComb)):
            if listComb[i] == 1:
                mat = 1.0*(midpointsZ == midpointsUnique[count]) 
                count +=1
            elif listComb[i] == 2:
                mat1 = 1.0*(midpointsZ == midpointsUnique[count]) 
                mat2 = 1.0*(midpointsZ == midpointsUnique[count+1])
                mat = mat1 + mat2
                count +=2
            elif listComb[i] == 3:
                mat1 = 1.0*(midpointsZ == midpointsUnique[count]) 
                mat2 = 1.0*(midpointsZ == midpointsUnique[count+1])
                mat3 = 1.0*(midpointsZ == midpointsUnique[count+2])
                mat = mat1 + mat2 + mat3
                count += 3 
            alpha.append(mat)
    else:
        for i in range(len(midpointsUnique)):
            alpha.append(1.0*(midpointsZ == midpointsUnique[i]))
        
alpha = np.stack(alpha,axis=0)
r0 = np.ones(p)*0.5 # 0.5 radius in mm
r0 = np.array([0.25,0.75])
R = np.einsum('ij,i->j',alpha,r0)

plotStructure(R,nodeXY,connectivity,str(p)+' unique radius ',plotDeformed = False,TrueScale=True,fig=plt.figure(1),
    thicknessPlot=True,elemAnnotate=False,nodeAnnotate=False)


inputCurve = (np.array([0.0000000000000000, 0.6350000000000000, 1.2700000000000000,
1.9050000000000000, 2.5400000000000000, 3.1749999999999998,
3.8099999999999996, 4.4449999999999994, 5.0799999999999992,
5.7149999999999990, 6.3499999999999988])/1.25, 
2.0*np.array([0.0000000000000000, 1,1,1,1,1,1,1,1,1,1])) # 

max_x = 1.0
min_x = 0.2
ObjType = "FD"   
        
varSetup = {'min_x':min_x,'max_x':max_x}
path = params['Name']+'NN'+str(p)+'var/' # to save the results and data

geometrySavePath = 'Geometry/'+params['Name']

if not  os.path.exists(geometrySavePath+'.igs'):
    saveIGS(geometrySavePath,nodeXY,connectivity)
else:
    print("Geometry file already exists, skipping IGS generation")
    
print("##########################################")

seedNum = 0
if p==2:
    dataType = params['Name']+'uniform_2var_seedNum0'
else:
    dataType = params['Name']+'LHS_'+str(p)+'var_seedNum'+str(seedNum)
    
dataSavePath = 'DataVar/'+dataType+'.txt'
if not os.path.exists(dataSavePath):
    dataGen(p,2500,dataSavePath,seedNum)
else:
    print("Data file already exists, skipping data generation")

print("##########################################")
abaqusResultsPath = path +'abaqusComp40'+dataType+'.txt'
print(abaqusResultsPath)
# Run abaqus analysis to get data 
if not os.path.exists(abaqusResultsPath):
    print("Run abaqus on the data file:",dataSavePath)
    print(f"Abaqus file {abaqusResultsPath} does not exists")
    sys.exit()
else:
    print("Abaqus analysis file already exists, continuing to check surrogate model")
    
print("##########################################")
# Build NN surrogate
surrogateModelLoc = path + 'SurrogateModel.pth'
if not os.path.exists(surrogateModelLoc):
    print("Surrogate model does not exist, run the surrogate model training using SurrogateBuild.ipynb")
    sys.exit()
else:
    print("Surrogate model already exists, using it")

print("##########################################")
print("Running optimization usign the NN surrogate start....")
print("Target Curve = ",inputCurve)

########################################################
frameNN = torch.load(surrogateModelLoc)# switch device to cpu if needed,map_location=torch.device('cpu')) # Load to device
frameNN.eval()  # Set to evaluation mode
optFrame = OptimizeFrame(frameNN,inputCurve)
inputdim = optFrame.frameNN.nnSettings['inputDim']

fileSavePath = path+'NNmultiStart' # to save the figures of optimization run
# Check if the savePath exists
if not os.path.exists(fileSavePath):
    # Create the directory
    os.makedirs(fileSavePath)
    
st = time.perf_counter()
Algo = 'MMA'# 'L-BFGS-B', 'SLSQP','TNC'
seedOptNum = 2 #2 for C4, and 3 for C12 
np.random.seed(seedOptNum)
print("seed number for optimization = ", seedOptNum)
x0_all = np.random.rand(5,inputdim)
xopt_all = np.zeros_like(x0_all)
for i in range(x0_all.shape[0]):
    print(f"x0 = {x0_all[i,:]}, for start point = {i+1}")
    x,fun=optFrame.optimizerRun(x0_all[i,:],algorithm=Algo,tol=1e-10,fileSaveLoc=fileSavePath)
    print("Opt x = ",x)
    xopt_all[i,:] = x
    print("##################################################")
    
Total_opt_time = time.perf_counter()-st
print("Time for optimization in sec = ", Total_opt_time)   
print("##########################################")

###################
output_filename = 'DataVar/'+params['Name']+'solution_'+str(p)+'var_seedNum0.txt'

with open(output_filename, 'w') as f:
    f.write(f"# Seed: {0}, Variables: {p}, Samples: {10}\n")
    f.write(f"# Each row represents a sample in the format [val1 ... valp]\n")

    # Write the samples data row by row in the desired format
    for sample_row in np.concatenate((x0_all,xopt_all),axis=0):
        # Format each float value to a specific number of decimal places and join with a space
        # Then wrap the entire string in square brackets
        formatted_row = " ".join([f"{val:.16f}" for val in sample_row]) # Using .16f for high precision
        f.write(f"[{formatted_row}]\n")
print("Initial guesses and optimization results saved to file:",output_filename)
print(f" Run abaqus on these {output_filename} for validation")
###################

print("##########################################")
solutionfilename = path + 'abaqusComp40'+params['Name']+'solution_'+str(p)+'var_seedNum0.txt'
if not os.path.exists(solutionfilename):
    print("Run abaqus on result file:",output_filename)
    sys.exit()
else:    
    print(f"Abaqus analysis file {solutionfilename} already exists, using it. Delete this if its an older file")
    #### read the abaqus solution curves
    R, U, F, analysisState = extract_ABQ_values(solutionfilename,torch.from_numpy(inputCurve[0]).reshape(-1)) # for abaqus code results
    DataAbaqusOpt = np.stack((U[5::,:],F[5::,:]),axis=2)
    analysisState = analysisState[5::]
            
    V0   = np.zeros_like(x0_all[:,0])
    Vopt = np.zeros_like(x0_all[:,0])
        
    V0, Vopt,VDataAbaqusOpt = plot_optimization_curves(optFrame, x0_all, xopt_all, V0, Vopt, DataAbaqusOpt)
    plt.savefig(path+'/Opt'+str(inputdim)+'var.png', bbox_inches='tight') # 'tight' removes extra whitespace
    
    print("########")
    print("x0_all,xopt_all = ",x0_all,xopt_all)
    print("V0,Vopt,VDataAbaqusOpt  = ",V0,Vopt,VDataAbaqusOpt)
    print("########")
    
    VDataAbaqusOpt[analysisState==0.0] = 0.0
    barPlotofOptimization(V0,Vopt,VDataAbaqusOpt)
    plt.savefig(path+'/BarOpt'+str(inputdim)+'var.png', bbox_inches='tight') # 'tight' removes extra whitespace
    
    if inputdim==2:
        xGlobalNN,ax =plot_objective_landscape(optFrame, 50)
    
        ax = plotOptxk(ax,optFrame,x0_all,"Initial Guesses ",'red','o',markersize=100)
        ax = plotOptxk(ax,optFrame,xopt_all,"Optimized Results ",'aqua' ,'^',markersize=100)
        
        ax = plotOptxk(ax,optFrame,xGlobalNN,"Gridsearch Minima ",'lime' ,'h',markersize=200)
        
        ax.legend(loc='lower right',fontsize=16,bbox_to_anchor=(0.95,0.05))
        plt.savefig(path+'/ObjSpaceNNFullData.png', bbox_inches='tight') # 'tight' removes extra whitespace
