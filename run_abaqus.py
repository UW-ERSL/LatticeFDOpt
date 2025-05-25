import os
import time
import sys
sys.path.append(os.path.realpath('./src/'))
from abaqusFEA import *

    
unitCellName = 'C4'
p = 7 # number of variables
sampleType ="LHS" # uniform or LHS or solution 
seedNum = 1 # 0 or 1
compression = 40.0 # percentage of compression 
meshsize = 0.8 #mm

# Note: all the following combinations are availables
# (2, "uniform", 0) # data gen, only C4
# (2, "solution", 0) # validation, only C4
# (7, "LHS", 0)
# (7, "LHS", 1)
# (7, "solution", 0) # validation


# Define the path to script directory
script_dir = os.getcwd()

job_name = 'rad'+unitCellName
node_set_name = 'MOVEDOWN'
lattice_z_height = 12.7 #mm
target_u3 = lattice_z_height*compression*0.01 #mm

if sampleType == 'solution':
    seedNum = 0
if p==2: 
    unitCellName = 'C4'
    if sampleType == "LHS":
        sampleType ="uniform"
        seedNum = 0
elif p==7:
    if sampleType == "uniform":
        sampleType ="LHS"
    
input_file = os.path.join(script_dir,'DataVar/'+unitCellName+sampleType+ "_"+str(p) +"var""_seedNum"+str(seedNum)+".txt") # C4uniform_var2, C4LHS_var2_seedNum0, C4LHS_var2_seedNum1
output_file = os.path.join(script_dir,unitCellName+'NN'+str(p)+'var/'+"abaqusComp"+str(int(compression))+unitCellName+ sampleType + "_"+str(p) +"var_seedNum" + str(seedNum) + ".txt")

input_ij = read_input_ij(input_file)
# input_ij =input_ij[0:1] # for testing just one line
total_runs = len(input_ij)
total_time = 0.0

removeExtras(script_dir)
result_lines = []
hit_count = 0
counter = 0
Li = 0.2 #mm
Ui = 1.0 #mm

##
# Define the IGES file path
iges_file = os.path.join(script_dir, 'Geometry/'+unitCellName +'.igs')

# Open the IGES file
mdb.openIges(iges_file, msbo=False, scaleFromFile=OFF,
             topology=WIRE, trimCurve=DEFAULT)

# Create a part from the imported geometry
mdb.models['Model-1'].PartFromGeometryFile(combine=False, convertToAnalytical=1,
                                         dimensionality=THREE_D, geometryFile=mdb.acis, name=unitCellName, stitchEdges=1,
                                         stitchTolerance=1.0, topology=WIRE, type=DEFORMABLE_BODY)
                                         
for k in range(total_runs):

    x0 = radiusValue(input_ij[k],Li,Ui)

    start_time = time.time()  # Start the timer
    
    if unitCellName == 'C4':
       runFEAC4(mdb,x0, job_name,target_u3,meshsize) # target u2 is Uz_mm, displacement in -ve direction, meshsize to control the length of element
    elif unitCellName == 'C12':
       runFEAC12(mdb,x0, job_name,target_u3,meshsize) # target u2 is Uz_mm, displacement in -ve direction, meshsize to control the length of element

    # wait for job to finish by looking at lck file existance
    lckFile = os.path.join(script_dir, job_name + '.lck')
    while os.path.isfile(lckFile) == True:
        time.sleep(0.1)
    
    elapsed_time = time.time()  - start_time
    total_time += elapsed_time
    
    odb_name = os.path.join(script_dir, job_name + '.odb')
    
    # if odb file exists, read results, or skip
    if os.path.exists(odb_name):
        avg_u3_series, total_rf3_series = readOdbRes(odb_name,node_set_name)
        line, hit_count, counter = process_line(
           input_ij[k], avg_u3_series, total_rf3_series,
            target_u3, hit_count, counter, total_runs, elapsed_time
        )

        # result_lines.append(line)
        if os.path.exists(odb_name):
            os.remove(odb_name)

    else:
        line, hit_count, counter = process_line(
            input_ij[k], [0,0,0,0], [0,0,0,0],
            target_u3, hit_count, counter, total_runs, elapsed_time
        )
         
    ## save line to file
    if counter == 1:
        # Write all results to file
        with open(output_file, "w") as f:
            f.write(line + "\n")
    else:    
        # Append all results to file
        with open(output_file, "a") as f:
            f.write(line + "\n")
            
## Save the total time
line = "Total time = {:0.5f}".format(total_time)
with open(output_file, "a") as f:
            f.write(line + "\n")

removeExtras(script_dir)
