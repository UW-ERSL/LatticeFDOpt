
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *

# Function for postprocssing odb
def readOdbRes(odb_name,node_set_name):
    odb = openOdb(path=odb_name, readOnly=True)
    assembly = odb.rootAssembly
     
    step = list(odb.steps.values())[-1]
    frames = step.frames
    node_set = assembly.nodeSets[node_set_name]
    avg_u3_series = []
    total_rf3_series = []

    for frame in frames:
        u_field = frame.fieldOutputs['U']
        rf_field = frame.fieldOutputs['RF']

        subsetU = u_field.getSubset(region=node_set)
        subsetRF = rf_field.getSubset(region=node_set)

        avgU3 = get_average_u3(subsetU)
        totalRF3 = get_total_rf3(subsetRF)

        avg_u3_series.append(-avgU3)
        total_rf3_series.append(-totalRF3)
    
    odb.close()

    return avg_u3_series,total_rf3_series
# Function to compute average U3
def get_average_u3(subset):
    total = 0.0
    count = 0
    for v in subset.values:
        if len(v.data) >= 3:
            total += v.data[2]
            count += 1
    return total / count if count > 0 else None

# Function to compute total RF3 (reaction force in Z)
def get_total_rf3(subset):
    total_rf3 = 0.0
    for v in subset.values:
        if len(v.data) >= 3:
            total_rf3 += v.data[2]  # RF3
    return total_rf3
            
def radiusValue(i_list,Li,Ui):
    # using Ui and Li to convert 0-1 to 0.2,1.0 mm radius
    
    return [Li + i * (Ui - Li) for i in i_list]
    
def process_line(x0, avg_u3_series, total_rf3_series, target_u3, hit_count, counter, total_runs, elapsed_time):
    # Final values from last frame
    final_u3 = avg_u3_series[-1]
    final_rf3 = total_rf3_series[-1]
    max_rf3 = max(total_rf3_series)

    # Check if final displacement is close to the target
    is_close = int(abs(final_u3 - target_u3) < 0.05)
    hit_count += is_close
    counter += 1

    # Format output line
    line = "res = (array([{}]), {}, {}, {}, tensor([{}]), tensor([{}]), tensor({:.20f}), {:.2f}, {:0.5f})".format(
        ", ".join("{:.17f}".format(x) for x in x0), 
        is_close, counter, total_runs,
        ", ".join("{:.17f}".format(x) for x in avg_u3_series),
        ", ".join("{:.17f}".format(x) for x in total_rf3_series),
        max_rf3,
        hit_count*100.0 / float(total_runs),
        elapsed_time
    )

    return line, hit_count, counter
def removeExtras(script_dir):
    # Extensions to keep
    allowed_extensions = ('.txt', '.igs', '.py','.ipynb','.npy','.md','.gitignore')

    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)
        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item.lower())
            if ext not in allowed_extensions:
                try:
                    os.remove(item_path)
                except Exception as e:
                    print("Could not remove") 

def read_input_ij(filepath):
    """
    Reads the input_ij.txt file and returns the i, j values as a list of lists.

    Args:
        filepath (str): The path to the input_ij.txt file.

    Returns:
        list: A list of lists, where each inner list contains two floats [i, j].
              Returns an empty list if the file is not found or an error occurs.
    """
    input_data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            line = line.strip().strip('[]')  # Remove brackets and extra whitespace
            values = line.split()
            if len(values) == 2:
                i = float(values[0])
                j = float(values[1])
                input_data.append([i, j])
            else:
                input_data.append([float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4]), float(values[5]), float(values[6])])    
    
    return input_data


def runFEAC4(mdb,x0,job_name,Uz_mm,meshsize, xyzV, minLoadStep):
    
    #create wire part
    mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=13.0)
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(0.0, 0.0), point2=(
        -1.0, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
        addUndoState=False, entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry[2])
    mdb.models['Model-1'].Part(dimensionality=THREE_D, name='C4', type=DEFORMABLE_BODY)
    mdb.models['Model-1'].parts['C4'].BaseWire(sketch=
        mdb.models['Model-1'].sketches['__profile__'])
    del mdb.models['Model-1'].sketches['__profile__']
    
    # print(xyzV).shape
    xyz = tuple(
    ((row[0], row[1], row[2]), (row[3], row[4], row[5]))
    for row in xyzV
    )
    
    mdb.models['Model-1'].parts['C4'].WirePolyLine(mergeType=MERGE, meshable=ON, 
        points=xyz)
    mdb.models['Model-1'].parts['C4'].Set(edges=
        mdb.models['Model-1'].parts['C4'].edges.getSequenceFromMask(('[#1eff ]', ), 
        ), name='Wire-2-Set-1')
            
    del mdb.models['Model-1'].parts['C4'].features['Wire-1']

    ### Material
    mdb.models['Model-1'].Material(name='E17Material')
    mdb.models['Model-1'].materials['E17Material'].Elastic(table=((17.0, 0.3), ))

    if len(x0) <= 2:
        r1 = x0[0]
        r2 = x0[0]
        r3 = x0[0]
        r4 = (x0[0] + x0[1]) / 2.0
        r5 = x0[1]
        r6 = x0[1]
        r7 = x0[1]
    else:
        r1 = x0[0]
        r2 = x0[1]
        r3 = x0[2]
        r4 = x0[3]
        r5 = x0[4]
        r6 = x0[5]
        r7 = x0[6]
        
    ### 7 radius
    mdb.models['Model-1'].CircularProfile(name='One', r=r1)
    mdb.models['Model-1'].CircularProfile(name='Two', r=r2)
    mdb.models['Model-1'].CircularProfile(name='Three', r=r3)
    mdb.models['Model-1'].CircularProfile(name='Four', r=r4)
    mdb.models['Model-1'].CircularProfile(name='Five', r=r5)
    mdb.models['Model-1'].CircularProfile(name='Six', r=r6)
    mdb.models['Model-1'].CircularProfile(name='Seven', r=r7)

        # total 7 set creation
    part = mdb.models['Model-1'].parts['C4']
    tol = 1e-6

    # Define Z ranges for each set
    ranges = [
        ('OneSet',   lambda z: abs(z) <= tol),
        ('TwoSet',   lambda z: (z > 0) and (z <= 1.859872)),
        ('ThreeSet', lambda z: (z > 1.859872) and (z < 12.7/2.0)),
        ('FourSet',  lambda z: (z >= 12.7/2.0 - tol) and (z <= 12.7/2.0 + tol)),
        ('FiveSet',  lambda z: (z > 12.7/2.0) and (z < 10.840128)),
        ('SixSet',   lambda z: (z >= 10.840128) and (z < 12.7)),
        ('SevenSet', lambda z: (z >= 12.7)),
    ]

    # Loop over each range
    for set_name, condition in ranges:
        edge_array = None
        for i, edge in enumerate(part.edges):
            x, y, z = edge.pointOn[0]
            if condition(z):
                if edge_array is None:
                    edge_array = part.edges[i:i+1]
                else:
                    edge_array = edge_array + part.edges[i:i+1]
        # Only create set if edges found
        if edge_array:
            part.Set(edges=edge_array, name=set_name)
        else:
            print("No edges found for:", set_name)

    ### Beam section and profile connect
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='OneSection', poissonRatio=0.0,
        profile='One', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='TwoSection', poissonRatio=0.0,
        profile='Two', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='ThreeSection', poissonRatio=0.0,
        profile='Three', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='FourSection', poissonRatio=0.0,
        profile='Four', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='FiveSection', poissonRatio=0.0,
        profile='Five', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='SixSection', poissonRatio=0.0,
        profile='Six', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='SevenSection', poissonRatio=0.0,
        profile='Seven', temperatureVar=LINEAR)

    ### Section assignment to set
    part.SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        part.sets['OneSet'], sectionName='OneSection',
        thicknessAssignment=FROM_SECTION)
    part.SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        part.sets['TwoSet'], sectionName='TwoSection',
        thicknessAssignment=FROM_SECTION)
    part.SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        part.sets['ThreeSet'], sectionName='ThreeSection',
        thicknessAssignment=FROM_SECTION)

    part.SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        part.sets['FourSet'], sectionName='FourSection',
        thicknessAssignment=FROM_SECTION)

    part.SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        part.sets['FiveSet'], sectionName='FiveSection',
        thicknessAssignment=FROM_SECTION)
    part.SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        part.sets['SixSet'], sectionName='SixSection',
        thicknessAssignment=FROM_SECTION)
    part.SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        part.sets['SevenSet'], sectionName='SevenSection',
        thicknessAssignment=FROM_SECTION)
            
        ## set the orientation
    part.assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        part.sets['OneSet'])
    part.assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        part.sets['TwoSet'])
    part.assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        part.sets['ThreeSet'])
    part.assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        part.sets['FourSet'])
    part.assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        part.sets['FiveSet'])
    part.assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        part.sets['SixSet'])
    part.assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        part.sets['SevenSet'])    
    
    ### Assembly
    mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)

    mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='C4-1', part=
        part)

    ### Step settings
    t_calc = (1.0/minLoadStep)
    mdb.models['Model-1'].StaticStep(initialInc=t_calc, maxInc=t_calc, maxNumInc=500, 
        name='Step-1', nlgeom=ON, previous='Initial', timePeriod=1.0, minInc=t_calc*1e-3)
    
    # if stabilization needed for NR
    # mdb.models['Model-1'].steps['Step-1'].setValues(adaptiveDampingRatio=None, 
    #     continueDampingFactors=False, stabilizationMagnitude=0.0002, 
    #     stabilizationMethod=DAMPING_FACTOR)

    ### Boundary condition

    # onset assigned to fixed bottom
    mdb.models['Model-1'].rootAssembly.Set(
        name='OneSetAsm',
        edges= mdb.models['Model-1'].rootAssembly.instances['C4-1'].sets['OneSet'].edges
    )
    mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Step-1', 
        distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
        'FixBot', region=mdb.models['Model-1'].rootAssembly.sets['OneSetAsm'], u1=0.0, 
        u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)
    
    # set seven assigned to move down for BC
    mdb.models['Model-1'].rootAssembly.Set(
        name='MoveDown',
        edges= mdb.models['Model-1'].rootAssembly.instances['C4-1'].sets['SevenSet'].edges
    )
    mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Step-1', 
        distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
        'MoveDownTop', region=mdb.models['Model-1'].rootAssembly.sets['MoveDown'], 
        u1=0.0, u2=0.0, u3=-Uz_mm, ur1=UNSET, ur2=UNSET, ur3=UNSET)
    
    ### Mesh 
    part.seedPart(deviationFactor=0.1, minSizeFactor=
        0.1, size=meshsize)
    part.generateMesh()
    part.setElementType(elemTypes=(ElemType(
        elemCode=B31, elemLibrary=STANDARD), ), regions=(
        part.edges.getSequenceFromMask((
        '[#ffffffff #f ]', ), ), ))
    mdb.models['Model-1'].rootAssembly.regenerate()

    ### create job
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
        memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF, 
        multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE, 
        numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=
        ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)# submit code	
    mdb.jobs[job_name].waitForCompletion()
    
    pass
    

def runFEAC12(mdb,x0,job_name,Uz_mm,meshsize, xyzV, minLoadStep):
        
    #create wire part
    mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=13.0)
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(0.0, 0.0), point2=(
        -1.0, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
        addUndoState=False, entity=
        mdb.models['Model-1'].sketches['__profile__'].geometry[2])
    mdb.models['Model-1'].Part(dimensionality=THREE_D, name='C12', type=DEFORMABLE_BODY)
    mdb.models['Model-1'].parts['C12'].BaseWire(sketch=
        mdb.models['Model-1'].sketches['__profile__'])
    del mdb.models['Model-1'].sketches['__profile__']

    # print(xyzV).shape
    xyz = tuple(
    ((row[0], row[1], row[2]), (row[3], row[4], row[5]))
    for row in xyzV
    )

    mdb.models['Model-1'].parts['C12'].WirePolyLine(mergeType=MERGE, meshable=ON, 
        points=xyz)
    mdb.models['Model-1'].parts['C12'].Set(edges=
        mdb.models['Model-1'].parts['C12'].edges.getSequenceFromMask(('[#1eff ]', ), 
        ), name='Wire-2-Set-1')
            
    del mdb.models['Model-1'].parts['C12'].features['Wire-1']

        ### Material
    mdb.models['Model-1'].Material(name='E17Material')
    mdb.models['Model-1'].materials['E17Material'].Elastic(table=((17.0, 0.3), ))

    if len(x0) <= 2:
        r1 = x0[0]
        r2 = x0[0]
        r3 = x0[0]
        r4 = (x0[0] + x0[1]) / 2.0
        r5 = x0[1]
        r6 = x0[1]
        r7 = x0[1]
    else:
        r1 = x0[0]
        r2 = x0[1]
        r3 = x0[2]
        r4 = x0[3]
        r5 = x0[4]
        r6 = x0[5]
        r7 = x0[6]
        
    ### 7 radius
    mdb.models['Model-1'].CircularProfile(name='One', r=r1)
    mdb.models['Model-1'].CircularProfile(name='Two', r=r2)
    mdb.models['Model-1'].CircularProfile(name='Three', r=r3)
    mdb.models['Model-1'].CircularProfile(name='Four', r=r4)
    mdb.models['Model-1'].CircularProfile(name='Five', r=r5)
    mdb.models['Model-1'].CircularProfile(name='Six', r=r6)
    mdb.models['Model-1'].CircularProfile(name='Seven', r=r7)

    # total 7 set creation
    part = mdb.models['Model-1'].parts['C12']
    tol = 1e-6

    # Define Z ranges for each set
    ranges = [
        ('OneSet',   lambda z: abs(z) <= tol),
        ('TwoSet',   lambda z: (z > 0) and (z <= 2.012869)),
        ('ThreeSet', lambda z: (z > 2.012869) and (z < 5.187869)),
        ('FourSet',  lambda z: (z >= 5.187869) and (z <= 7.512131)),
        ('FiveSet',  lambda z: (z > 7.512131) and (z < 10.687131)),
        ('SixSet',   lambda z: (z >= 10.687131) and (z < 12.7)),
        ('SevenSet', lambda z: (z >= 12.7)),
    ]

    # Loop over each range
    for set_name, condition in ranges:
        edge_array = None
        for i, edge in enumerate(part.edges):
            x, y, z = edge.pointOn[0]
            if condition(z):
                if edge_array is None:
                    edge_array = part.edges[i:i+1]
                else:
                    edge_array = edge_array + part.edges[i:i+1]
        # Only create set if edges found
        if edge_array:
            part.Set(edges=edge_array, name=set_name)
        else:
            print("No edges found for:", set_name)

        
    ### Beam section and profile connect
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='OneSection', poissonRatio=0.0,
        profile='One', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='TwoSection', poissonRatio=0.0,
        profile='Two', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='ThreeSection', poissonRatio=0.0,
        profile='Three', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='FourSection', poissonRatio=0.0,
        profile='Four', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='FiveSection', poissonRatio=0.0,
        profile='Five', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='SixSection', poissonRatio=0.0,
        profile='Six', temperatureVar=LINEAR)
    mdb.models['Model-1'].BeamSection(consistentMassMatrix=False, integration=
        DURING_ANALYSIS, material='E17Material', name='SevenSection', poissonRatio=0.0,
        profile='Seven', temperatureVar=LINEAR)

    ### Section assignment to set
    mdb.models['Model-1'].parts['C12'].SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['C12'].sets['OneSet'], sectionName='OneSection',
        thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['C12'].SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['C12'].sets['TwoSet'], sectionName='TwoSection',
        thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['C12'].SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['C12'].sets['ThreeSet'], sectionName='ThreeSection',
        thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['C12'].SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['C12'].sets['FourSet'], sectionName='FourSection',
        thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['C12'].SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['C12'].sets['FiveSet'], sectionName='FiveSection',
        thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['C12'].SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['C12'].sets['SixSet'], sectionName='SixSection',
        thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['C12'].SectionAssignment(offset=0.0, offsetField='',
        offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['C12'].sets['SevenSet'], sectionName='SevenSection',
        thicknessAssignment=FROM_SECTION)

    # assign the orientation for each set
    mdb.models['Model-1'].parts['C12'].assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        mdb.models['Model-1'].parts['C12'].sets['OneSet'])
    mdb.models['Model-1'].parts['C12'].assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        mdb.models['Model-1'].parts['C12'].sets['TwoSet'])
    mdb.models['Model-1'].parts['C12'].assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        mdb.models['Model-1'].parts['C12'].sets['ThreeSet'])
    mdb.models['Model-1'].parts['C12'].assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        mdb.models['Model-1'].parts['C12'].sets['FiveSet'])
    mdb.models['Model-1'].parts['C12'].assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        mdb.models['Model-1'].parts['C12'].sets['SixSet'])
    mdb.models['Model-1'].parts['C12'].assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 0.0, -1.0), region=
        mdb.models['Model-1'].parts['C12'].sets['SevenSet'])

    mdb.models['Model-1'].parts['C12'].assignBeamSectionOrientation(method=
        N1_COSINES, n1=(0.0, 1.0, 0.0), region=
        mdb.models['Model-1'].parts['C12'].sets['FourSet'])
    

        
    ### Assembly
    mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)

    mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='C12-1', part=
        mdb.models['Model-1'].parts['C12'])

    ### Step settings
    t_calc = (1.0/minLoadStep)
    mdb.models['Model-1'].StaticStep(initialInc=t_calc, maxInc=t_calc, maxNumInc=500, 
        name='Step-1', nlgeom=ON, previous='Initial')

    ### Boundary condition
   # onset assigned to fixed bottom
    mdb.models['Model-1'].rootAssembly.Set(
        name='FixSet',
        edges= mdb.models['Model-1'].rootAssembly.instances['C12-1'].sets['OneSet'].edges
    )
    # set seven assigned to move down for BC
    mdb.models['Model-1'].rootAssembly.Set(
        name='MoveDown',
        edges= mdb.models['Model-1'].rootAssembly.instances['C12-1'].sets['SevenSet'].edges
    )
    
    mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Step-1', 
        distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
        'FixBot', region=mdb.models['Model-1'].rootAssembly.sets['FixSet'], u1=0.0, 
        u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)
   
    mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Step-1', 
        distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
        'MoveDownTop', region=mdb.models['Model-1'].rootAssembly.sets['MoveDown'], 
        u1=0.0, u2=0.0, u3=-Uz_mm, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        

    ### Mesh 
    mdb.models['Model-1'].parts['C12'].seedPart(deviationFactor=0.1, minSizeFactor=
        0.1, size=meshsize)
    mdb.models['Model-1'].parts['C12'].generateMesh()
    mdb.models['Model-1'].parts['C12'].setElementType(elemTypes=(ElemType(
        elemCode=B31, elemLibrary=STANDARD), ), regions=(
        mdb.models['Model-1'].parts['C12'].edges.getSequenceFromMask((
        '[#ffffffff #f ]', ), ), ))
    mdb.models['Model-1'].rootAssembly.regenerate()

    ### create job
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
        memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF, 
        multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE, 
        numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=
        ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)# submit code	
    mdb.jobs[job_name].waitForCompletion()
    
    pass
    