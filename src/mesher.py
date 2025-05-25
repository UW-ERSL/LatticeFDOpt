import numpy as np

class MeshFrame():

    def refineFrameMesh(self,nodeXY, connectivity, numElemsPerBeam):
        numBeams = connectivity.shape[0]
        numnodeXY = nodeXY.shape[0]
        numNodesPerBeam = numElemsPerBeam+1;
        numNewNodesPerBeam = numElemsPerBeam-1
        numNodes = numnodeXY + numBeams*numNewNodesPerBeam;
        nodalCoords = np.zeros((numNodes,nodeXY.shape[1]));
        nodalCoords[0:numnodeXY,:] = nodeXY;
        conn = np.empty((0,2)); 
        t = np.linspace(0,1,numNodesPerBeam)
        for i in range (0,numBeams):
            
            startNode = nodalCoords[connectivity[i,0],:];
            endNode = nodalCoords[connectivity[i,1],:];
        
            xyz = np.outer((1-t),startNode) + np.outer(t,endNode);
            start = numnodeXY+numNewNodesPerBeam*(i)
            stop = numnodeXY+(i)*(numNewNodesPerBeam)+numNewNodesPerBeam
            newNodeNums = np.arange(start,stop);
            temp = np.repeat(newNodeNums,2);
            newConn = np.hstack((connectivity[i,0],temp,connectivity[i,1]));
            newConn = newConn.reshape((-1,2))
            nodalCoords[newNodeNums,:] = xyz[1:-1,:];
            conn = np.vstack((conn,newConn));
      
        return nodalCoords.astype(np.float64), conn.astype(int)
    
    def refineFrameMeshElemSize(self, nodeXY, connectivity, elemSize):
        numBeams = connectivity.shape[0]
        numnodeXY = nodeXY.shape[0]
        nodalCoords = np.copy(nodeXY)
        
        node1 = nodeXY[connectivity[:, 0]]
        node2 = nodeXY[connectivity[:, 1]]
        
        # Compute the Euclidean distance (L2 norm) between node1 and node2
        beamLength =  np.linalg.norm(node1 - node2, axis=1)                           
        
        numElemsPerBeam = np.rint(beamLength / np.array([elemSize])).astype(int)
        
        conn = np.zeros((np.sum(numElemsPerBeam),2))
        n = 0
        for i in range(numBeams):
            # Get start and end nodes for the beam
            startNode = nodalCoords[connectivity[i, 0], :]
            endNode = nodalCoords[connectivity[i, 1], :]
            
            # Determine the number of elements for this beam
            numNodesPerBeam = numElemsPerBeam[i] + 1
            numNewNodesPerBeam = numElemsPerBeam[i] - 1
            
            # Create a parameterized line for nodes along the beam
            t = np.linspace(0, 1, numNodesPerBeam)
            xyz = np.outer(1 - t, startNode) + np.outer(t, endNode)
            
            # Add new nodes to nodalCoords
            newNodeNums = np.arange(numnodeXY, numnodeXY + numNewNodesPerBeam)
            nodalCoords = np.vstack((nodalCoords, xyz[1:-1, :]))
            numnodeXY += numNewNodesPerBeam  # Update the total number of nodes
            
            temp = np.repeat(newNodeNums,2);
            newConn = np.hstack((connectivity[i,0],temp,connectivity[i,1]));
            newConn = newConn.reshape((-1,2))
            conn[n:n+newConn.shape[0],:] = newConn
            n += newConn.shape[0]
    
        return nodalCoords.astype(np.float64), conn.astype(int)
    
    def remeshFrame(self,nodeXY,connMat,elemsToBeDeleted):
    
        # Some of this should be available in self
    
        numNodes = np.max(connMat)
    
        connMat = np.delete(connMat, elemsToBeDeleted, 0)
        
        # next check for missing nodes if any
        
        nodeXYUnique = np.unique(connMat)
    
        reference_arr = np.arange(numNodes + 1)
    
        missingNodes =  np.setdiff1d(reference_arr, nodeXYUnique)
    
        nodeXY = np.delete(nodeXY,missingNodes,0)
        
        missingNodes = np.flip(missingNodes)

        for i in range(missingNodes.shape[0]):
    
            mask = connMat > missingNodes[i]
    
            connMat[mask] -= 1
    
        return nodeXY, connMat
  
    def nSideBase3Dunit(self,params=None,n_sides=8):
      if params is None:
        params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0} 
      
      def rotCord(theta, axis='x'):
        """
        This function takes an angle of rotation theta in radians and an optional axis of rotation, 
        and returns the corresponding rotation matrix.
      
        Args:
            theta: The angle of rotation in radians.
            axis: The axis of rotation. Defaults to 'x'.
      
        Returns:
            A rotation matrix.
        """
        if axis == 'x':
          return [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
        elif axis == 'y':
          return [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        elif axis == 'z':
          return [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        else:
          raise ValueError("Invalid axis")
          
      
      B = params['base']
      H = params['height']
      
    #   n_sides = 6
    #   print(n_sides)
      beta =1.;
      if n_sides== 4 or n_sides == 6 or n_sides == 10:
        beta = 0. #skip rotate by half angle
      # Calculate the radius R using the generalized formula
    #   R = B / (2 * np.sin(np.pi / n_sides))
      R = B/np.sqrt(2)
  
      # Angle between the vertices
      angle = np.linspace(0, 2 * np.pi, n_sides + 1)
      angle = angle - beta*angle[1]/2 # rotate the octagon by half the distance
  
      # X and Y coordinates of the polygon
      x = R * np.cos(angle[0:-1])
      y = R * np.sin(angle[0:-1])
      
      numVertices = int(n_sides*6)
      vertices = np.zeros((numVertices, 3),dtype=np.float64)
        
      nodeNumbers = np.arange(0,numVertices)
      
      vertices[0:n_sides,0] = x
      vertices[0:n_sides,1] = y
      vertices[0:n_sides,2] = -H/2
      
      vertices[n_sides:int(n_sides*2),:] = (rotCord(np.pi, axis='y')@vertices[0:n_sides,:].T).T
      vertices[int(n_sides*2):int(n_sides*3),:] = (rotCord(np.pi/2, axis='y')@vertices[0:n_sides,:].T).T
      vertices[int(n_sides*3):int(n_sides*4),:] = (rotCord(-np.pi/2, axis='y')@vertices[0:n_sides,:].T).T
      
      vertices[int(n_sides*4):int(n_sides*5),:] = (rotCord(np.pi/2, axis='x')@vertices[0:n_sides,:].T).T
      vertices[int(n_sides*5)::,:] = (rotCord(-np.pi/2, axis='x')@vertices[0:n_sides,:].T).T
      
      vertices = vertices - np.min(vertices,axis=0)
      
      connMatbase = (np.concatenate((np.arange(n_sides),np.arange(1,n_sides+1))).reshape((-1,n_sides)).T)

      connMatbase[connMatbase == n_sides] = 0
      
      connMat = connMatbase
      # connectivity generation of each of 6 polygons
      for i in range(5):
          connMat = np.concatenate((connMat,connMatbase+n_sides*(i+1)),dtype=int)
      
      # lets find nodes closest to top and bot panels
      # helps find the connections between the polygons
      def find_close_pairs(vertices,arr1,arr2):
        one_octagon = vertices[arr1, :]
        rest_octagons = vertices[arr2, :]
        distances = np.linalg.norm((one_octagon - rest_octagons[:,np.newaxis,:]),axis=2)
        min_distances = np.round(np.min(distances, axis=0),decimals=4)
        min_row_indices = np.argmin(distances, axis=0)
        # print(min_row_indices)
        has_repeats = len(min_row_indices) != len(np.unique(min_row_indices))
        if has_repeats:
          v = np.where(min_distances == np.min(min_distances))[0]
        else:
          v = np.arange(0,len(min_row_indices))
        # print(min_distances)
        # print(v)
        conn = np.concatenate((arr1[v],min_row_indices[v]+arr2[0])).reshape((-1,arr1[v].shape[0])).T
        return conn
      
      for i in range(4): # top, bot, left and right side connections
        conn = find_close_pairs(vertices,np.arange(int(n_sides*i),int(n_sides*(i+1))),np.arange(int(n_sides*(i+1)),int(n_sides*6)))
        connMat = np.concatenate((connMat,conn),dtype=int)      
            
      return vertices, connMat
      
    
    def combined3Dunit(self,params=None):
        if params is None:
          params = {'base': 5.,'height':5., 'theta': 0, 'alpha': 0,'epsValue':0,'Name':'DOV'} 
        vertices_list = []
        connMat_list = []
        # for i in range(len(params['Name'])): #
        if params['Name'][0] == 'C':
            params['base'] = params['base']/2
            vertices,connMat = self.nSideBase3Dunit(params,np.array(params['Name'][1:]).astype(int))
            params['base'] = params['base']*2
        
        vertices_list.append(vertices)
        connMat_list.append(connMat)
        
        all_vertices, all_connMat = self.merge_vertices_and_conn(vertices_list, connMat_list)
        # print(all_connMat)
        all_vertices, all_connMat = self.find_and_divide_intersections(all_vertices, all_connMat)
      
        # all_vertices, all_connMat = self.refineFrameMesh(all_vertices, all_connMat, 1)
        return all_vertices, all_connMat
    
    def generateCombined3DLattice(self,params): # any shape in either hexagon or square
        if params is None:
            params = {'base': 2.,'height':5., 'theta': 0, 'alpha':0,'epsValue':0.,'phi':0.0,'delta':0, 'nx': 3, 'ny':3, 'nz':1,'Name':'V','Shape':'Square'}
    
        vertices, connMat = self.combined3Dunit(params) # generates a unit cell which we repeat
        nx = params['nx']  # Number of repetitions along the X axis
        ny = params['ny']  # Number of repetitions along the Y axis
        nz = params['nz']  # Number of repetitions along the Z axis
    
        all_vertices = vertices.copy()
        all_connMat = connMat.copy()
        all_radii_indices = np.arange(vertices.shape[0])  # Start with the indices of the nodal radii of the unit cell
        temp_radii_indices = all_radii_indices.copy()  # Indices for nodal radii from the unit cell
    
        unit_cell_size = connMat.shape[0]  # Store the size of the unit cell's connectivity matrix
        Mov1 = vertices.copy()
        Mov1[:, [0, 1]] += -params['base'] / 2
        
        # Index array that maps each connection in the full lattice to the corresponding connection in the unit cell
        conn_index_array = np.arange(unit_cell_size)  # Initial index array for the first unit cell
        all_conn_indices = conn_index_array.copy()
    
        for k in range(nz):
            for j in range(ny):
                if params['Shape'] == 'Hexagon':  # Add extra lattice unit per side
                    if (j < 1) or j >= ny - 1:
                        etr = 0
                    elif j < 3 or j >= ny - 3:
                        etr = 1  # This should be 1 for panther project, 0 for other reasons
                    elif j < 5 or j >= ny - 5:
                        etr = 2
                    else:
                        etr = 3
                else:
                    etr = 0
    
                for i in range(nx + int(2 * etr)):
                    angRad = np.deg2rad(params['phi'] * (i + j))
                    c = np.cos(angRad)
                    s = np.sin(angRad)
                    RotM = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
                    Mov2 = (RotM @ Mov1.T).T
                    Mov2[:, [0, 1]] += params['base'] / 2
    
                    translated = Mov2
                    translated[:, 0] += i * (params['base'] + params['delta']) - min(j, etr) * params['base']
                    translated[:, 1] += j * (params['base'] + params['delta'])
                    translated[:, 2] += k * (params['height'])
    
                    if i + j + k != 0:
                        connMat_temp = -connMat
                        current_indices = conn_index_array.copy()  # Create a copy of the index array for this repetition
                        
                        for vertex, node_num in zip(translated, range(translated.shape[0])):
                            vertice_loc = np.argwhere((np.sum(np.isclose(vertex, all_vertices), axis=1) == all_vertices.shape[1]) == 1)
                            vertice_size = vertice_loc.size
                            connMat_loc = np.argwhere(connMat_temp == -node_num)
    
                            if vertice_size == 0:  # New vertex
                                all_vertices = np.append(all_vertices, vertex.reshape((1, -1)), axis=0)
                                all_radii_indices = np.append(all_radii_indices, temp_radii_indices[node_num])  # Append the corresponding radius index
                                connMat_temp[connMat_loc[:, 0], connMat_loc[:, 1]] = all_vertices.shape[0] - 1
                            else:  # Repeated vertex
                                merged_node = vertice_loc[0, 0]
                                connMat_temp[connMat_loc[:, 0], connMat_loc[:, 1]] = merged_node
                                
                                # No need to average radii indices; just keep the first occurrence
    
                        sumTemp = np.sum(connMat_temp, axis=1)
                        allSum = np.sum(all_connMat, axis=1)
    
                        checkPoints = np.where(1.0 * (sumTemp == allSum.reshape(-1, 1)) == 1.0)
                        removeElem = []
    
                        for cp in range(len(checkPoints[0])):
                            cond1 = connMat_temp[checkPoints[1][cp], :] == all_connMat[checkPoints[0][cp], :]
                            cond2 = connMat_temp[checkPoints[1][cp], :] == np.flip(all_connMat[checkPoints[0][cp], :])
    
                            if np.sum(cond1) == 2 or np.sum(cond2) == 2:
                                removeElem.append(cp)
                        
                        connMat_temp = np.delete(connMat_temp, checkPoints[1][removeElem], axis=0)
                        current_indices = np.delete(current_indices, checkPoints[1][removeElem], axis=0)
                        
                        all_connMat = np.concatenate((all_connMat, connMat_temp))
                        all_conn_indices = np.concatenate((all_conn_indices, current_indices))  # Append the current index array
    
        all_vertices = all_vertices - np.min(all_vertices, axis=0)
    
        return all_vertices, all_connMat, all_conn_indices
        
    
    
    def merge_vertices_and_conn(self,vertices_list, connMat_list):
        all_vertices = np.vstack(vertices_list)
            
        # Find the unique vertices and map the original indices to new indices
        unique_vertices, unique_indices = np.unique(all_vertices.round(decimals=10), axis=0, return_inverse=True)
        
        if len(unique_vertices) == len(all_vertices):
            updated_connMats = []
            offset = 0
            for connMat, vertices in zip(connMat_list, vertices_list):
                updated_connMat = connMat + offset
                updated_connMats.append(updated_connMat)
                offset += len(vertices)
            
            # Combine the updated connectivity matrices
            all_connMat = np.vstack(updated_connMats)
            
        else:    
            all_vertices = unique_vertices 
            # Update the connectivity matrices with new indices
            updated_connMats = []
            offset = 0
            # print("Con", connMat_list)
            for connMat, vertices in zip(connMat_list,vertices_list):
                updated_connMat = unique_indices[offset:offset + len(vertices)][connMat]
                updated_connMats.append(updated_connMat)
                offset += len(vertices)
                # print("This=",updated_connMat)
                # print("HERE")

            
            # Combine the updated connectivity matrices
            combined_connMat = np.vstack(updated_connMats)
            
            # Remove duplicate connections
            all_connMat = np.unique(combined_connMat, axis=0)
            
            # Remove self-connections (rows where the two values are the same)
            all_connMat = all_connMat[all_connMat[:, 0] != all_connMat[:, 1]]
            
            # print(all_connMat)
            # import sys;sys.exit()

        return all_vertices,all_connMat

    def intersect_lines(self,P1, P2, P3, P4):
        # Calculate direction vectors
        d4321 = np.dot(P4 - P3, P2 - P1)
        d2121 = np.dot(P2 - P1, P2 - P1)
        d4343 = np.dot(P4 - P3, P4 - P3)
        d1343 = np.dot(P1 - P3, P4 - P3)
        d1321 = np.dot(P1 - P3, P2 - P1)
        
        denom = d2121 * d4343 - d4321 * d4321
        
        if np.abs(denom) < 1e-7:
            # Lines are parallel or coincident
            is_intersecting = False
            Pa = np.array([])
            Pb = np.array([])
        else:
            mua = (d1343 * d4321 - d1321 * d4343) / denom
            mub = (d1343 + mua * d4321) / d4343
            
            # Calculate intersection points
            Pa = P1 + mua * (P2 - P1)
            Pb = P3 + mub * (P4 - P3)
            
            # Check if the intersection points are the same and within segments
            if np.linalg.norm(Pa - Pb) < 1e-7:
                if (0 <= mua <= 1) and (0 <= mub <= 1):
                    # Check that intersection is not at endpoints
                    if not np.any(np.all(np.isclose(Pa, np.array([P1, P2, P3, P4]), atol=1e-7), axis=1)):
                        is_intersecting = True
                    else:
                        is_intersecting = False
                else:
                    is_intersecting = False
            else:
                is_intersecting = False
        
        return Pa, Pb, is_intersecting
    
    # Function to find the intersection and divide connections
    def find_and_divide_intersections(self,vertices, connMat):
        new_vertices = vertices.tolist()
        new_connMat = connMat.tolist()
        intersections = {}
    
        for i in range(len(connMat)):
            for j in range(i + 1, len(connMat)):
                v1, v2 = connMat[i]
                v3, v4 = connMat[j]
    
                # Extract coordinates of the vertices
                p1, p2 = vertices[v1], vertices[v2]
                p3, p4 = vertices[v3], vertices[v4]
    
                # Find the intersection point
                # if p1==p3 or p1==p4 or p2==p3 or p2==p4:
                #   is_intersecting=False
                # else:
                Pa, Pb, is_intersecting  = self.intersect_lines(p1, p2, p3, p4)
                
                # if i==1 and j==9:
                #   print(p1,p2,p3,p4)
                #   print(Pa)
                #   import sys;
                #   sys.exit()
    
                if is_intersecting:
                    intersection_point = tuple(Pa)
                    
                    if intersection_point not in intersections:
                        intersections[intersection_point] = len(new_vertices)
                        new_vertices.append(intersection_point)
    
                    intersection_idx = intersections[intersection_point]
                    
                    new_connMat.append([v1, intersection_idx])
                    new_connMat.append([intersection_idx, v2])
                    new_connMat.append([v3, intersection_idx])
                    new_connMat.append([intersection_idx, v4])
                    new_connMat.remove([v1,v2])
                    new_connMat.remove([v3,v4])
    
        return np.array(new_vertices), np.array(new_connMat)
        
