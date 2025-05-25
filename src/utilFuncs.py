import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from itertools import chain
import math
import os
import sys
from mmapy import kktcheck,mmasub
from scipy.optimize import Bounds # minimize is used for testing
from scipy.interpolate import BSpline
from types import SimpleNamespace
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import re
from matplotlib.gridspec import GridSpec
from pyDOE2 import lhs

warnings.filterwarnings("ignore", category=RuntimeWarning)

# sys.path.append(os.path.realpath('./src/'))
#--------------------------#
def set_seed(manualSeed):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(manualSeed)
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  np.random.seed(manualSeed)
  random.seed(manualSeed)
#--------------------------#
def curve_setup(cp, t, curve='points'):
    """
    Calculates points on a Bézier or Spline curve of arbitrary degree.

    Args:
        cp: A tensor of control points (shape (n+1, d)), where n is the 
            degree of the curve and d is the dimension of the points 
            (e.g., d=1 for a 2D curve represented as (x,y) pairs, d=1 for your case).
        t: A tensor of parameter values (shape (m,)), where m is the number 
           of points to calculate on the curve.  t should be in the range [0, 1].

    Returns:
        A tensor of points on the Bézier curve (shape (m, d)).
    """
    if curve == 'points':
        curve_points = cp.clone().T
    elif curve == 'spline':
        p = 2 # degree of the curve
        n = cp.shape[1] # number of control points
        U = np.zeros(p+n+1)
        U[p:n+1] = np.linspace(0,1,n-p+1)
        U[-p::] = 1
        N = np.zeros((n,t.shape[0]))
        I =  np.eye(n)
        for i in range(n):
            N[i,:] = BSpline(U,I[i],p)(t.cpu().detach().numpy())
        N = torch.from_numpy(N).float().to(cp.device)
        curve_points = N.T @ cp.T
    elif curve == 'bezier':
        n = cp.shape[1] - 1  # Degree of the Bézier curve
        m = t.shape[0]  # Number of points to calculate
        if n == 4:
            b = torch.tensor([1,4,6,4,1.]).to(cp.device)
        elif n == 5:
            b = torch.tensor([1,5,10,10,5,1]).to(cp.device)

        # Calculate Bernstein basis polynomials
        bernstein_matrix = torch.zeros((m, n + 1)).to(cp.device)
        for i in range(n + 1):
            bernstein_matrix[:, i] =  b[i]* (t ** i) * ((1 - t) ** (n - i))
        
        # Calculate points on the Bézier curve
        curve_points = bernstein_matrix @ cp.T

    return curve_points.T
#--------------------------#
# # Neural network
class NeuralNet(nn.Module):
    def __init__(self, nnSettings, useCPU=False):  # Add device parameter
        super().__init__()
        manualSeed = 96
        set_seed(manualSeed)
        
        # useCPU = True
        if(torch.cuda.is_available() and (useCPU == False) ):
          self.device = torch.device("cuda:0")
          print("Running on GPU")
        else:
          self.device = torch.device("cpu")
          torch.set_num_threads(18)  
          print("Running on CPU\n")
          print("Number of CPU threads PyTorch is using:", torch.get_num_threads())
        
        self.layers = nn.ModuleList()
        self.bnLayer = nn.ModuleList()
        self.dropout = nn.ModuleList() # List of dropouts

        self.criterion = nn.MSELoss(reduction='mean')
        # self.criterion = nn.HuberLoss(reduction='mean',delta=1.0)
        self.criterion2 = nn.KLDivLoss(reduction='mean')
        
        self.nnSettings = nnSettings
        input_dim = nnSettings['inputDim']
        base_neurons = nnSettings['numNeuronsPerLyr']
        num_layers = nnSettings['numLayers']
        output_dim = nnSettings['outputDim']
        bottleneck_factor = nnSettings.get('bottleneck_factor', 1.0) # Get bottleneck_factor, default to 1.0

        self.curve = nnSettings['curve']
        self.dropout_p = nnSettings['dropout'] # Dropout probability
        
        # Calculate bottleneck layer sizes
        layer_sizes = self.calculate_layer_sizes(base_neurons, num_layers, bottleneck_factor)

        # Input layer
        self.layers.append(nn.Linear(input_dim, layer_sizes[0]))
        self.bnLayer.append(nn.BatchNorm1d(layer_sizes[0]))
        self.dropout.append(nn.Dropout(self.dropout_p))

        # Hidden layers
        for i in range(num_layers - 1):
            l = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            self.bnLayer.append(nn.BatchNorm1d(layer_sizes[i+1]))
            self.dropout.append(nn.Dropout(self.dropout_p))

        # Output layer
        self.layers.append(nn.Linear(layer_sizes[-1], output_dim))
    
        # Calculate total weights and biases
        self.total_weights, self.total_biases = self.calculate_total_params()
        self.activation = nn.LeakyReLU()

        self.to(self.device) # Move the entire model to the specified device
    
    def calculate_total_params(self):
        """Calculates total weights and biases in the network."""
        total_weights = 0
        total_biases = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                total_weights += layer.weight.numel()
                total_biases += layer.bias.numel()
        return total_weights, total_biases
        
    def calculate_layer_sizes(self, base_neurons, num_layers, bottleneck_factor):
        """Calculates layer sizes based on bottleneck factor."""
        if bottleneck_factor == 1.0: # No bottleneck
            return [base_neurons] * num_layers

        # Example Bottleneck shape, you can change this part to get other shapes.
        middle_index = num_layers // 2
        layer_sizes = []
        for i in range(num_layers):
            if i < middle_index:
                current_neurons = int(base_neurons - (base_neurons - base_neurons*bottleneck_factor) * (i / middle_index))
            elif i == middle_index:
                current_neurons = int(base_neurons * bottleneck_factor)
            else:
                current_neurons = int(base_neurons - (base_neurons - base_neurons*bottleneck_factor) * ((num_layers-1-i) / middle_index))
            layer_sizes.append(current_neurons)

        return layer_sizes
       
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.bnLayer[i](self.layers[i](x)))
            x = self.dropout[i](x) # Apply dropout after activation and batchnorm

        x = self.layers[-1](x)
        return x
    
    def loss_fn(self,x,y,loss):
        c = 0.001
        loss += self.criterion(x, y) 
        
        # Calculate L1 regularization term
        l1_penalty = 0.0
        l2_penalty = 0.0
        for param in self.parameters():
            if param.ndim > 1:  # Exclude bias terms
                l1_penalty += torch.norm(param, p=1)
                l2_penalty += torch.norm(param, p=2)
        # loss += c * l1_penalty + c * l2_penalty
        
        return loss
        
    
    def train_model(self, train_loader, test_loader, num_epochs=10, lr=0.0001, tol=1e-3,prntNum=1):
    
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)  # T_max is usually num_epochs
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        epoch = 0.0
        self.trainlossArr = []
        self.vallossArr = []
        self.Narr = []
        torch.autograd.set_detect_anomaly(True)
        P = 2.0
        loss = torch.zeros(1).to(self.device)
        change = 1.0
        pbar_epochs = tqdm(desc="Training", dynamic_ncols=True) #remove total, and use dynamic_ncols.
        loss_last = loss.clone()
        loss0 = torch.ones(1).to(self.device)
        for epoch in range(num_epochs):
            loss = torch.zeros(1).to(self.device)
            all_absSum_train=[]
            all_absSum_test=[]
            for batch in train_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device).float()  # Move inputs to device
                targets = targets.to(self.device).float() # Move targets to device. 
                self.Npt = targets.shape[1]
                mask = inputs[:,self.nnSettings['inputDim']::]
                optimizer.zero_grad()
                outputs = self.forward(inputs[:,0:self.nnSettings['inputDim']])
                
                outputs2 = curve_setup(outputs,t=torch.linspace(0,1,self.Npt).to(self.device),curve=self.curve)
                   
                loss += self.loss_fn(outputs2*mask,targets*mask,loss)
                
                all_absSum_train.append(torch.abs(outputs2 - targets).mean(dim=1).detach())
            loss = loss/loss0
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            
            scheduler.step() 
            
            epoch += 1
            
            
            if ((epoch) % int(prntNum)== 0 or epoch == 1):
                self.trainlossArr.append(loss.cpu().detach().numpy().item())
                self.Narr.append(epoch-1)
                with torch.no_grad():
                    lossV = 0.0
                    for inputs, targets in test_loader:
                        inputs = inputs.to(self.device).float()  # Move inputs to device
                        targets = targets.to(self.device).float() # Move targets to device
                        self.Npt = targets.shape[1]
                        mask = inputs[:,self.nnSettings['inputDim']::]
    
                        outputs = self.forward(inputs[:,0:self.nnSettings['inputDim']])
                        
                        outputs2 = curve_setup(outputs,t=torch.linspace(0,1,self.Npt).to(self.device),curve=self.curve)
                        
                        lossV += self.loss_fn(outputs2*mask,targets*mask,lossV)
                    
                        all_absSum_test.append(torch.abs(outputs2 - targets).mean(dim=1).detach())
                    lossV = lossV/loss0
    
                    self.vallossArr.append(lossV.cpu().detach().numpy().item())
            
                maxV = torch.max(torch.cat(all_absSum_train))
                maxV2 = torch.max(torch.cat(all_absSum_test))
                
                print(f"Epoch [{epoch}/{num_epochs}], Training loss: {loss.item():.4f} , Validation loss: {lossV.item():.4f},\n \
                Max absolute difference on Train: {maxV.item():.4f}, and Test: {maxV2.item():.4f}")
            
            if epoch > num_epochs:
                break
            pbar_epochs.update(1)

        pbar_epochs.close() #close the progress bar after the loop ends.
            
        iter_lossPlot(np.array(self.trainlossArr).reshape(-1),np.array(self.vallossArr).reshape(-1),np.array(self.Narr).reshape(-1))
        
    
    def evaluate_NN(self, test_loader):
        self.eval()
        all_targets_np = []
        all_outputs_np = []
        all_absSum_data = []
        with torch.no_grad():
            loss = 0.0
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()
                mask = inputs[:, self.nnSettings['inputDim']::]
    
                outputs = self.forward(inputs[:, 0:self.nnSettings['inputDim']])
                outputs2 = curve_setup(outputs, t=torch.linspace(0, 1, targets.shape[1]).to(self.device), curve=self.curve)
    
                masked_outputs = (outputs2 * mask).cpu().numpy()
                masked_targets = (targets * mask).cpu().numpy()
    
                loss += self.loss_fn(outputs2 * mask, targets * mask, loss)
                all_absSum_data.append(torch.abs(outputs2 * mask - targets * mask).mean(dim=1))
    
                # Append the masked numpy arrays
                all_targets_np.append(masked_targets)
                all_outputs_np.append(masked_outputs)
    
        # Concatenate all predictions and targets
        all_targets_np = np.concatenate(all_targets_np, axis=0)
        all_outputs_np = np.concatenate(all_outputs_np, axis=0)
    
        # Calculate MAE using scikit-learn
        avg_mae = mean_absolute_error(all_targets_np, all_outputs_np)
    
        # Calculate RMSE using scikit-learn
        avg_rmse = root_mean_squared_error(all_targets_np, all_outputs_np)
    
        # Calculate R² score using scikit-learn
        r2_scores = r2_score(all_targets_np, all_outputs_np, multioutput='raw_values')
        avg_r2 = np.mean(r2_scores)
    
        maxV_index = torch.topk(torch.cat(all_absSum_data).to(self.device), k=5, dim=0)
        minV_index = torch.topk(torch.cat(all_absSum_data).to(self.device), k=5, dim=0, largest=False)
    
        return avg_mae, avg_rmse, avg_r2, maxV_index, minV_index
        
    def predict_out(self, input_data,t=None):
        if t is None:
            t=torch.linspace(0,1,self.Npt)
            
        self.eval()
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float().to(self.device)  # Move to device
            elif isinstance(input_data, torch.Tensor):
                input_tensor = input_data.float().to(self.device)  # Move to device
            else:
                raise TypeError("Input data must be a NumPy array or PyTorch tensor.")
            
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t).float().to(self.device)  # Move to device
            elif isinstance(t, torch.Tensor):
                t = t.float().to(self.device)  # Move to device
            else:
                raise TypeError("Input data t must be a NumPy array or PyTorch tensor.")
            
            output_tensor2 = self.forward(input_tensor[:,0:self.nnSettings['inputDim']])
            output_tensor = curve_setup(output_tensor2,t,curve=self.curve)

            predictions = output_tensor.cpu().numpy() if isinstance(input_data, np.ndarray) else output_tensor.cpu().numpy() # Move to CPU for numpy conversion
            cp =  output_tensor2.cpu().numpy() if isinstance(input_data, np.ndarray) else output_tensor2.cpu().numpy() # Move to CPU for numpy conversion
         
            return predictions, cp

#--------------------------#
# plotting functions
def iter_lossPlot(TrainLoss,valLoss,N):
    
    plt.figure(figsize=(8, 6))
    plt.plot(N,TrainLoss,'r-o',label='Training loss',markersize=8)
    plt.plot(N,valLoss,'b-s',label='Validation loss',markersize=8)
    # # Labels
    plt.ylabel('Loss',fontsize=16)
    plt.xlabel('Iteration',fontsize=16)
    
    # Title
    titleStr = f"Training Loss {TrainLoss[-1]:.2e}\nValidation Loss {valLoss[-1]:.2e}"
    plt.title(titleStr, fontsize=18)
    plt.legend(fontsize=16)
    plt.show()
#--------------------------#
def extract_ABQ_values(filename,uinput):
    # uinput is where we want to find the force at 
    # extract the force displacement from abaqus files and write them to a file after iterpolations
    # Define regex pattern
    pattern = re.compile(
        r"res\s*=\s*\(\s*array\s*\(\s*\[\s*([-+0-9eE.,\s]+?)\s*\]\s*\)\s*,\s*"  # NumPy array (group 1)
        r"([-+]?\d+)\s*,\s*"  # First integer (group 2)
        r"(\d+)\s*,\s*"  # Second integer (group 3)
        r"(\d+)\s*,\s*"  # Third integer (group 4)
        r"tensor\s*\(\s*\[\s*([-+0-9eE.,\s]+?)\s*\]\s*\)\s*,\s*"  # First tensor values (group 5)
        r"tensor\s*\(\s*\[\s*([-+0-9eE.,\s]+?)\s*\]\s*\)\s*,\s*"  # Second tensor values (group 6)
        r"tensor\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)\s*,\s*" # Third tensor float (group 7)
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"  # Third float (group 8)
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"  # Fourth float (group 9)
        )

    # Read the file
    with open(filename, "r") as file:
        content = file.read()

    # Find all matches
    matches = pattern.findall(content)

    # Initialize empty lists for outputs
    all_arrays = []
    tensor1_values = []
    tensor2_values = []
    metadata_values = []
    tensor3_values = []

    for match in matches:
        arr_str, float1, int1, int2, tensor1_str, tensor2_str, tensor3, float2, float3_time = match

        # Convert array string to a NumPy array (handling extra spaces and empty values)
        arr_values = np.array([float(x) for x in arr_str.replace(" ", "").split(",") if x], dtype=float)
        all_arrays.append(arr_values)

        # Convert metadata values to integer format
        metadata_values.append([int(float(float1))])
        # metadata_values.append([int(float(float1)), int(int1), int(int2), int(float(float2))])

        Uar = np.array([float(x) for x in tensor1_str.replace(" ", "").split(",") if x], dtype=float)
        Far = np.array([float(x) for x in tensor2_str.replace(" ", "").split(",") if x], dtype=float)
        Utensor = torch.from_numpy(Uar)
        Ftensor = torch.from_numpy(Far)
        if Ftensor[-1]>1e5 or Ftensor[-1]<-1e5:
            Ftensor = Ftensor[:-1]
            Utensor = Utensor[:-1]
        F, U = linear_interpolation_torch(Utensor,Ftensor,uinput)
        # Convert tensor values into lists of floats
        tensor1_values.append(U.detach().numpy())
        tensor2_values.append(F.detach().numpy())
        tensor3_values.append(tensor3)
        
    # Convert metadata to a NumPy array with integer dtype
    metadata_values = np.array(metadata_values, dtype=int).reshape(-1)
    all_arrays = np.array(all_arrays)  # Use dtype=object for variable-length arrays
    Uall = np.array(tensor1_values, dtype=float)
    Fall = np.array(tensor2_values, dtype=float)
    return all_arrays, Uall, Fall, metadata_values

#--------------------------#
# Linear interpolation function with boundary handling
def linear_interpolation_torch(x_known, y_known, x_in):
    y_out = torch.zeros_like(x_in)
    x_out = torch.zeros_like(x_in)
    for i in range(len(x_in)):
        if x_in[i] < x_known[0]:
            y_out[i] = y_known[0]
            x_out[i] = x_known[0]
        elif x_in[i] > x_known[-1]:
            y_out[i] = y_known[-1]
            x_out[i] = x_known[-1]
        else:
            for j in range(len(x_known) - 1):
                if x_known[j] <= x_in[i] <= x_known[j + 1]:
                    # Perform linear interpolation
                    t = (x_in[i] - x_known[j]) / (x_known[j + 1] - x_known[j])
                    y_out[i] = y_known[j] + t * (y_known[j + 1] - y_known[j])
                    x_out[i] = x_in[i]
                    break
    return y_out, x_out    

#--------------------------#

def dataGen(p,N,output_filename,seedNum=0):
    num_variables = p # number of design variables
    num_samples =  N
    if p == 2:
        intervals_I = np.linspace(0.0, 1.0, np.sqrt(num_samples).astype(int))
        # Create mesh grid
        I,J = np.meshgrid(intervals_I,intervals_I,indexing='ij')
        samples = np.stack((I.reshape(-1),J.reshape(-1)),axis=1)
        seedNum = 0
    elif p>2:
        # Generate Latin Hypercube samples directly in the range [0, 1]
        samples = np.array(lhs(num_variables, num_samples,random_state=seedNum))
        
    with open(output_filename, 'w') as f:
        # Write the header information as a comment
        f.write(f"# Seed: {seedNum}, Variables: {num_variables}, Samples: {num_samples}\n")
        f.write(f"# Each row represents a sample in the format [val1 ... valp]\n")

        # Write the samples data row by row in the desired format
        for sample_row in samples:
            # Format each float value to a specific number of decimal places and join with a space
            # Then wrap the entire string in square brackets
            formatted_row = " ".join([f"{val:.16f}" for val in sample_row]) # Using .16f for high precision
            f.write(f"[{formatted_row}]\n")
            
            
def writeOFFfile(vertices,connectivity,fname):
    # scriptDir = os.path.dirname(os.path.abspath(__file__))
    # print(scriptDir)
    # fname = os.path.join(scriptDir,fileName)
    
    with open (fname,'w') as file:
        file.write("OFF\n")
        file.write(f"{vertices.shape[0]}  0 {connectivity.shape[0]}\n")
        np.savetxt(file,vertices,fmt='%f',delimiter=' ')
        np.savetxt(file,connectivity,fmt='%d',delimiter=' ')

def saveIGS(pathFile,nodeXY,conectivity):
    fname = pathFile+'.off'
    print(fname)
    writeOFFfile(nodeXY,conectivity,fname)
    off2igs(fname)
    print(f"Filed saved to:{fname}")

def hollerith(s):
    return "{}H{}".format(len(s), s)


class Iges:

    def __init__(self):
        self.buffer = { 'D':"", 'P':"" }
        self.lineno= { 'D':0, 'P':0 }

    def add_line(self, section, line, index=""):
        index = str(index)
        self.lineno[section] += 1
        lineno = self.lineno[section]
        buf = "{:64s}{:>8s}{}{:7d}\n".format(line, index, section, lineno)
        self.buffer[section] += buf

    def update(self, section, params, index=""):
        line = None
        for s in params:
            s = str(s)
            if line is None:
                line = s
            elif len(line + s) + 1 < 64:
                line += "," + s
            else:
                self.add_line(section, line + ',', index=index)
                line = s
        self.add_line(section, line + ';', index=index)

    def start_section(self, comment=""):
        self.buffer["S"] = ""
        self.lineno["S"] = 0
        self.update("S", [comment])

    def global_section(self, filename=""):
        self.buffer["G"] = ""
        self.lineno["G"] = 0
        self.update("G", [
            "1H,",       # 1  parameter delimiter 
            "1H;",       # 2  record delimiter 
            "6HNoname",  # 3  product id of sending system
            hollerith(filename),  # 4  file name
            "6HNoname",  # 5  native system id
            "6HNoname",  # 6  preprocessor system  
            "32",        # 7  binary bits for integer
            "38",        # 8  max power represented by float
            "6",         # 9  number of significant digits in float
            "308",       # 10 max power represented in double
            "15",        # 11 number of significant digits in double
            "6HNoname",  # 12 product id of receiving system
            "1.00",      # 13 model space scale
            "6",         # 14 units flag (2=mm, 6=m)
            "1HM",       # 15 units name (2HMM)
            "1",         # 16 number of line weight graduations
            "1.00",      # 17 width of max line weight
            "15H20181210.181412",  # 18 file generation time
            "1.0e-006",  # 19 min resolution
            "0.00",      # 20 max coordinate value
            "6HNoname",  # 21 author
            "6HNoname",  # 22 organization
            "11",        # 23 specification version
            "0",         # 24 drafting standard
            "15H20181210.181412",  # 25 time model was created
        ])

    def entity(self, code, params, label="", child=False):
        code = str(code)
        status = "00010001" if child else "1"
        dline = self.lineno["D"] + 1
        pline = self.lineno["P"] + 1
        self.buffer['D'] += (
            "{:>8s}{:8d}{:8d}{:8d}{:8d}{:8d}{:8d}{:8d}{:>8s}D{:7d}\n".format(
            code, pline, 0, 0, 0, 0, 0, 0, status, dline) +
            "{:>8s}{:8d}{:8d}{:8d}{:8d}{:8d}{:8d}{:8s}{:8d}D{:7d}\n".format(
            code, 1, 0, 1, 0, 0, 0, label, 0, dline + 1))
        self.update("P", [code] + list(params), index=dline)
        self.lineno["D"] = dline + 1
        return dline

    def pos(self, pt, origin):
        x, y, z = origin
        return (pt[0] + x, pt[1] + y, + pt[2] + z)

    def origin(self, size, origin, centerx=False, centery=False):
        w, h = size
        x, y, z = origin
        if centerx: x -= w / 2
        if centery: y -= h / 2
        return x, y, z

    def mapping(self, points, origin):
        start = points[-1]
        refs = []
        for p in points:
            refs.append(self.line(start, p, origin, child=True))
            start = p
        return self.entity(102, [len(refs)] + refs, child=True) 

    def surface(self, directrix, vector, points, origin):
        surface = self.entity(122, [directrix] + list(vector), child=True)
        mapping = self.mapping(points, origin)
        curve = self.entity(142, [1, surface, 0, mapping, 2], child=True)
        self.entity(144, [surface, 1, 0, curve])

    def cylinder(self, directrix, vector, origin):
        self.entity(120, [directrix, vector, 0, 2 * math.pi])

    ################

    def write(self, filename=None):
        self.start_section()
        self.global_section(filename)
        f = open(filename, "w") if filename else sys.stdout
        f.write(self.buffer['S'])
        f.write(self.buffer['G'])
        f.write(self.buffer['D'])
        f.write(self.buffer['P'])
        f.write("S{:7d}G{:7d}D{:7d}P{:7d}{:40s}T{:7d}\n".format(
            self.lineno['S'], self.lineno['G'], 
            self.lineno['D'], self.lineno['P'], "", 1))
        if filename: f.close()

    def line(self, start, end, origin=(0,0,0), child=False):
        start = self.pos(start, origin)
        end = self.pos(end, origin)
        return self.entity(110, start + end, child=child)

    def xzplane(self, size, origin=(0,0,0)):
        w, h = size
        x, y, z = origin
        points = [(w, 0, 0), (w, 0, h), (0, 0, h), (0, 0, 0)]
        directrix = self.line((0, 0, 0), (w, 0, 0), origin, child=True)
        self.surface(directrix, (x, y, z + h), points, origin)

    def yzplane(self, size, origin=(0,0,0)):
        w, h = size
        x, y, z = origin
        points = [(0, w, 0), (0, w, h), (0, 0, h), (0, 0, 0)]
        directrix = self.line((0, 0, 0), (0, w, 0), origin, child=True)
        self.surface(directrix, (x, y, z + h), points, origin)

    def plane(self, size, origin=(0,0,0), **kw):
        w, h = size
        x, y, z = origin = self.origin(size, origin, **kw)
        points = [(w, 0, 0), (w, h, 0), (0, h, 0), (0, 0, 0)]
        directrix = self.line((0, 0, 0), (w, 0, 0), origin, child=True)
        self.surface(directrix, (x, y + h, z), points, origin)

    def wedge(self, w, h, origin=(0,0,0), flipx=False):
        x, y, z = origin
        if flipx: w, x = -w, x + w
        origin = x, y, z
        points = [(0, 0, 0), (w, 0, 0), (w, h, 0)]
        directrix = self.line((0, 0, 0), (w, 0, 0), origin, child=True)
        self.surface(directrix, (x, y + h, z), points, origin)

    def cube(self, size, origin=(0,0,0)):
        w, l, h = size
        x, y, z = origin
        self.plane((w, l), origin=origin)
        self.plane((w, l), origin=(x, y, z + h))
        self.yzplane((l, h), origin=origin)
        self.yzplane((l, h), origin=(x + w, y, z))
        self.xzplane((w, h), origin=origin)
        self.xzplane((w, h), origin=(x, y + l, z))

    def yslabline(self, length, rad, origin=(0,0,0)):
        directrix = self.line((0, 0, 0), (0, 1, 0), origin, child=True)
        vector = self.line((0, 0, 0), (0, 0, rad), origin, child=True)
        self.cylinder(directrix, vector, origin)
        vector = self.line((0, length, 0), (0, length, rad), origin, child=True)
        self.cylinder(directrix, vector, origin)
        vector = self.line((0, 0, rad), (0, length, rad), origin, child=True)
        self.cylinder(directrix, vector, origin)

    def xslabline(self, length, rad, origin=(0,0,0)):
        directrix = self.line((0, 0, 0), (1, 0, 0), origin, child=True)
        vector = self.line((0, 0, 0), (0, 0, rad), origin, child=True)
        self.cylinder(directrix, vector, origin)
        vector = self.line((length, 0, 0), (length, 0, rad), origin, child=True)
        self.cylinder(directrix, vector, origin)
        vector = self.line((0, 0, rad), (length, 0, rad), origin, child=True)
        self.cylinder(directrix, vector, origin)

    def ypipe(self, length, rad, origin=(0,0,0)):
        directrix = self.line((0, 0, 0), (0, 1, 0), origin, child=True)
        vector = self.line((0, 0, rad), (0, length, rad), origin, child=True)
        self.cylinder(directrix, vector, origin)

    def xpipe(self, length, rad, origin=(0,0,0)):
        directrix = self.line((0, 0, 0), (1, 0, 0), origin, child=True)
        vector = self.line((0, 0, rad), (length, 0, rad), origin, child=True)
        self.cylinder(directrix, vector, origin)



def split_list_of_words(list_):
    return list(chain.from_iterable(map(str.split, list_)))

def off2igs(fileName):
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(scriptDir,fileName)
  
    scriptDir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_file = os.path.join(scriptDir,fileName)
    with open(path_to_file) as f:
        contents = f.readlines()
  
    numNodes = int(contents[1].split()[0])
    numBeams = int(contents[1].split()[2])
  
    nodalInfo= split_list_of_words(contents[2:numNodes+2])
    nodalInfo = np.asarray([float(x) for x in nodalInfo]).reshape((-1,3))
  
    connInfo= split_list_of_words(contents[numNodes+2:numNodes+2+numBeams])
    connInfo = np.asarray([int(x) for x in connInfo]).reshape((-1,2))
  
    x0 = nodalInfo[connInfo[:,0],0]
    x1 = nodalInfo[connInfo[:,1],0]
    y0 = nodalInfo[connInfo[:,0],1]
    y1 = nodalInfo[connInfo[:,1],1]
    z0 = nodalInfo[connInfo[:,0],2]
    z1 = nodalInfo[connInfo[:,1],2]
    mat = np.asarray([x0,y0,z0,x1,y1,z1]).T
    iges = Iges()
    for i in range(numBeams):
      iges.line( mat[i,0:3], mat[i,3:], origin=(0,0,0))
  
    base_fileName = path_to_file[:-4]
    new_filename = base_fileName + ".igs"
    iges.write(new_filename)
  
def custom_minimize(objCall, x0, bounds,method='GCMMA',constraintCall=None, options=None,callback=None):
    """
    Optimize using the MMA or GCMMA (default) method.
    
    Parameters:
    - objCall: Callable for the objective function. Should return:
        - Objective value: shape (1,)
        - Gradient: shape (n,)
    - x0: Initial guess for the optimization (shape (n,))
    - bounds: scipy.optimize.Bounds object for variable bounds.
    - constraintCall: Callable for the constraint function. Should return:
        - Constraint values: shape (m,1)
        - Gradient: shape (m, n)
    - options: Dictionary for optimization options (e.g., maxiter, kkttol, etc.).
    - callback: Callable function which takes x as input every iter, can be used for plot and early stop.
    
    Returns:
    - xval: Optimized variable values (shape (n,))
    - f0val: Final objective value (shape (1,))
    - func_evals: Number of function evaluations.
    """
    n = len(x0)
    if constraintCall== None: # size of 1 constraint
        def constraintCall(x):
            c = 1e-6*np.ones((1,1))
            dc = 1e-12*(x[np.newaxis]*0+1.)
            return c,dc
            
    if bounds == None:
        bounds = Bounds([0.]*n,[1.]*n) # scipy bounds
            
    # Handle options
    default_options = {
    'maxiter': 200,
    'miniter':10,
    'kkttol': 1e-5,
    'disp': False,
    'maxfun': 1000,
    'move_limit':0.1
    }
    
    # Use defaults for missing options
    if options is None:
        options = {}  # Initialize empty if none provided
    
    # Check for each key in the options; if missing, use default value
    options['maxiter'] = options['maxiter'] if 'maxiter' in options else default_options['maxiter']
    options['miniter'] = options['miniter'] if 'miniter' in options else default_options['miniter']
    options['kkttol'] = options['kkttol'] if 'kkttol' in options else default_options['kkttol']
    options['disp'] = options['disp'] if 'disp' in options else default_options['disp']
    options['maxfun'] = options['maxfun'] if 'maxfun' in options else default_options['maxfun']
    options['move_limit'] = options['move_limit'] if 'move_limit' in options else default_options['move_limit']

    print('Running '+method+' optimizer from mmapy')
    
    x,fun,fun_eval = optimizeMMA(objCall, x0, bounds, constraintCall, options=options,callback=callback)
   
    result = SimpleNamespace()                
    result.x = x
    result.fun = fun
    result.fun_eval = fun_eval
    
    return result
    

def optimizeMMA(objCall, x0, bounds, constraintCall, options,callback):
    """
    Optimize using the MMA method.
    
    Parameters:
    - objCall: Callable for the objective function. Should return:
        - Objective value: shape (1,)
        - Gradient: shape (n,)
    - x0: Initial guess for the optimization (shape (n,))
    - bounds: scipy.optimize.Bounds object for variable bounds.
    - constraintCall: Callable for the constraint function. Should return:
        - Constraint values: shape (m,1)
        - Gradient: shape (m, n)
    - options: Dictionary for optimization options (e.g., maxiter, kkttol, etc.).
    
    Returns:
    - xval: Optimized variable values (shape (n,))
    - f0val: Final objective value (shape (1,))
    - func_evals: Number of function evaluations.
    """
    
    # unpack options
    max_iter = options['maxiter'] 
    min_iter = options['miniter']
    kkttol = options['kkttol'] 
    move_limit = options['move_limit'] 
    disp = options['disp']
    max_fun = options['maxfun']
    func_evals = 0

    # Initial values
    f0val_init, df0dx_init = objCall(x0)
    fval, dfdx = constraintCall(x0)
    func_evals += 1

    n = len(x0)
    m = np.shape(fval)[0]

    xmin = bounds.lb.reshape((-1, 1))
    xmax = bounds.ub.reshape((-1, 1))
    
    xval = x0.copy()[np.newaxis].T
    xold1 = xval.copy()
    xold2 = xval.copy()
    
    c = 1000 * np.ones((m, 1))
    d = np.ones((m, 1))
    a0 = 1
    a = np.zeros((m, 1))
    move = move_limit
    scale_grad = 100.0
    
    outer_iter = 0
    kktnorm = np.array([kkttol + 10])
    change = kktnorm.copy()
    change4 = kktnorm.copy()
    L_best = np.zeros(max_iter) + f0val_init.copy()
    xbest = xval.copy()
            
    f0valnew, df0dxnew = f0val_init.copy(), df0dx_init.copy()
    fvalnew, dfdxnew = fval.copy(), dfdx.copy()
    
    # The main iteration loop
    for outer_iter in range(1,max_iter):
        
        success = False
        move = np.abs(1-np.sin(np.pi/300 + (outer_iter+3)*np.pi/10))
        
        for _ in range(10):
            # Solve the MMA subproblem
            xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
                m, n, outer_iter, xval, xmin, xmax, xold1, xold2, f0valnew[np.newaxis], df0dxnew[np.newaxis].T, fvalnew, dfdxnew, xmin, xmax, a0, a, c, d, move)
                    
            # Update the variables for the next iteration
            xold2 = xold1.copy()
            xold1 = xval.copy()
            xval = xmma.copy()
    
            # Compute the objective and constraint values at the new point
            f0valnew,df0dxnew = objCall(xval.reshape(-1))
            fvalnew,dfdxnew = constraintCall(xval.reshape(-1))
            func_evals += 1  # Increment the function evaluation count
            
            L = f0valnew + (fvalnew.T@lam)
            # print(L,f0valnew,(fvalnew.T@lam),L_best[outer_iter-1],np.mean(xval))
            if L < L_best[outer_iter-1]:
                success = True
                # print(success)
                break
          
        if success:
            f0val, df0dx = f0valnew.copy(), df0dxnew.copy()
            fval, dfdx = fvalnew.copy(), dfdxnew.copy()
        else:
            xold2 = xbest.copy()
            xold1 = xbest.copy()
            xval = xbest.copy()
            # Compute the objective and constraint values at the new point
            f0val, df0dx = objCall(xval.reshape(-1))
            fval, dfdx = constraintCall(xval.reshape(-1))
            func_evals += 1  # Increment the function evaluation count
            # find the best lam again    
            xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
                    m, n, outer_iter, xval, xmin, xmax, xold1, xold2, f0val[np.newaxis], df0dx[np.newaxis].T, fval, dfdx, xmin, xmax, a0, a, c, d, move)
                
             
        # Check KKT conditions
        residu, kktnorm, residumax = kktcheck(
            m, n, xbest, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx[np.newaxis].T, fval, dfdx, a0, a, c, d)
        
        xbest = xval.copy()
        L_best[outer_iter] = f0val + (fval.T@lam)
        
        if outer_iter > 1:
            change = abs(L_best[outer_iter]-L_best[outer_iter-2])
            if outer_iter > 3:
                change4 = abs(L_best[outer_iter]-L_best[outer_iter-4])
        else: 
            change = abs(L_best[outer_iter]-L_best[outer_iter-1])
        
        if disp:
            print(f"Iter {outer_iter}: f0 = {f0val.item():.4g}, L-best = {L_best[outer_iter].item():.4g}, KKT norm = {kktnorm.item():.4g}, Function evals = {func_evals}, n = {len(xbest)}")
        if callback != None:
            callback(xbest.reshape(-1))
            
        # Check for termination condition
        if ((kktnorm <= kkttol  or change <= kkttol or func_evals >= max_fun) and (outer_iter >= min_iter or change4 <= kkttol)):
            if f0val < f0val_init:
                print("Local minima found")
                if change4 <= kkttol:
                    print("Objective did not change for last 4 iterations")
            break    
            
            
    return xbest.reshape(-1), f0val, func_evals
        
def barPlotofOptimization(V0, Vopt, VDataAbaqusOpt,threshold=3.0,hatch_patterns=None,all_labels=None):
    n_groups = len(V0)
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.5
    if hatch_patterns is None:
        # Hatch patterns for shading
        hatch_patterns = ['///', 'xx', 'oo']  # Different patterns for Type A, B, C
    if all_labels is None:
        all_labels = ['Initial $(\phi_{0})$','Optimized $(\phi_{opt NN})$',
                    'Optimized Abaqus $(\phi_{opt FEA})$']
    
    c_map_name = 'jet'
    colorsSet = plt.get_cmap(c_map_name)(np.linspace(0, 1.0,3)) 
    
    # Determine if there's a significant outlier (you might need to adjust the threshold)
    max_val = max(max(V0), max(Vopt), max(VDataAbaqusOpt))
    gap_val = int((max_val - threshold)/2.0)
    
    if max_val > threshold:
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.1)

        # Create the bottom subplot first
        ax_bottom = fig.add_subplot(gs[1])

        # Create the top subplot and share its x-axis with ax_bottom
        ax_top = fig.add_subplot(gs[0])

        # Plot on the bottom axes (for smaller values)
        ax_bottom.bar(index - bar_width, V0, bar_width, alpha=opacity, color=colorsSet[0], label =all_labels[0], hatch=hatch_patterns[0], edgecolor='black')
        ax_bottom.bar(index, Vopt, bar_width, alpha=opacity, color=colorsSet[1], label =all_labels[1], hatch=hatch_patterns[1], edgecolor='black')
        ax_bottom.bar(index + bar_width, VDataAbaqusOpt, bar_width, alpha=opacity, color=colorsSet[2], label =all_labels[2], hatch=hatch_patterns[2], edgecolor='black')
        ax_bottom.set_ylim(0, threshold) # Adjust upper limit

        # Plot on the top axes (for the large value)
        ax_top.bar(index - bar_width, V0, bar_width, alpha=opacity, color=colorsSet[0], hatch=hatch_patterns[0], edgecolor='black')
        ax_top.bar(index, Vopt, bar_width, alpha=opacity, color=colorsSet[1], hatch=hatch_patterns[1], edgecolor='black')
        ax_top.bar(index + bar_width, VDataAbaqusOpt, bar_width, alpha=opacity, color=colorsSet[2], hatch=hatch_patterns[2], edgecolor='black')
        ax_top.set_ylim(threshold+gap_val*1.01, max_val * 1.4) # Adjust lower and upper limits

        # Annotate " " for VDataAbaqusOpt == 0
        for i, val in enumerate(VDataAbaqusOpt):
            if val == 0:
                ax_bottom.text(index[i] + 0.9 * bar_width, 0.1,  # Adjust y-position as needed
                        'Abaqus Failed', ha='center', va='bottom', color='black',
                        fontweight='bold',rotation=90)
    

        # Diagonal lines for break
        d = .5
        kwargs = dict(marker=[(-1, -d), (1, d)], linestyle='none',
                      transform=ax_top.transAxes, clip_on=False,
                      markersize=10, color='black') # Added markersize and color
        ax_top.plot([0, 1], [0, 0], **kwargs)

        kwargs_bottom = dict(marker=[(-1, -d), (1, d)], linestyle='none',
                             transform=ax_bottom.transAxes, clip_on=False,
                             markersize=10, color='black') # Added markersize and color
        ax_bottom.plot([0, 1], [1, 1], **kwargs_bottom)
        
         # Hide spines
        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)

        ax_top.set_xticks([]) 
        ax_bottom.set_xticks([]) 
       

        # ax_bottom.set_xlabel('Initial Guess Index', fontsize=16)
        ax_bottom.set_ylabel('Objective Value', fontsize=16)
        ax_bottom.set_xticks(index)
        ax_bottom.set_xticklabels([f'IG {i+1}' for i in range(n_groups)], fontsize=14)
        ax_bottom.tick_params(axis='y', labelsize=16)
        ax_top.tick_params(axis='y', labelsize=16)
        ax_bottom.grid(axis='y', linestyle='--', alpha=0.7)
        ax_top.grid(axis='y', linestyle='--', alpha=0.7)


        # Set y-axis tick labels as floats for the top plot
        yticks_bottom = np.arange(0, threshold*1.01, 1.0) # Adjust range and step as needed
        ax_bottom.set_yticks(yticks_bottom)
        ax_bottom.set_yticklabels([f'{y:.1f}' for y in yticks_bottom], fontsize=16) # Format as float with one decimal place

        # Set y-axis tick labels as floats for the top plot
        yticks_top = np.linspace(threshold+gap_val, max_val * 1.4, 3) # Adjust range and number of ticks as needed
        ax_top.set_yticks(yticks_top)
        ax_top.set_yticklabels([f'{y:.1f}' for y in yticks_top], fontsize=16) # Format as float

        # ax_bottom.legend(fontsize=18, loc='upper right')

        # plt.tight_layout()
        # Create a single legend outside the subplots
        handles, labels = ax_bottom.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=16, loc='upper right', 
        bbox_to_anchor=(0.9, 0.9), bbox_transform=fig.transFigure)
        # fig.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
      
        rects1 = ax.bar(index, V0, bar_width,
                        alpha=opacity,
                        color=colorsSet[0],
                        label =all_labels[0],
                        hatch=hatch_patterns[0],
                        edgecolor='black')
    
        rects2 = ax.bar(index + bar_width, Vopt, bar_width,
                        alpha=opacity,
                        color=colorsSet[1],
                        label =all_labels[1],
                        hatch=hatch_patterns[1],
                        edgecolor='black')
    
        rects3 = ax.bar(index + 2 * bar_width, VDataAbaqusOpt, bar_width,
                        alpha=opacity,
                        color=colorsSet[2],
                        label =all_labels[2],
                        hatch=hatch_patterns[2],
                        edgecolor='black')
    
        # Annotate "Failure" for VDataAbaqusOpt == 0
        for i, val in enumerate(VDataAbaqusOpt):
            if val == 0:
                ax.text(index[i] + 2 * bar_width, 0.1,  # Adjust y-position as needed
                        'Abaqus Failed', ha='center', va='bottom', color='black',
                        fontweight='bold',rotation=90)
    
        # ax.set_xlabel('Initial Guess Index')
        ax.set_ylabel('Objective Value',fontsize=16)
        ax.set_yticklabels(ax.get_yticks(),fontsize=16)
        # ax.set_title('Lower the better')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels([f'IG {i+1}' for i in range(n_groups)],fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.set_ylim(None,1.4*max(V0))

        #  Create a single legend outside the subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=16, loc='upper right', 
        bbox_to_anchor=(0.95, 0.95), bbox_transform=fig.transFigure)
        fig.tight_layout()
        # plt.savefig('ComparisonBarPlot.png', bbox_inches='tight', dpi=300)
        # plt.show()

def plot_optimization_curves(optFrame, x0_all, xopt_all, V0, Vopt, DataAbaqusOpt=None):
    num_curves = len(x0_all)
    markers = ['o', 's', '^', 'd','p','h','<', '>']  # Add more if needed
    colors = plt.cm.viridis(np.linspace(0, 1, num_curves + 2))  # Adjust for the 2 extra curves.
    marker_stride = 8  # Place a marker every 8 data points
    maxU = optFrame.uinput[-1]
    
    fHeight = 8 # inches
    lnWdth = 6
    mrkSize = 16
    fntSize = 26

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(3*fHeight, fHeight), sharey=True)  # sharey=True to link y-axis limits

    # Plot Target in both figures
    ax_a.plot(optFrame.uinput / maxU, optFrame.Finput, linestyle='solid',  linewidth=lnWdth, label='Target', color='red')
    ax_b.plot(optFrame.uinput / maxU, optFrame.Finput, linestyle='solid',  linewidth=lnWdth, color='red', label='Target')  # No label to avoid duplication
    ax_c.plot(optFrame.uinput / maxU, optFrame.Finput, linestyle='solid',  linewidth=lnWdth, color='red', label='Target')  # No label to avoid duplication

    for i in range(num_curves):
        marker = markers[i % len(markers)]
        color = colors[i]

        # Initial Guess (Figure a)
        V0[i], _ = optFrame.objectiveCall1(x0_all[i, :], Jac=False)
        ax_a.plot(optFrame.ui[::marker_stride] / maxU, optFrame.Fi[::marker_stride], linestyle='', marker=marker, color=color,
                 label=f'Initial {i+1}',markersize=mrkSize)
        ax_a.plot(optFrame.ui / maxU, optFrame.Fi, linestyle='-',  linewidth=lnWdth, color=color)  # Add a faint line connecting all points

        # Optimized Results (Figure b)
        Vopt[i], _ = optFrame.objectiveCall1(xopt_all[i, :], Jac=False)
        ax_b.plot(optFrame.ui[::marker_stride] / maxU, optFrame.Fi[::marker_stride], linestyle='', marker=marker, color=color,
                 label=f'Optimized {i+1}',markersize=mrkSize)
        ax_b.plot(optFrame.ui / maxU, optFrame.Fi, linestyle='-',  linewidth=lnWdth, color=color)  # Add a faint line connecting all points

    if DataAbaqusOpt is not None:
        VDataAbaqusOpt = np.zeros(num_curves)
        for i in range(num_curves):
            marker = markers[i % len(markers)]
            color = colors[i]
            VDataAbaqusOpt[i],_ = optFrame.objectiveCall2(DataAbaqusOpt[i,:,1]) 
            ax_c.plot(DataAbaqusOpt[i,:,0] / maxU, DataAbaqusOpt[i,:,1], linestyle='', linewidth=lnWdth, \
            marker=marker, color=color,label=f'Abaqus FEA {i+1}',markersize=mrkSize)
            ax_c.plot(DataAbaqusOpt[i,:,0] / maxU, DataAbaqusOpt[i,:,1], linestyle='-', linewidth=lnWdth, \
            color=color)
    
    
    # Labels and Titles for Figure a
    ax_a.set_xlabel('Normalized displacement', fontsize=fntSize+4)
    ax_a.set_ylabel('Force in N', fontsize=fntSize+4)
    ax_a.set_title('Initial Guesses', fontsize=fntSize)
    ax_a.legend(loc='upper right', fontsize=fntSize)
    ax_a.grid(True)
    ax_a.tick_params(axis='both', which='major', labelsize=fntSize-2)
    oldYlim = ax_a.get_ylim()[1]  # Get the current ymax
    ax_a.set_ylim(None, 1.4 * oldYlim)  # Set the new y-axis limits # make it 1.0 for 7 var

    # Labels and Titles for Figure b
    ax_b.set_xlabel('Normalized displacement', fontsize=fntSize+4)
    # ax_b.set_ylabel('Force in N', fontsize=fntSize) # Added the missing y-label
    ax_b.set_title('Optimized Results', fontsize=fntSize)
    ax_b.legend(loc='upper right', fontsize=fntSize)
    ax_b.grid(True)
    ax_b.tick_params(axis='both', which='major', labelsize=fntSize-2)

    # Labels and Titles for Figure c
    ax_c.set_xlabel('Normalized displacement', fontsize=fntSize+4)
    # ax_c.set_ylabel('Force in N', fontsize=fntSize) # Added the missing y-label
    ax_c.set_title('Abaqus FEA on Optimized Results', fontsize=fntSize)
    ax_c.legend(loc='upper right', fontsize=fntSize)
    ax_c.grid(True)
    ax_c.tick_params(axis='both', which='major', labelsize=fntSize-2)

    fig.tight_layout()

    return V0, Vopt,VDataAbaqusOpt
    
def plotOptxk(ax,optFrame,xk_array,strAdd,colors,markers,markersize):
    v = np.zeros(xk_array.shape[0])
    deltax = 0.05
    # Calculate v and plot points and text
    for i in range(xk_array.shape[0]):
    
        v[i], _ = optFrame.objectiveCall1(xk_array[i,:], Jac=False)
        
        if xk_array.shape[0] > 2:
            textPrnt = f'{i + 1}'
        else:
            textPrnt = f' '
            
        # Ensure text position is within 0-1 bounds
        text_x = max(deltax, min(1-deltax, xk_array[i, 0]))  # Clamp x-coordinate
        text_y = max(deltax, min(1-deltax, xk_array[i, 1]))  # Clamp y-coordinate

        # Add text annotation at (text_x, text_y)
        ax.text(text_x, text_y, textPrnt, color='w', fontsize=20)
    # Plot the point
    ax.scatter(xk_array[:, 0], xk_array[:, 1], color= colors, marker=markers, s=markersize,label=strAdd)
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)
    
    return ax
    

def plot_objective_landscape(optFrame, n: int, square_size: tuple = (12, 12)):
    """
    Generates and plots the objective landscape for a given optFrame using a neural network,
    identifies global minima, and returns relevant data and the plot axes.

    Args:
        optFrame: An object that must have an 'objectiveCall1' method.
                  This method should accept a NumPy array (e.g., np.array([val1, val2]))
                  and a boolean 'Jac' (Jacobian flag), and return a tuple
                  (objective_value, _).
        n (int): The number of intervals for the meshgrid along each axis.
                 A higher 'n' results in a finer grid and more detailed plot,
                 but increases computation time.
        square_size (tuple, optional): A tuple (width, height) in inches for the
                                       figure size. Defaults to (12, 12).

    Returns:
        tuple: A tuple containing:
            - xGlobalNN (np.ndarray): A 2x2 NumPy array. Each row represents the
                                      (Radius 1, Radius 2) coordinates of the
                                      top two global minima found in the objective space.
            - ax (matplotlib.axes.Axes): The Matplotlib Axes object containing the
                                         generated contour plot. This can be used
                                         to further customize or display the plot.
    """

    # Define the intervals for the two dimensions (Radii 1 and Radii 2)
    # These intervals typically range from 0.0 to 1.0, divided into 'n' steps.
    intervals_I = np.linspace(0.0, 1.0, n)

    # Create a 2D meshgrid from the intervals.
    # 'I' will hold the 'Radii 1' values for each point in the grid.
    # 'J' will hold the 'Radii 2' values for each point in the grid.
    # 'indexing='ij'' ensures that I and J correspond to row and column indices.
    I, J = np.meshgrid(intervals_I, intervals_I, indexing='ij')

    # Initialize a 2D array 'V' to store the objective values for each point in the grid.
    V = np.zeros((n, n))

    # Record the start time for performance measurement
    st = time.perf_counter()

    # Iterate through each point in the grid (i, j)
    # and call the objective function to populate the 'V' array.
    for i in range(n):
        for j in range(n):
            # optFrame.objectiveCall1 is expected to be a method that calculates
            # the objective value for a given input array [Radius 1, Radius 2].
            # 'Jac=False' indicates that the Jacobian is not required for this call.
            # The second return value (Jacobian or other) is ignored with '_'.
            V[i, j], _ = optFrame.objectiveCall1(np.array([intervals_I[i], intervals_I[j]]), Jac=False)

    # Calculate the total time taken for objective function calls
    Total_tm = time.perf_counter() - st
    print(f"Total time for NN objspace = {Total_tm:.4f} seconds")

    # Create a Matplotlib figure and axes for the plot.
    # 'num=1' assigns a figure number.
    fig = plt.figure(num=1)
    # Set the figure size based on the 'square_size' parameter.
    fig.set_size_inches(square_size[0], square_size[1])
    # Add a subplot to the figure. '111' means 1x1 grid, first subplot.
    ax = fig.add_subplot(111)

    # Create filled contour plot using 'contourf'.
    # 'I' and 'J' are the grid coordinates, 'V' is the objective value at each point.
    # 'cmap='jet'' sets the colormap. 'levels=50' creates 50 contour levels.
    cntr = ax.contourf(I, J, V, cmap='jet', levels=50)

    # Create contour lines using 'contour' for specific objective values.
    # 'colors='y'' sets the line color to yellow.
    # 'levels' specifies the exact objective values for which to draw lines.
    # 'linewidth' sets the thickness of the contour lines.
    cntr_line = ax.contour(I, J, V, colors='y', levels=[0.5, 1, 2, 3, 5, 10, 20, 23.0], linewidths=2.0)

    # Add labels to the contour lines for readability.
    # 'inline=True' ensures labels are placed within the lines.
    ax.clabel(cntr_line, inline=True, fontsize=10)

    # Set axis labels with specified font size.
    ax.set_xlabel('Radii 1', fontsize=18)
    ax.set_ylabel('Radii 2', fontsize=18)

    # The following lines are commented out in the original snippet but show
    # potential for 3D plotting or title setting.
    # ax.set_zlabel('Objective Value')
    # ax.view_init(elev=el1, azim=az1)
    # ax.set_title("Objective landscape with NN")

    # Ensure the plot has a square aspect ratio, which is good for 2D landscapes.
    ax.set_box_aspect(1)

    # Add a color bar to indicate the mapping of colors to objective values.
    # 'fraction' controls the size of the color bar relative to the axes.
    cbar = fig.colorbar(cntr, fraction=0.04)
    # Set the font size for the color bar tick labels.
    cbar.ax.tick_params(labelsize=16)

    # Set the font size for the main plot's axis tick labels.
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Find the top 10 global minima (lowest objective values).
    # Convert the 'V' array (flattened) to a PyTorch tensor.
    # 'largest=False' ensures we get the smallest values.
    tV = torch.topk(torch.tensor(V.reshape(-1)), k=10, largest=False)

    # Print the values of the top 10 minima.
    print("Global minima values (top 10):", tV.values.numpy())

    # Print the corresponding Radii 1 values for these minima.
    print(f"Corresponding Radius 1 values: {I.reshape(-1)[tV.indices.numpy()]}")

    # Print the corresponding Radii 2 values for these minima.
    print(f"Corresponding Radius 2 values: {J.reshape(-1)[tV.indices.numpy()]}")

    # Extract the (Radius 1, Radius 2) coordinates for the top 2 global minima.
    # tV.indices.numpy()[0:2] gets the indices of the first two smallest values.
    # I.reshape(-1)[indices] and J.reshape(-1)[indices] retrieve their coordinates.
    # np.stack combines these into a 2x2 array where each row is [R1, R2].
    xGlobalNN = np.stack((I.reshape(-1)[tV.indices.numpy()[0:2]],
                          J.reshape(-1)[tV.indices.numpy()][0:2]),
                         axis=1)

    # Return the extracted global minima coordinates and the plot axes object.
    return xGlobalNN, ax
    
## plotter radius and beam

def plotStructure(R,nodeXY,connectivity, titleStr, plotDeformed = True,TrueScale=False,
      fig=plt.figure(1),nodeAnnotate=False,elemAnnotate=False,thicknessPlot=False):
    
    d = nodeXY[connectivity[:, 1]] - nodeXY[connectivity[:, 0]]

    d_magnitude = np.linalg.norm(d, axis=1, keepdims=True)
    
    d_normalized = np.where(d_magnitude == 0, 0, d / d_magnitude)
    
    predominant_axis = np.argmax(np.abs(d_normalized), axis=1)
    
    v = np.zeros((d.shape[0], 3))
    
    mask_xy_axis = (predominant_axis == 0) | (predominant_axis == 1)
    mask_z_axis = (predominant_axis == 2)
    
    v[mask_xy_axis] = [0., 0., -1.]
    v[mask_z_axis] = [1., 0., 0.]

    fig.clear()
    # Clear any colorbars
    for ax in fig.axes:
        if ax.collections:  # Colorbars are often stored as collections
            ax.remove()
    
    ax = fig.add_subplot(111, projection='3d')
  
    if nodeAnnotate:
      dx = np.max(nodeXY)*0.01
      for i, point in enumerate(nodeXY):
        ax.text(point[0]+dx, point[1]+dx, point[2]+dx,f'{i}', color='m', fontsize=8, ha='right')
  
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Points and Lines')
    
    R_unq = np.unique(R)
    
    # Generate colors using the colormap 
    c_map_name = 'jet'
    colors_plt_1 = plt.get_cmap(c_map_name)(np.linspace(0, 0.5, int(0+len(R_unq)//2),endpoint=False)) 
    colors_plt_2 = plt.get_cmap(c_map_name)(np.linspace(0.5, 1, int(1+len(R_unq)//2))) 
    # Combine the two color arrays 
    colors_plt = np.concatenate((colors_plt_1, colors_plt_2), axis=0)
    
    eleNum = 0
    for conn in connectivity:
      points = nodeXY[conn]
      ax.plot(points[:, 0], points[:, 1], points[:, 2], c='k')
      if elemAnnotate:
        dx = np.max(nodeXY)*0.01
        ax.text(np.mean(points[:, 0])+dx, np.mean(points[:, 1])+dx, np.mean(points[:, 2])+dx,f'{eleNum}', color='r', fontsize=10, ha='right')
      if thicknessPlot:
        indxs = np.where( R_unq == R[eleNum])
        r_value = R[eleNum]
        ax = create_cylinder(ax, points[0,:], points[1,:], R=r_value, color=colors_plt[indxs],not_v=v[eleNum])  # Adjust radius and color
        
      eleNum = eleNum + 1;
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(titleStr)
  
    # Manually set the aspect ratio by adjusting the limits
    # ax.set_xlim3d(nodeXY[:, 0].min(), nodeXY[:, 0].max())
    # ax.set_ylim3d(nodeXY[:, 1].min(), nodeXY[:, 1].max())
    # ax.set_zlim(nodeXY[:, 2].min(), nodeXY[:, 2].max())
    # ax.view_init(0,0,60)
    # ax.view_init(90,-90,0) # xy plane
    # ax.view_init(elev=90, azim=0) # xy plane
    # ax.view_init(elev=0, azim=-90) # xz plane

    r_unq = R_unq
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(c_map_name, len(r_unq)) )
    sm.set_array(r_unq)  # Dummy array for the colorbar
    
    # # Zoom to a specific region by setting the view 
    # ax.set_xlim([0, 40]) 
    # ax.set_ylim([15, 35]) 
    # ax.set_zlim([-1, 12])
    
    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax, shrink=1.0, aspect=15, pad=0.15,
    ticks=r_unq) 
    # Change the font size of the ticks 
    cbar.ax.tick_params(labelsize=14)    
    ax.set_aspect('equal')

  # plt.show()
  
def create_cylinder(ax,p0,p1,R,color,not_v):
    
    slices = 20
    origin = np.array([0, 0, 0])
    #axis and radius
    
    #vector in direction of axis
    v = p1 - p0
    #find magnitude of vector
    mag = np.linalg.norm(v)
    #unit vector in direction of axis
    v = v / mag
    # #make some vector not in the same direction as v
    # not_v = np.array([0, 0, 1])
    # if (v == not_v).all() or (v == -not_v ).all():
    #     not_v = np.array([1, 0, 0])
        
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    #normalize n1
    n1 /= np.linalg.norm(n1)
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, slices)
    theta = np.linspace(0, 2 * np.pi, slices)
    #use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    #generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R.item() * np.sin(theta) * n1[i] + R.item() * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    
    # ax.plot_surface(X, Y, Z,color=color,shade=True)
    
    # Plot the surface with shading
    ax.plot_surface(X, Y, Z, color=color, shade=True)
    
    return ax