
import torch
from scipy.optimize import minimize,least_squares, Bounds
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

from src.utilFuncs import *
torch.autograd.set_detect_anomaly(True)

class OptimizeFrame:

    def __init__(self,frameNN,inputCurve):
        self.frameNN=frameNN;
        self.Finput=torch.from_numpy(inputCurve[1]);
        self.uinput=torch.from_numpy(inputCurve[0]).reshape((-1,1));
        
        self.obj0 = np.array([1.]); # to normalize the optimization wrt initial guess objective value
        
        self.iterations = []
        self.objective_values = []
        self.con = 0.
                        
        self.memo = {}
        self.analysisState = True
        self.p = 2.0
        # self.funCount = 0; # count the function calls
        
    
    # @profile
    def objectiveCall1(self,xin,Jac=True):    
        
        Finput = self.Finput.clone()
        
        Finput =  (Finput - self.frameNN.minVoutput)/(self.frameNN.maxVoutput-self.frameNN.minVoutput)
        # self.frameNN.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.tensor(xin,requires_grad=True,device=self.frameNN.device).float()
        
        Fcp = self.frameNN(x.reshape(1,-1))
        
        uiMean = torch.linspace(0,1,81)# same as number of loadsteps
        FiSum = curve_setup(Fcp,t=uiMean.to(Fcp.device),curve='spline').cpu().reshape(-1)
        # # Linear interpolation function with boundary handling (out of the bounds handling)
        FiSumN,uiMeanN = linear_interpolation_torch(uiMean*self.uinput[-1],FiSum,self.uinput.reshape(-1)) # N numbers = self.uinput size        
        
        # FiSumN,uiMeanN = FiSum.clone(),uiMean.clone()
        dF = torch.norm((FiSumN[3:-3]- Finput[3:-3])/((Finput[-1] - Finput[0])/1.),self.p) # 3 to -3 importance 
        
        objValue_FU = 0*dF + torch.norm((FiSumN - Finput)/((Finput[-1] - Finput[0])/1.),self.p)                 
        objValue =  objValue_FU/torch.from_numpy(self.obj0)
        
        if Jac:
            # objValue.backward(retain_graph=True)    
            Sensitivity = torch.autograd.grad(objValue,x,retain_graph=True, create_graph=True)[0].detach().cpu().numpy()

            # forceCon.backward(retain_graph=True)
            # forceConGrad = torch.autograd.grad(forceCon,x,retain_graph=True, create_graph=True)[0].detach().cpu().numpy()
        
            # print("Sens = ",objValue,Sensitivity)
            # import sys;sys.exit()
        else:
            Sensitivity = 0*x.detach().cpu().numpy()
        
        objValue = objValue.detach().cpu().numpy()
        # x.grad.zero_()
        
        self.Fi = FiSum.clone().detach() 
        self.ui = uiMean.clone()*self.uinput[-1]

        return objValue, Sensitivity
    
    # this Function is used to calculate the objective function thorugh generated data
    def objectiveCall2(self,F):    
        # print(F)
        # import sys;sys.exit()
        Finput = self.Finput.clone()
        
        # Finput =  (Finput - self.frameNN.minVoutput)/(self.frameNN.maxVoutput-self.frameNN.minVoutput)
        
        FiSumN = torch.tensor(F.copy())
        dF = torch.norm((FiSumN[3:-3]- Finput[3:-3])/((Finput[-1] - Finput[0])/1.),self.p) # 3 to -3 importance 
        
        objValue_FU = dF*0 + torch.norm((FiSumN - Finput)/((Finput[-1] - Finput[0])/1.),self.p)                 
        objValue = objValue_FU/torch.from_numpy(self.obj0)
        
        
        Sensitivity = 0.0
        
        objValue = objValue.detach().cpu().numpy()
        # x.grad.zero_()
        
        self.Fi = FiSumN.clone().detach()
        self.ui = self.uinput.clone()

        return objValue, Sensitivity
    
    
    def callbackFun(self,xk):
        self.iterations.append(len(self.iterations) + 1)
        v,_ = self.objectiveCall1(xk)
        self.objective_values.append(v)
        
        plt.figure(2)
        self.plotIterVsObj()
        # plt.close(2)
        # print(self.iterations)
        plt.figure(4)
        self.plotFDcurves(self.ui,self.Fi,'Current_Iter'+' Obj Value = '+str(np.round(v,4)),
        fileSaveName=self.fileSaveLoc+'/IterSp' + str(len(self.iterations))+'.png')
        
    def plotIterVsObj(self):
        plt.ion()
        plt.clf()
        plt.plot(self.iterations, self.objective_values, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Iteration vs Objective Value')
        # plt.ylim([0,self.objective_values[0]+1.0])
        # plt.pause(0.01)
        plt.show()
        # plt.show(block=True)
    
    def optimizerRun(self,x0,algorithm='TNC',tol=1e-6,fileSaveLoc=None):
    
        self.fileSaveLoc = fileSaveLoc
        self.alpha = 0.0 # start with uniform phase 1 opt
        self.funCount = 0.0
        self.con = 0.0
        
        # xbest = x0[0:1].copy()
        xbest = x0.copy()
        obj_init,_= self.objectiveCall1(xbest)
        self.obj0 = obj_init.copy()
        # self.obj0 = obj_init.copy()*0.0 + 1.0
        fbest = 1.0
        print("Normalized wrt = ", self.obj0)
                
        boundsWeights = Bounds([0.] * len(xbest), [1.] * len(xbest))  
        
        if ((algorithm == 'MMA') or (algorithm == 'GCMMA')) :
            res = custom_minimize(self.objectiveCall1,x0=xbest, bounds=boundsWeights,method=algorithm,callback=self.callbackFun,\
                options={'maxiter':50,'disp':True,'move_limit':0.05,'maxfun':500,'kkttol':tol,'miniter':10}) # 8-i
        else:
            res = minimize(self.objectiveCall1,xbest, method=algorithm, jac=True, bounds=boundsWeights,callback=self.callbackFun,\
                options={'maxiter':50,'disp':True,'maxfun':500,'ftol':tol})#,'gtol':1e-6,'xtol':1e-3,'esp':1e-2})
     
        xbest = res.x
            
        # this sets the ui and Fi and A->area of best
        fbest,_ = self.objectiveCall1(xbest) # dont take values from memory-new analysis
      
        print("Solution = ",torch.from_numpy(xbest))
        print("Opt Obj/Obj0 = ",torch.from_numpy(fbest))
        print("Opt Obj = ",torch.from_numpy(fbest*self.obj0))

        self.obj0 = fbest*self.obj0;
        
        plt.figure(2)
        self.plotIterVsObj()
        if fileSaveLoc != None:
            plt.savefig(fileSaveLoc+'/IterVobj.png')
        
        return xbest, fbest
    
    def plotFDcurves(self,ui,Fi,labelTag='',TargetOnly=False,fileSaveName=None,fig=plt.figure(),hold=False,Normalize=False):
        # dofMax = np.where(np.abs(ui[-1,:].detach().cpu().numpy()) == np.max(np.abs(ui[-1,:].detach().cpu().numpy())))
        if hold:
            print("hold fig is on.")
        else:
            plt.ion()
            plt.clf()
            
        if Normalize:
            u_norm = self.uinput[-1]
        else:
            u_norm = 1.0
            
        if TargetOnly:
            plt.plot(self.uinput/u_norm,self.Finput,linestyle = 'solid',linewidth=3,label='Target',color='red')
        else:
            
            FF, UU = Fi.cpu().detach(),ui.cpu().detach()
               
             # Linear interpolation function with boundary handling (out of the bounds handling)
            FiSumN,uiMeanN = linear_interpolation_torch(UU,FF,self.uinput.reshape(-1)) # N numbers = self.uinput size        
        
            if fileSaveName != None: 
                if "OptCurve_Phase1" in fileSaveName:
                    print('UU_OptCurve_Phase1 = ',UU);print('FF_OptCurve_Phase1 = ',FF);
                elif "OptCurve_Phase2" in fileSaveName:
                    print('UU_OptCurve_Phase2 = ',UU);print('FF_OptCurve_Phase2 = ',FF);
                elif "OptCurve_FEM" in fileSaveName:
                    print('UU_OptCurve_FEM = ',UU);print('FF_OptCurve_FEM = ',FF);
                elif 'InitialGuess' in fileSaveName:
                    print('UU_Init_Iter = ',UU);print('FF_Init_Iter = ',FF);
            plt.plot(self.uinput/u_norm,self.Finput,linestyle = 'solid',linewidth=3,label='Target',color='red')
            plt.plot(UU/u_norm,FF, linestyle = 'solid',linewidth=3,color = 'black')
            plt.plot(uiMeanN/u_norm,FiSumN, linestyle = 'none',marker='o',label=labelTag,markersize=10,color = 'black')
            
            # x_line = np.zeros(2)
            # y_line = np.zeros(2)
            # for i in range(len(self.uinput)):
            #     x_line[0] = self.uinput[i]/u_norm
            #     x_line[1] = x_line[0]
            #     y_line[0] = self.Finput[i]
            #     y_line[1] = FiSumN[i]
            #     plt.plot(x_line,y_line,linestyle ='solid',linewidth=1,color='b')
            
            
        plt.legend(loc='upper left',fontsize=16)
        # plt.axis('auto')
        if Normalize:
            plt.xlabel('Normalized displacement',fontsize=16)
        else:
            plt.xlabel('Displacement in mm',fontsize=16)

        plt.ylabel('Force in N',fontsize=16)
        plt.title('Force Displacement curve',fontsize=18)
        # plt.pause(1)
        # plt.xlim([0,3.2])
        if TargetOnly:
            maxF =  self.Finput[-1].numpy().reshape(-1) * 1.4
        else:
            maxF =  np.max(np.concatenate((FiSumN.numpy().reshape(-1), self.Finput[-1].numpy().reshape(-1)))) * 1.4
        plt.ylim([None, maxF])

        if fileSaveName != None:
            plt.savefig(fileSaveName)
        fig.show()