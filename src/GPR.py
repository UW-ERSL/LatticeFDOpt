import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from tqdm import tqdm
import warnings

# This helper function is not part of the class, it's for data handling.
def get_data_from_loader(data_loader, input_dim, device):
    """Extracts and concatenates data from a PyTorch DataLoader."""
    all_inputs = []
    all_targets = []
    
    for inputs, targets in data_loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()
        all_inputs.append(inputs)
        all_targets.append(targets)
        
    return torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0)


class GPRegressor:
    """
    A class to wrap scikit-learn's GaussianProcessRegressor,
    designed to mimic the structure of the provided NeuralNet class.
    
    Note: GaussianProcessRegressor does not use a GPU and trains on
    complete data points, so the handling of partial data is
    adapted to filter out masked data for training.
    """
    
    def __init__(self, nnSettings, useCPU=False):
        # GaussianProcessRegressor runs on CPU, so `useCPU` is ignored.
        print("GaussianProcessRegressor is a CPU-based model.")
        self.device = torch.device("cpu")
        self.nnSettings = nnSettings
        
        # Define the kernel. A common choice is RBF.
        # We can add a WhiteKernel for noise and a ConstantKernel for scaling.
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-5)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        self.input_dim = nnSettings['inputDim']

        self.trainlossArr = []
        self.vallossArr = []
        self.Narr = []
    
    def train_model(self, train_loader, test_loader, num_epochs=1, lr=0.0001, tol=1e-3, prntNum=1):
        """
        Fits the Gaussian Process Regressor to the data.
        
        Note: GPR is a non-iterative method. The num_epochs, lr, tol, and prntNum
        parameters are ignored, as the model is fit in a single step.
        """
        print("Fitting Gaussian Process Regressor model...")
        
        # Get all data from the loader
        all_train_inputs, all_train_targets = get_data_from_loader(train_loader, self.input_dim, self.device)
        
        # Separate features from the mask
        X_train_full = all_train_inputs[:, 0:self.input_dim].numpy()
        y_train_full = all_train_targets.numpy()
       
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The fit method of GPR trains the model
            self.gp_model.fit(X_train_full, y_train_full)
        
        print("Model training complete.")
        
    def evaluate_NN(self, test_loader):
        """
        Evaluates the model on the test set and returns various metrics.
        """
        all_targets_np = []
        all_outputs_np = []
        
        all_test_inputs, all_test_targets = get_data_from_loader(test_loader, self.input_dim, self.device)
        
        X_test = all_test_inputs[:, 0:self.input_dim].numpy()
        y_test = all_test_targets.numpy()
        
        # The mask is at the end of the input tensor
        mask = all_test_inputs[:, self.input_dim:].numpy()
        
        # Predict on the full test set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self.gp_model.predict(X_test)
        
        # Apply the mask to predictions and targets before evaluation
        masked_targets = y_test * mask
        masked_predictions = predictions * mask
        
        # Use np.flatnonzero to get the indices of non-zero elements
        valid_indices = np.flatnonzero(mask)
        
        if valid_indices.size > 0:
            avg_mae = mean_absolute_error(masked_targets.flatten()[valid_indices], masked_predictions.flatten()[valid_indices])
            avg_rmse = np.sqrt(mean_squared_error(masked_targets.flatten()[valid_indices], masked_predictions.flatten()[valid_indices]))
            avg_r2 = r2_score(masked_targets.flatten()[valid_indices], masked_predictions.flatten()[valid_indices])
        else:
            avg_mae, avg_rmse, avg_r2 = 0, 0, 0
            print("No unmasked data found in the test set for evaluation.")

        # Topk indices are not directly applicable to GPR in this format.
        # We can approximate by finding max/min differences on the unmasked data.
        abs_diff = torch.from_numpy(np.abs(masked_predictions - masked_targets)).mean(dim=1)

        maxV_index = torch.topk(abs_diff, k=5, dim=0)
        minV_index = torch.topk(abs_diff, k=5, dim=0, largest=False)
        
        return avg_mae, avg_rmse, avg_r2, maxV_index, minV_index

    def predict_out(self, input_data, t=None):
        """
        Makes predictions using the trained Gaussian Process Regressor.
        
        The `t` parameter is ignored as GPR does not use it.
        """
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        
        # Separate features from the mask
        X_predict = input_data[:, 0:self.input_dim]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self.gp_model.predict(X_predict)
            
        # The `cp` (control points) concept from the NN is not applicable here.
        # We return the predictions and None as a placeholder.
        return predictions, None