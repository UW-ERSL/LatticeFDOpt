from scipy.interpolate import RBFInterpolator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import numpy as np
from tqdm import tqdm
import warnings
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

class RBFRegressor:
    """
    Replaces GPR with RBF interpolation using SciPy's RBFInterpolator.
    This method is deterministic and does not model uncertainty.
    """

    def __init__(self, nnSettings, useCPU=False):
        print("Using RBF interpolation (not probabilistic).")
        self.device = torch.device("cpu")
        self.nnSettings = nnSettings
        self.input_dim = nnSettings['inputDim']
        self.rbf_model = None

    def train_model(self, train_loader, test_loader=None, num_epochs=1, lr=0.0001, tol=1e-3, prntNum=1):
        all_train_inputs, all_train_targets = get_data_from_loader(train_loader, self.input_dim, self.device)
        X_train = all_train_inputs[:, :self.input_dim].numpy()
        y_train = all_train_targets.numpy()

        # Fit the RBF interpolator (supports multidimensional y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.rbf_model = RBFInterpolator(X_train, y_train, kernel='thin_plate_spline',epsilon=1.0)

    def evaluate_NN(self, test_loader):
        all_test_inputs, all_test_targets = get_data_from_loader(test_loader, self.input_dim, self.device)
        X_test = all_test_inputs[:, :self.input_dim].numpy()
        y_test = all_test_targets.numpy()
        mask = all_test_inputs[:, self.input_dim:].numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self.rbf_model(X_test)

        masked_targets = y_test * mask
        masked_predictions = predictions * mask

        valid_indices = np.flatnonzero(mask)

        if valid_indices.size > 0:
            avg_mae = mean_absolute_error(masked_targets.flatten()[valid_indices], masked_predictions.flatten()[valid_indices])
            avg_rmse = np.sqrt(mean_squared_error(masked_targets.flatten()[valid_indices], masked_predictions.flatten()[valid_indices]))
            avg_r2 = r2_score(masked_targets.flatten()[valid_indices], masked_predictions.flatten()[valid_indices])
        else:
            avg_mae, avg_rmse, avg_r2 = 0, 0, 0
            print("No unmasked data for evaluation.")

        abs_diff = torch.from_numpy(np.abs(masked_predictions - masked_targets)).mean(dim=1)
        maxV_index = torch.topk(abs_diff, k=5, dim=0)
        minV_index = torch.topk(abs_diff, k=5, dim=0, largest=False)

        return avg_mae, avg_rmse, avg_r2, maxV_index, minV_index

    def predict_out(self, input_data, t=None):
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        X_predict = input_data[:, :self.input_dim]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self.rbf_model(X_predict)
        return predictions, None
