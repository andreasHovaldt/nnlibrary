import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union
from nnlibrary.models import TCN, TCNRegression
from nnlibrary.datasets import MpcDatasetHDF5
from nnlibrary.utils.operations import Standardize
from torch.utils.data import DataLoader


class TCNExplainer:
    """
    Explainability and visualization tools for Temporal Convolutional Networks.
    Provides multiple methods to understand what the model focuses on.
    """
    
    def __init__(self, model: Union[TCN, TCNRegression] , device: str = 'cpu'):
        """
        Initialize the explainer with a trained TCN model.
        
        Args:
            model: Trained TCN or TCNRegression model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def integrated_gradients(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attribution for each feature at each timestep.
        
        Args:
            input_tensor: Input of shape (batch, time, features)
            target_class: Target class for classification (None for regression)
            n_steps: Number of integration steps
            baseline: Baseline input (zeros by default)
            
        Returns:
            Attribution map of shape (batch, time, features)
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        x = input_tensor.to(self.device)
        b = baseline.to(self.device)

        # Alphas for path integral approximation
        alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=self.device)

        attributions = torch.zeros_like(x, device=self.device)

        for alpha in alphas:
            interpolated = b + alpha * (x - b)
            interpolated.requires_grad_(True)  # enable grad for this tensor

            # Forward pass
            output = self.model(interpolated)

            # Select target
            if target_class is not None:
                scalar = output[:, target_class].sum()
            else:
                scalar = output.sum()

            # Compute gradients w.r.t. the interpolated inputs
            grads = torch.autograd.grad(
                outputs=scalar,
                inputs=interpolated,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            attributions += grads / float(n_steps)

        # Scale by (x - baseline)
        integrated_grads = (x - b) * attributions

        return integrated_grads.detach().cpu()
    
    def smooth_grad(
        self, 
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_samples: int = 50,
        noise_level: float = 0.1
    ) -> torch.Tensor:
        """
        Compute SmoothGrad - averages gradients over noisy inputs for smoother attributions.
        
        Args:
            input_tensor: Input of shape (batch, time, features)
            target_class: Target class for classification
            n_samples: Number of noisy samples
            noise_level: Standard deviation of noise relative to input range
            
        Returns:
            Attribution map of shape (batch, time, features)
        """
        x = input_tensor.to(self.device)

        # Calculate noise scale
        input_std = x.std()
        noise_std = noise_level * (input_std if float(input_std) > 0 else 1.0)

        accumulated_grads = torch.zeros_like(x)

        for _ in range(n_samples):
            # Add noise to input
            noise = torch.randn_like(x) * noise_std
            noisy_input = (x + noise).requires_grad_(True)

            # Forward pass
            output = self.model(noisy_input)

            # Select target
            scalar = output[:, target_class].sum() if target_class is not None else output.sum()

            grads = torch.autograd.grad(
                outputs=scalar,
                inputs=noisy_input,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            accumulated_grads += grads

        smooth_grads = accumulated_grads / float(n_samples)

        return smooth_grads.detach().cpu()
    
    def temporal_occlusion_analysis(
        self,
        input_tensor: torch.Tensor,
        window_size: int = 5,
        stride: int = 1,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Analyze importance of temporal segments by occluding them.
        
        Args:
            input_tensor: Input of shape (batch, time, features)
            window_size: Size of occlusion window
            stride: Stride for sliding window
            target_class: Target class for classification
            
        Returns:
            Importance scores for each temporal position
        """
        input_tensor = input_tensor.to(self.device)
        batch_size, seq_len, n_features = input_tensor.shape
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            if target_class is not None:
                baseline_score = baseline_output[:, target_class].mean().item()
            else:
                baseline_score = baseline_output.mean().item()
        
        # Slide occlusion window
        importance_scores = np.zeros(seq_len)
        
        for t in range(0, seq_len - window_size + 1, stride):
            # Create occluded input
            occluded_input = input_tensor.clone()
            occluded_input[:, t:t+window_size, :] = 0
            
            # Get prediction with occlusion
            with torch.no_grad():
                occluded_output = self.model(occluded_input)
                if target_class is not None:
                    occluded_score = occluded_output[:, target_class].mean().item()
                else:
                    occluded_score = occluded_output.mean().item()
            
            # Calculate importance (drop in performance)
            importance = baseline_score - occluded_score
            importance_scores[t:t+window_size] += importance / window_size
        
        return importance_scores
    
    def feature_occlusion_analysis(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Analyze importance of each feature by occluding it across all timesteps.
        
        Args:
            input_tensor: Input of shape (batch, time, features)
            target_class: Target class for classification
            
        Returns:
            Importance scores for each feature
        """
        input_tensor = input_tensor.to(self.device)
        batch_size, seq_len, n_features = input_tensor.shape
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            if target_class is not None:
                baseline_score = baseline_output[:, target_class].mean().item()
            else:
                baseline_score = baseline_output.mean().item()
        
        feature_importance = np.zeros(n_features)
        
        for f in range(n_features):
            # Occlude feature f across all timesteps
            occluded_input = input_tensor.clone()
            occluded_input[:, :, f] = 0
            
            # Get prediction with occlusion
            with torch.no_grad():
                occluded_output = self.model(occluded_input)
                if target_class is not None:
                    occluded_score = occluded_output[:, target_class].mean().item()
                else:
                    occluded_score = occluded_output.mean().item()
            
            # Calculate importance
            feature_importance[f] = baseline_score - occluded_score
        
        return feature_importance
    
    def get_conv_activations(
        self,
        input_tensor: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract convolutional layer activations to see what patterns are detected.
        
        Args:
            input_tensor: Input of shape (batch, time, features)
            layer_idx: Which residual block to extract from (-1 for last)
            
        Returns:
            Activations of shape (batch, channels, time)
        """
        input_tensor = input_tensor.to(self.device)
        x = input_tensor.transpose(1, 2)  # (batch, time, features) -> (batch, features, time)
        
        # Forward through residual blocks
        with torch.no_grad():
            for i, residual_block in enumerate(self.model.tcn_residual_blocks):
                x = residual_block(x)
                if i == layer_idx or (layer_idx == -1 and i == len(self.model.tcn_residual_blocks) - 1):
                    return x.cpu()
        
        return x.cpu()
    
    def visualize_attributions(
        self,
        input_tensor: torch.Tensor,
        attributions: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        sample_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 8),
        title: str = "Feature Attributions Over Time"
    ):
        """
        Create a heatmap visualization of attributions.
        
        Args:
            input_tensor: Original input
            attributions: Attribution scores
            feature_names: Names for features
            sample_idx: Which sample in batch to visualize
            figsize: Figure size
            title: Plot title
        """
        # Select single sample or aggregate across the batch when sample_idx is None
        if sample_idx is None:
            # Average attributions across batch: (B, T, F) -> (T, F)
            attr = attributions.mean(dim=0).cpu().numpy()
        else:
            attr = attributions[sample_idx].cpu().numpy()
        
        seq_len, n_features = attr.shape
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot attributions heatmap
        sns.heatmap(
            attr.T,
            ax=axes[0],
            cmap='RdBu_r',
            center=0,
            xticklabels=10,
            yticklabels=feature_names,
            cbar_kws={'label': 'Attribution Score (|↑| = importance)'}
        )
        batch_note = " (batch mean)" if sample_idx is None else ""
        axes[0].set_title(f"{title}{batch_note} [|higher| = more importance]")
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Features')
        
        # Plot temporal importance (sum across features)
        temporal_importance = np.abs(attr).mean(axis=1)
        axes[1].plot(temporal_importance, linewidth=2)
        axes[1].fill_between(range(seq_len), temporal_importance, alpha=0.3)
        axes[1].set_title('Temporal Importance (Averaged Across Features) [higher = more importance]')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Importance Score')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_feature_importance(
        self,
        feature_importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Visualize feature importance as a bar chart.
        
        Args:
            feature_importance: Importance scores for each feature
            feature_names: Names for features
            figsize: Figure size
        """
        n_features = len(feature_importance)
        
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(n_features)]
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_importance = feature_importance[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['red' if x < 0 else 'green' for x in sorted_importance]
        bars = ax.bar(range(n_features), sorted_importance, color=colors, alpha=0.7)
        
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importance (Higher = More Important)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_comprehensive_report(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        sample_idx: Optional[int] = None
    ):
        """
        Create a comprehensive visualization report combining multiple methods.
        
        Args:
            input_tensor: Input of shape (batch, time, features)
            target_class: Target class for classification
            feature_names: Names for features
            sample_idx: Which sample to visualize
        """
        print("Generating comprehensive explainability report...")
        
        # 1. Integrated Gradients
        print("Computing Integrated Gradients...")
        ig_attributions = self.integrated_gradients(input_tensor, target_class)
        
        # 2. SmoothGrad
        print("Computing SmoothGrad...")
        smooth_attributions = self.smooth_grad(input_tensor, target_class)
        
        # 3. Temporal Occlusion
        print("Computing Temporal Occlusion Analysis...")
        temporal_importance = self.temporal_occlusion_analysis(input_tensor, target_class=target_class)
        
        # 4. Feature Occlusion
        print("Computing Feature Occlusion Analysis...")
        feature_importance = self.feature_occlusion_analysis(input_tensor, target_class)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Prepare safe feature labels based on attribution shape
        def _safe_feature_labels(n: int):
            if feature_names is not None and len(feature_names) == n:
                return list(feature_names)
            elif feature_names is not None and len(feature_names) != n:
                try:
                    print(f"WARN: feature_names length ({len(feature_names)}) does not match number of input features ({n}); using generic labels.")
                except Exception:
                    pass
            return [f"F{i}" for i in range(n)]

        # Integrated Gradients heatmap
        ax1 = plt.subplot(2, 3, 1)
        if sample_idx is None:
            attr = ig_attributions.mean(dim=0).cpu().numpy()  # (T, F)
        else:
            attr = ig_attributions[sample_idx].cpu().numpy()
        n_feat_attr = attr.shape[1]
        sns.heatmap(attr.T, ax=ax1, cmap='RdBu_r', center=0,
                    yticklabels=_safe_feature_labels(n_feat_attr))
        ax1.set_title('Integrated Gradients [|higher| = more importance]' + (' (batch mean)' if sample_idx is None else ''))
        ax1.set_xlabel('Time Step')
        
        # SmoothGrad heatmap
        ax2 = plt.subplot(2, 3, 2)
        if sample_idx is None:
            smooth_attr = smooth_attributions.mean(dim=0).cpu().numpy()
        else:
            smooth_attr = smooth_attributions[sample_idx].cpu().numpy()
        sns.heatmap(smooth_attr.T, ax=ax2, cmap='RdBu_r', center=0,
                    yticklabels=_safe_feature_labels(n_feat_attr))
        ax2.set_title('SmoothGrad [|higher| = more importance]' + (' (batch mean)' if sample_idx is None else ''))
        ax2.set_xlabel('Time Step')
        
        # Combined attribution (average)
        ax3 = plt.subplot(2, 3, 3)
        combined = (attr + smooth_attr) / 2
        sns.heatmap(combined.T, ax=ax3, cmap='RdBu_r', center=0,
                    yticklabels=_safe_feature_labels(n_feat_attr))
        ax3.set_title('Combined Attribution [|higher| = more importance]' + (' (batch mean)' if sample_idx is None else ''))
        ax3.set_xlabel('Time Step')
        
        # Temporal importance from occlusion
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(temporal_importance, linewidth=2, color='blue')
        ax4.fill_between(range(len(temporal_importance)), temporal_importance, alpha=0.3)
        ax4.set_title('Temporal Importance (Occlusion) [higher = more importance]')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Importance')
        ax4.grid(True, alpha=0.3)
        
        # Feature importance (Inputs): ensure label count matches number of input features
        ax5 = plt.subplot(2, 3, 5)
        n_feat_imp = int(len(feature_importance))
        default_labels = [f"F{i}" for i in range(n_feat_imp)]
        labels = default_labels
        if feature_names is not None and len(feature_names) == n_feat_imp:
            labels = list(feature_names)
        elif feature_names is not None and len(feature_names) != n_feat_imp:
            # Provided names don't match input feature count; fall back to generic labels
            try:
                print(f"WARN: feature_names length ({len(feature_names)}) does not match number of input features ({n_feat_imp}); using generic labels.")
            except Exception:
                pass
        topk = min(10, n_feat_imp)
        sorted_idx = np.argsort(feature_importance)[::-1][:topk]  # Top-K
        ax5.barh(range(topk), feature_importance[sorted_idx])
        ax5.set_yticks(range(topk))
        ax5.set_yticklabels([labels[i] for i in sorted_idx])
        ax5.set_title('Top 10 Feature Importance [higher = more importance]')
        ax5.set_xlabel('Importance Score')
        
        # Temporal x Feature importance matrix
        ax6 = plt.subplot(2, 3, 6)
        temporal_feature_importance = np.abs(combined)
        sns.heatmap(temporal_feature_importance.T, ax=ax6, cmap='YlOrRd',
                    yticklabels=_safe_feature_labels(n_feat_attr))
        ax6.set_title('Absolute Importance (Time x Feature) [higher = more importance]')
        ax6.set_xlabel('Time Step')
        
        plt.suptitle('TCN Model Explainability Report', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
        return fig, {
            'integrated_gradients': ig_attributions,
            'smooth_grad': smooth_attributions,
            'temporal_importance': temporal_importance,
            'feature_importance': feature_importance
        }


# Example usage
def example_usage():
    """
    Example of how to use the TCNExplainer with your model.
    """
    # Assuming you have a trained model and some test data
    # model = TCN(input_dim=10, sequence_length=100, num_classes=3, hidden_layer_sizes=[64, 64, 128])
    # model.load_state_dict(torch.load('your_model.pth'))
    
    # Create explainer
    # explainer = TCNExplainer(model)
    
    # Generate sample input (batch_size=1, seq_len=100, features=10)
    # input_data = torch.randn(1, 100, 10)
    
    # Generate comprehensive report
    # fig, results = explainer.create_comprehensive_report(
    #     input_data,
    #     target_class=0,  # For classification, None for regression
    #     feature_names=['Feature1', 'Feature2', ...]  # Optional
    # )
    
    # Individual analysis methods:
    
    # 1. Integrated Gradients
    # ig_attrs = explainer.integrated_gradients(input_data, target_class=0)
    # explainer.visualize_attributions(input_data, ig_attrs)
    
    # 2. Temporal importance
    # temporal_imp = explainer.temporal_occlusion_analysis(input_data)
    
    # 3. Feature importance
    # feature_imp = explainer.feature_occlusion_analysis(input_data)
    # explainer.visualize_feature_importance(feature_imp)
    
    pass

def main():
    dataset_dir = Path("/home/ahojrup/GitLab/mpc_ahu_neural_network/data/730days_2023-09-24_2025-09-23/dataset-regression-mode")
    dataset_metadata = json.loads((dataset_dir / "stats/metadata.json").read_text())
    checkpoint_path = Path("/home/ahojrup/GitLab/mpc_ahu_neural_network/exp/730days_2023-09-24_2025-09-23/TCNRegression/model/model_best.pth")
    class_names = None # [
    #     "fan_speed_cmd_10001",
    #     "fresh_air_damper_cmd_10001",
    #     "setpoint_supply_air_mpc_10001",
    #     "setpoint_heating_mpc_10001",
    #     "setpoint_cooling_mpc_10001",
    # ]
    
    model_args = dict(
        input_dim=dataset_metadata["feature_dim"],
        sequence_length=dataset_metadata["window"],
        num_classes=dataset_metadata["num_classes"],
        regression_head_hidden_dim=64,
        hidden_layer_sizes=[64, 64, 128, 128],
        kernel_size=3,
        dropout=0.3,
        dropout_type="channel",
    )
    
    model = TCNRegression(**model_args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    
    model_explainer = TCNExplainer(model=model)
    
    # Generate sample input (batch_size=1, seq_len=32, features=?)
    standardize_transform = Standardize(
        mean=np.load(dataset_dir / "stats" / "target_mean.npy").astype(float).tolist(),
        std=np.load(dataset_dir / "stats" / "target_std.npy").astype(float).tolist(),
    )

    dataset = MpcDatasetHDF5(
        hdf5_file=dataset_dir / "train.h5",
        task='regression',
        target_transform=standardize_transform,
    )
    X, y = dataset.__getitem__(index=0)
    print(X.shape, y.shape)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=512,
        shuffle=True,
    )
    
    
    input_data = X.unsqueeze(dim=0)
    
    # Generate comprehensive report
    fig, results = model_explainer.create_comprehensive_report(
        input_data,
        target_class=None,  # For classification, None for regression
        feature_names=class_names  # Optional
    )
    
    # Individual analysis methods:
    
    # 1. Integrated Gradients
    ig_attrs = model_explainer.integrated_gradients(input_data, target_class=0)
    model_explainer.visualize_attributions(input_data, ig_attrs)
    
    # 2. Temporal importance
    temporal_imp = model_explainer.temporal_occlusion_analysis(input_data)
    
    # 3. Feature importance
    feature_imp = model_explainer.feature_occlusion_analysis(input_data)
    model_explainer.visualize_feature_importance(feature_imp)


if __name__ == "__main__":
    main()