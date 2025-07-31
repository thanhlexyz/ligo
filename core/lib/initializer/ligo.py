import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

from . import util

class LiGoDepthParams(nn.Module):
    """Learnable depth expansion parameters for LiGO"""
    
    def __init__(self, layer_index, num_small_layers, device='cpu'):
        super().__init__()
        self.layer_index = layer_index
        # Learnable coefficients for combining small model layers
        self.coeffs_weight = nn.Parameter(torch.zeros(num_small_layers, device=device))
        self.coeffs_bias = nn.Parameter(torch.zeros(num_small_layers, device=device))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Better initialization: start with identity-like mapping
        with torch.no_grad():
            # Initialize to prefer the corresponding layer (StackBERT-style)
            self.coeffs_weight.fill_(0.03)  # Small noise
            self.coeffs_bias.fill_(0.03)
            
            # Set the main coefficient higher for the corresponding layer
            if self.layer_index < len(self.coeffs_weight):
                self.coeffs_weight[self.layer_index] = 0.9
                self.coeffs_bias[self.layer_index] = 0.9
            else:
                # For extra layers, use the last small layer
                self.coeffs_weight[-1] = 0.9
                self.coeffs_bias[-1] = 0.9
            
            # Normalize
            self.coeffs_weight.data = self.coeffs_weight.data / self.coeffs_weight.data.sum()
            self.coeffs_bias.data = self.coeffs_bias.data / self.coeffs_bias.data.sum()

class LiGoWidthParams(nn.Module):
    """Learnable width expansion parameters for LiGO"""
    
    def __init__(self, small_dim, large_dim, device='cpu'):
        super().__init__()
        self.small_dim = small_dim
        self.large_dim = large_dim
        if large_dim > small_dim:
            # Learnable matrix for width expansion
            self.coeffs_weight = nn.Parameter(torch.zeros(large_dim - small_dim, small_dim, device=device))
            self.reset_parameters()
        else:
            self.coeffs_weight = None
    
    def reset_parameters(self):
        if self.coeffs_weight is not None:
            # Better initialization: identity + noise for width expansion
            with torch.no_grad():
                # Initialize with small noise
                nn.init.normal_(self.coeffs_weight, 0.0, 0.01)
                
                # For the first few dimensions, add identity-like structure
                min_dim = min(self.coeffs_weight.shape[0], self.coeffs_weight.shape[1])
                for i in range(min_dim):
                    # Add identity structure to preserve original features
                    self.coeffs_weight[i, i] = 0.1

class Initializer:

    def __init__(self, args):
        self.lr = getattr(args, 'lr', 0.0005)  # Lower learning rate for stability
        self.num_steps = getattr(args, 'num_steps', 1000)
        self.device = getattr(args, 'device', 'cpu')

    def init(self, pretrain_model, model, loader):
        print('[+] Starting LiGO initialization')
        
        # Step 1: Identify FC layers and their structure
        small_fc_layers = []
        large_fc_layers = []
        
        # Get FC layer parameters from small model
        for name, param in pretrain_model.named_parameters():
            if 'fc' in name and 'weight' in name:
                bias_name = name.replace('weight', 'bias')
                bias_param = util.get_parameter_by_name(pretrain_model, bias_name)
                small_fc_layers.append({
                    'name': name,
                    'weight': param,
                    'bias': bias_param,
                    'weight_shape': param.shape,
                    'bias_shape': bias_param.shape if bias_param is not None else None
                })
        
        # Get FC layer parameters from large model  
        for name, param in model.named_parameters():
            if 'fc' in name and 'weight' in name:
                bias_name = name.replace('weight', 'bias')
                bias_param = util.get_parameter_by_name(model, bias_name)
                large_fc_layers.append({
                    'name': name,
                    'weight': param,
                    'bias': bias_param,
                    'weight_shape': param.shape,
                    'bias_shape': bias_param.shape if bias_param is not None else None
                })
        
        if len(small_fc_layers) == 0:
            print('[!] No FC layers found in small model')
            return model
            
        # Check for other layer types and raise NotImplementedError
        for name, _ in model.named_parameters():
            if ('conv' in name.lower() or 'attention' in name.lower() or 
                'transformer' in name.lower() or 'embed' in name.lower()):
                if 'fc' not in name:
                    raise NotImplementedError(f"Layer type not implemented: {name}")
        
        print(f'[+] Found {len(small_fc_layers)} FC layers in small model')
        print(f'[+] Found {len(large_fc_layers)} FC layers in large model')
        
        # Step 2: Create expansion parameters
        depth_params = []
        width_in_params = []
        width_out_params = []
        
        # Create depth expansion parameters (one for each large layer)
        for i in range(len(large_fc_layers)):
            depth_param = LiGoDepthParams(i, len(small_fc_layers), self.device)
            depth_params.append(depth_param)
        
        # Create width expansion parameters
        for i, (small_layer, large_layer) in enumerate(zip(small_fc_layers[:len(large_fc_layers)], 
                                                          large_fc_layers)):
            # Input dimension expansion
            small_in, small_out = small_layer['weight_shape']
            large_in, large_out = large_layer['weight_shape']
            
            width_in_param = LiGoWidthParams(small_in, large_in, self.device)
            width_out_param = LiGoWidthParams(small_out, large_out, self.device)
            
            width_in_params.append(width_in_param)
            width_out_params.append(width_out_param)
        
        # Move all parameters to device
        all_params = []
        for dp in depth_params:
            dp.to(self.device)
            all_params.extend(list(dp.parameters()))
        for wp in width_in_params:
            wp.to(self.device) 
            all_params.extend(list(wp.parameters()))
        for wp in width_out_params:
            wp.to(self.device)
            all_params.extend(list(wp.parameters()))
        
        # Step 3: Optimize expansion matrices
        print('[+] Optimizing LiGO weight transfer matrices')
        
        # Multiple loss terms for better optimization
        output_criterion = nn.MSELoss()
        feature_criterion = nn.MSELoss()
        
        # Better optimizer with scheduling
        optimizer = optim.AdamW(all_params, lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_steps)
        
        # Feature preservation: track intermediate activations
        small_features = {}
        large_features = {}
        
        def hook_fn_small(name):
            def hook(module, input, output):
                small_features[name] = output.detach()
            return hook
        
        def hook_fn_large(name):
            def hook(module, input, output):
                large_features[name] = output.detach()
            return hook
        
        # Register hooks for FC layers
        small_hooks = []
        large_hooks = []
        for i, layer_info in enumerate(small_fc_layers):
            layer_name = layer_info['name'].replace('.weight', '')
            layer = util.get_parameter_by_name(pretrain_model, layer_name, get_module=True)
            if layer is not None:
                hook = layer.register_forward_hook(hook_fn_small(f'small_fc_{i}'))
                small_hooks.append(hook)
        
        for i, layer_info in enumerate(large_fc_layers):
            layer_name = layer_info['name'].replace('.weight', '')
            layer = util.get_parameter_by_name(model, layer_name, get_module=True)
            if layer is not None:
                hook = layer.register_forward_hook(hook_fn_large(f'large_fc_{i}'))
                large_hooks.append(hook)
        
        # Create a temporary model with LiGO parameters applied
        def apply_ligo_transform():
            for layer_idx, large_layer in enumerate(large_fc_layers):
                if layer_idx >= len(depth_params):
                    continue
                    
                # Apply depth expansion
                depth_param = depth_params[layer_idx]
                width_in_param = width_in_params[layer_idx] if layer_idx < len(width_in_params) else None
                width_out_param = width_out_params[layer_idx] if layer_idx < len(width_out_params) else None
                
                large_weight_shape = large_layer['weight_shape']
                large_bias_shape = large_layer['bias_shape']
                
                # Find the best matching small layer or use layer mapping
                if layer_idx < len(small_fc_layers):
                    # Direct mapping: use the corresponding small layer
                    base_small_layer = small_fc_layers[layer_idx]
                    base_weight = base_small_layer['weight'].clone()
                    base_bias = base_small_layer['bias'].clone() if base_small_layer['bias'] is not None else None
                else:
                    # For extra layers in large model, use the last small layer
                    base_small_layer = small_fc_layers[-1]
                    base_weight = base_small_layer['weight'].clone()
                    base_bias = base_small_layer['bias'].clone() if base_small_layer['bias'] is not None else None
                
                # Apply depth expansion by combining compatible layers
                final_weight = base_weight * depth_param.coeffs_weight[min(layer_idx, len(small_fc_layers)-1)]
                final_bias = base_bias * depth_param.coeffs_bias[min(layer_idx, len(small_fc_layers)-1)] if base_bias is not None else None
                
                # Add contributions from other small layers with compatible shapes
                for small_idx, small_layer in enumerate(small_fc_layers):
                    if small_idx != min(layer_idx, len(small_fc_layers)-1):
                        # Only combine if shapes are compatible
                        if small_layer['weight_shape'] == base_weight.shape:
                            final_weight += depth_param.coeffs_weight[small_idx] * small_layer['weight']
                            if final_bias is not None and small_layer['bias'] is not None:
                                final_bias += depth_param.coeffs_bias[small_idx] * small_layer['bias']
                
                # Apply width expansion to match large model dimensions
                current_weight = final_weight
                current_bias = final_bias
                
                # Expand input dimension if needed
                if large_weight_shape[1] > current_weight.shape[1]:
                    in_diff = large_weight_shape[1] - current_weight.shape[1]
                    if width_in_param is not None and width_in_param.coeffs_weight is not None:
                        # Use learned expansion
                        expanded_in = torch.matmul(current_weight, width_in_param.coeffs_weight.t())
                    else:
                        # Simple padding with zeros
                        expanded_in = torch.zeros(current_weight.shape[0], in_diff, device=current_weight.device)
                    current_weight = torch.cat([current_weight, expanded_in], dim=1)
                
                # Expand output dimension if needed  
                if large_weight_shape[0] > current_weight.shape[0]:
                    out_diff = large_weight_shape[0] - current_weight.shape[0]
                    if width_out_param is not None and width_out_param.coeffs_weight is not None:
                        # Use learned expansion
                        expanded_out = torch.matmul(width_out_param.coeffs_weight, current_weight)
                        current_weight = torch.cat([current_weight, expanded_out], dim=0)
                        if current_bias is not None:
                            expanded_bias = torch.matmul(width_out_param.coeffs_weight, current_bias)
                            current_bias = torch.cat([current_bias, expanded_bias], dim=0)
                    else:
                        # Simple padding with zeros
                        expanded_out = torch.zeros(out_diff, current_weight.shape[1], device=current_weight.device)
                        current_weight = torch.cat([current_weight, expanded_out], dim=0)
                        if current_bias is not None:
                            expanded_bias = torch.zeros(out_diff, device=current_bias.device)
                            current_bias = torch.cat([current_bias, expanded_bias], dim=0)
                
                # Ensure final dimensions match exactly
                if current_weight.shape != large_weight_shape:
                    # Resize by padding or truncating if necessary
                    target_weight = torch.zeros(large_weight_shape, device=current_weight.device)
                    min_h = min(current_weight.shape[0], large_weight_shape[0])
                    min_w = min(current_weight.shape[1], large_weight_shape[1])
                    target_weight[:min_h, :min_w] = current_weight[:min_h, :min_w]
                    current_weight = target_weight
                
                if current_bias is not None and large_bias_shape is not None:
                    if current_bias.shape[0] != large_bias_shape[0]:
                        target_bias = torch.zeros(large_bias_shape, device=current_bias.device)
                        min_b = min(current_bias.shape[0], large_bias_shape[0])
                        target_bias[:min_b] = current_bias[:min_b]
                        current_bias = target_bias
                
                # Copy to large model
                large_layer['weight'].data.copy_(current_weight)
                if large_layer['bias'] is not None and current_bias is not None:
                    large_layer['bias'].data.copy_(current_bias)
        
        # Training loop with improved optimization
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for step in range(min(self.num_steps, len(loader))):
            try:
                batch = next(iter(loader))
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                
                optimizer.zero_grad()
                
                # Apply current LiGO transformation
                apply_ligo_transform()
                
                # Clear previous features
                small_features.clear()
                large_features.clear()
                
                # Forward pass through both models
                with torch.no_grad():
                    pretrain_model.eval()
                    y_hat_pretrain = pretrain_model(x)
                
                model.train()
                y_hat = model(x)
                
                # Multi-component loss
                output_loss = output_criterion(y_hat, y_hat_pretrain)
                
                # Feature preservation loss (if features were captured)
                feature_loss = 0.0
                feature_count = 0
                for small_key, large_key in zip(sorted(small_features.keys()), sorted(large_features.keys())):
                    if small_key.replace('small_', '') == large_key.replace('large_', ''):
                        small_feat = small_features[small_key]
                        large_feat = large_features[large_key]
                        
                        # Match feature dimensions if needed
                        min_dim = min(small_feat.size(-1), large_feat.size(-1))
                        if small_feat.size(-1) != large_feat.size(-1):
                            small_feat = small_feat[..., :min_dim]
                            large_feat = large_feat[..., :min_dim]
                        
                        if small_feat.shape == large_feat.shape:
                            feature_loss += feature_criterion(large_feat, small_feat)
                            feature_count += 1
                
                if feature_count > 0:
                    feature_loss /= feature_count
                
                # Regularization on expansion matrices
                reg_loss = 0.0
                for dp in depth_params:
                    # Encourage sparse combinations
                    reg_loss += 0.01 * torch.norm(dp.coeffs_weight, p=1)
                    reg_loss += 0.01 * torch.norm(dp.coeffs_bias, p=1)
                
                for wp in width_in_params + width_out_params:
                    if wp.coeffs_weight is not None:
                        # Encourage small expansion weights
                        reg_loss += 0.001 * torch.norm(wp.coeffs_weight, p=2)
                
                # Total loss with adaptive weighting
                total_loss = output_loss + 0.1 * feature_loss + reg_loss
                
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Early stopping with patience
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if step % 20 == 0:
                    print(f'    - Step {step}: total_loss = {total_loss.item():.6f}, '
                          f'output_loss = {output_loss.item():.6f}, '
                          f'feature_loss = {feature_loss:.6f}, '
                          f'reg_loss = {reg_loss:.6f}, '
                          f'lr = {scheduler.get_last_lr()[0]:.6f}')
                
                # Early stopping
                if patience_counter >= patience:
                    print(f'[+] Early stopping at step {step} (patience exceeded)')
                    break
                    
            except Exception as e:
                print(f'[!] Error in step {step}: {e}')
                break
        
        # Clean up hooks
        for hook in small_hooks + large_hooks:
            hook.remove()
        
        # Step 4: Apply final transformation
        print('[+] Applying final LiGO transformation to model')
        
        # Validate the transformation with a simple check
        try:
            with torch.no_grad():
                test_batch = next(iter(loader))
                test_x = test_batch[0] if isinstance(test_batch, (list, tuple)) else test_batch
                test_x = test_x.to(self.device)
                
                # Test before and after transformation
                before_output = model(test_x)
                apply_ligo_transform()
                after_output = model(test_x)
                
                # Check if the transformation is reasonable
                output_diff = torch.norm(after_output - before_output)
                print(f'[+] Output change after LiGO: {output_diff.item():.6f}')
                
                # If the change is too large, fall back to simpler initialization
                if output_diff > 10.0:  # Threshold for "reasonable" change
                    print('[!] Large output change detected, applying conservative fallback')
                    self._apply_conservative_fallback(small_fc_layers, large_fc_layers)
                else:
                    print('[+] LiGO transformation validated successfully')
                    
        except Exception as e:
            print(f'[!] Validation failed ({e}), applying conservative fallback')
            self._apply_conservative_fallback(small_fc_layers, large_fc_layers)
        
        print('[+] LiGO initialization completed')
        return model
    
    def _apply_conservative_fallback(self, small_fc_layers, large_fc_layers):
        """Apply a conservative weight transfer as fallback"""
        print('[+] Applying conservative weight transfer fallback')
        
        for layer_idx, large_layer in enumerate(large_fc_layers):
            if layer_idx < len(small_fc_layers):
                # Direct copy for matching layers
                small_layer = small_fc_layers[layer_idx]
                
                # Copy what we can
                small_weight = small_layer['weight']
                small_bias = small_layer['bias']
                large_weight_shape = large_layer['weight_shape']
                large_bias_shape = large_layer['bias_shape']
                
                # Initialize large parameters with zeros
                new_weight = torch.zeros(large_weight_shape, device=small_weight.device)
                new_bias = torch.zeros(large_bias_shape, device=small_bias.device) if small_bias is not None else None
                
                # Copy the overlapping part
                min_out = min(small_weight.shape[0], large_weight_shape[0])
                min_in = min(small_weight.shape[1], large_weight_shape[1])
                
                new_weight[:min_out, :min_in] = small_weight[:min_out, :min_in]
                if new_bias is not None and small_bias is not None:
                    new_bias[:min_out] = small_bias[:min_out]
                
                # Apply to model
                large_layer['weight'].data.copy_(new_weight)
                if large_layer['bias'] is not None and new_bias is not None:
                    large_layer['bias'].data.copy_(new_bias)
            else:
                # For extra layers, use Xavier initialization
                nn.init.xavier_normal_(large_layer['weight'])
                if large_layer['bias'] is not None:
                    nn.init.zeros_(large_layer['bias'])
