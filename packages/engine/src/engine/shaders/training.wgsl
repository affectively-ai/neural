struct Dimensions {
    size: u32,
    batchSize: u32,
}

struct TrainingParams {
    learningRate: f32,
}

@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> values: array<f32>; // Batched Activations (N * B)
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> deltas: array<f32>; // Batched Deltas (N * B)
@group(0) @binding(4) var<storage, read> targets: array<f32>; // Batched Targets
@group(0) @binding(5) var<uniform> dims: Dimensions;
@group(0) @binding(6) var<uniform> params: TrainingParams;

fn tanh_derivative(val: f32) -> f32 {
    return 1.0 - (val * val);
}

// 1. Calculate Deltas (Backward Pass) - 3D Dispatched (64, 1, B)
@compute @workgroup_size(64, 1, 1)
fn calculate_deltas(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let batch = global_id.z;
    let size = dims.size;
    
    if (index >= size) { return; }

    let batch_offset = batch * size;
    let neuron_idx = batch_offset + index;

    let activation = values[neuron_idx];
    let derivative = tanh_derivative(activation);
    
    var error_sum: f32 = 0.0;
    
    // Backpropagate error from "Next Layer" (all other neurons k)
    // For each k (destination), we need delta_k.
    // delta_k is also batched! deltas[batch * size + k]
    for (var k: u32 = 0u; k < size; k = k + 1u) {
        // Weight FROM index TO k
        let w_idx = k * size + index; 
        let weight_ki = weights[w_idx];
        
        let delta_k_idx = batch_offset + k;
        let delta_k = deltas[delta_k_idx];
        
        error_sum = error_sum + (delta_k * weight_ki);
    }

    // Add immediate error (MSE derivative: y - t)
    // targets[batch * size + index]
    let target = targets[neuron_idx];
    if (target > -998.0) {
        error_sum = error_sum + (activation - target);
    }

    deltas[neuron_idx] = error_sum * derivative;
}

// 2. Update Weights (Optimizer Step) - 1D Dispatched (64, 1, 1) - Accumulates Gradients over Batch
@compute @workgroup_size(64)
fn update_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x; // Target neuron
    let size = dims.size;
    let batch_size = dims.batchSize;
    
    if (row >= size) { return; }

    let lr = params.learningRate;

    // Update incoming weights to this neuron 'row'
    // W_ji (row, col)
    for (var col: u32 = 0u; col < size; col = col + 1u) {
        let w_idx = row * size + col;
        
        // Accumulate gradient over batch
        var gradient_sum: f32 = 0.0;
        
        for (var b: u32 = 0u; b < batch_size; b = b + 1u) {
            let batch_offset = b * size;
            
            // delta_j (for this batch item)
            let delta_j = deltas[batch_offset + row];
            
            // input_i (activation of source col for this batch item)
            let input_val = values[batch_offset + col];
            
            gradient_sum = gradient_sum + (delta_j * input_val);
        }
        
        // SGD Update (Mean Gradient? Or Sum? Usually Mean for batch)
        // Let's use Sum * (LearningRate / BatchSize) effectively, or just keep LR as is and user adjusts.
        // Standard is Mean Gradient.
        
        let mean_gradient = gradient_sum / f32(batch_size);
        
        weights[w_idx] = weights[w_idx] - (lr * mean_gradient);
    }
}
