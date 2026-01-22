
// Structure of our compute shader
// Group 0: Bindings for data
// Binding 0: Matrix W (Weights) - N x N flattened array
// Binding 1: Vector X (Current Neuron Values) - N length array
// Binding 2: Vector B (Biases) - N length array
// Binding 3: Vector Y (Output Neuron Values) - N length array
// Binding 4: Dimensions Uniform - Struct { size: u32 }

struct Dimensions {
    size: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dimensions;

// Activation Functions
fn tanh_approx(x: f32) -> f32 {
    let e2x = exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let batch = global_id.z;
    let size = dims.size;

    if (row >= size) {
        return;
    }

    // Dot product: Row of W * Vector X
    var sum: f32 = 0.0;
    
    // Batch offset for input/output
    let batch_offset = batch * size;

    for (var col: u32 = 0u; col < size; col = col + 1u) {
        // W is shared (not batched): weights[row * size + col]
        let w_idx = row * size + col;
        
        // Input is batched: input[batch * size + col]
        let input_idx = batch_offset + col;
        
        sum = sum + (weights[w_idx] * input[input_idx]);
    }

    // Add Bias (Shared)
    sum = sum + biases[row];

    // Activation
    // Output is batched: output[batch * size + row]
    let out_idx = batch_offset + row;
    output[out_idx] = tanh_approx(sum);
}
