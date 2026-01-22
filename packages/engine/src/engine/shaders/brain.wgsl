
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let size = dims.size;

    if (row >= size) {
        return;
    }

    // Dot product: Row of W * Vector X
    var sum: f32 = 0.0;
    for (var col: u32 = 0u; col < size; col = col + 1u) {
        // W is flattened, so W[row][col] is weights[row * size + col]
        // Actually, for W * x, if W_ij is weight FROM j TO i.
        // Let's assume W is stored such that weights[i * size + j] is weight FROM j to i.
        
        let w_idx = row * size + col;
        sum = sum + (weights[w_idx] * input[col]);
    }

    // Add Bias
    sum = sum + biases[row];

    // Activation (Hardcoded to Tanh for now, could become a property)
    output[row] = tanh_approx(sum);
}
