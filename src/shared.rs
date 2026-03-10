use ndarray::Array2;

pub fn sigmoid(arr: &Array2<f64>) -> Array2<f64> {
    arr.mapv(|x| 1.0/(1.0 + (-x).exp()))
}

pub fn sigmoid_prime(arr: &Array2<f64>) -> Array2<f64> {
    let s = sigmoid(arr);
    &s * &s.mapv(|x| 1.0 - x)
}

pub fn quadratic_cost_derivative(output_activations: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    output_activations - y
}

pub fn argmax(arr: &Array2<f64>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}
