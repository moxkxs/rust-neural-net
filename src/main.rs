use mnist::MnistBuilder;
use ndarray::{Array1, Array2};
use rust_nn::naive::{CostFunction, NaiveNeuralNetwork, Regularization, WeightInitialization};

fn main() {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut training_data = to_dataset(&mnist.trn_img, &mnist.trn_lbl, 50_000, true);
    let test_data = to_dataset(&mnist.tst_img, &mnist.tst_lbl, 10_000, true);

    let arch = Array1::from_vec(vec![784u64, 30, 30, 10]);
    let mut net = NaiveNeuralNetwork::new(
        &arch,
        CostFunction::CrossEntropy,
        Some(Regularization::L2),
        WeightInitialization::Scaled,
    )
    .expect("Failed to build Network");

    net.sgd(&mut training_data, 30, 10, 0.5, 5.0, &test_data);
}

fn to_dataset(
    images: &[u8],
    labels: &[u8],
    count: usize,
    one_hot: bool,
) -> Vec<(Array2<f64>, Array2<f64>)> {
    (0..count)
        .map(|i| {
            let img = Array2::from_shape_vec(
                (784, 1),
                images[i * 784..(i + 1) * 784]
                    .iter()
                    .map(|&x| x as f64 / 255.0)
                    .collect(),
            )
            .unwrap();

            // Label: digit -> (10, 1) one-hot column vector
            let label = if one_hot {
                let mut lbl = Array2::<f64>::zeros((10, 1));
                lbl[[labels[i] as usize, 0]] = 1.0;
                lbl
            } else {
                Array2::from_shape_vec((1, 1), vec![labels[i] as f64]).unwrap()
            };

            (img, label)
        })
        .collect()
}
