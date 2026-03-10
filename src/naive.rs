use crate::shared::*;
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand::rng;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::*;

pub enum CostFunction {
    Quadratic,
    CrossEntropy,
}

pub enum Regularization {
    L1,
    L2,
}

pub enum WeightInitialization {
    Random,
    Scaled,
}

pub struct NaiveNeuralNetwork {
    pub architecture: Array1<u64>,
    pub num_layers: usize,
    pub biases: Vec<Array2<f64>>,
    pub weights: Vec<Array2<f64>>,
    pub cost_func: CostFunction,
    pub regularization: Option<Regularization>,
    pub weight_initialization: WeightInitialization,
}

impl NaiveNeuralNetwork {
    pub fn new(
        nn_arch: &Array1<u64>,
        cost_func: CostFunction,
        regularization: Option<Regularization>,
        weight_initialization: WeightInitialization,
    ) -> Result<NaiveNeuralNetwork, &'static str> {
        let n = nn_arch.len();

        let biases: Vec<Array2<f64>> = nn_arch
            .iter()
            .skip(1)
            .map(|l| {
                Array::random(
                    (*l as usize, 1),
                    rand_distr::Normal::new(0.0, 1.0).expect("biases not initiating"),
                )
            })
            .collect();

        let weights: Vec<Array2<f64>> = match weight_initialization {
            WeightInitialization::Random => nn_arch
                .iter()
                .zip(nn_arch.iter().skip(1))
                .map(|(x, y)| {
                    Array::random(
                        (*y as usize, *x as usize),
                        rand_distr::Normal::new(0.0, 1.0).expect("weights not initiating"),
                    )
                })
                .collect(),
            WeightInitialization::Scaled => nn_arch
                .iter()
                .zip(nn_arch.iter().skip(1))
                .map(|(x, y)| {
                    Array::random(
                        (*y as usize, *x as usize),
                        rand_distr::Normal::new(0.0, 1.0 / (*x as f64).sqrt())
                            .expect("weights not initiating"),
                    )
                })
                .collect(),
        };

        Ok(NaiveNeuralNetwork {
            architecture: nn_arch.clone(),
            num_layers: n,
            biases,
            weights,
            cost_func,
            regularization,
            weight_initialization,
        })
    }

    pub fn feedforward(&self, a: &Array2<f64>) -> Array2<f64> {
        let mut activation = a.clone();

        self.biases
            .iter()
            .zip(self.weights.iter())
            .for_each(|(b, w)| {
                let z = w.dot(&activation) + b;
                activation = sigmoid(&z)
            });

        activation
    }

    pub fn backprop(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|b| Array2::<f64>::zeros(b.dim()))
            .collect();
        let mut nabla_w: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::<f64>::zeros(w.dim()))
            .collect();

        let mut activation = x.clone();
        let mut activations: Vec<Array2<f64>> = vec![x.clone()];

        let mut zs: Vec<Array2<f64>> = Vec::new();

        self.biases
            .iter()
            .zip(self.weights.iter())
            .for_each(|(b, w)| {
                let z = w.dot(&activation) + b;
                activation = sigmoid(&z);
                zs.push(z);
                activations.push(activation.clone());
            });

        let mut delta = match self.cost_func {
            CostFunction::Quadratic => {
                quadratic_cost_derivative(&activations[activations.len() - 1], y)
                    * sigmoid_prime(&zs[zs.len() - 1])
            }
            CostFunction::CrossEntropy => &activations[activations.len() - 1] - y,
        };

        let nb_len = nabla_b.len();
        let nw_len = nabla_w.len();

        nabla_b[nb_len - 1] = delta.clone();
        nabla_w[nw_len - 1] = delta.dot(&activations[activations.len() - 2].t());

        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l];
            let sp = sigmoid_prime(&z);

            delta = self.weights[self.weights.len() - l + 1].t().dot(&delta) * &sp;

            nabla_w[nw_len - l] = delta.dot(&activations[activations.len() - l - 1].t());
            nabla_b[nb_len - l] = delta.clone();
        }

        (nabla_b, nabla_w)
    }

    pub fn update_mini_batch(
        &mut self,
        mini_batch: &[(Array2<f64>, Array2<f64>)],
        eta: f64,
        lmbda: f64,
        n: usize,
    ){
        let mut nabla_b: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|b| Array2::<f64>::zeros(b.dim()))
            .collect();
        let mut nabla_w: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::<f64>::zeros(w.dim()))
            .collect();

        for (x, y) in mini_batch.iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);

            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }

            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
        }

        let scale = eta / mini_batch.len() as f64;
        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter()) {
            match self.regularization {
                Some(Regularization::L1) => {
                    *w = &*w - &(w.mapv(|x| x.signum()) * (eta * lmbda / n as f64)) - &(nw * scale);
                }
                Some(Regularization::L2) => {
                    let weight_decay = 1.0 - eta * lmbda / n as f64;
                    *w = weight_decay * (&*w) - &(nw * scale);
                }
                None => {
                    *w = &*w - &(nw * scale);
                }
            }
        }

        for (b, nb) in self.biases.iter_mut().zip(nabla_b.iter()) {
            let scale = eta / mini_batch.len() as f64;
            *b = &*b - &(nb * scale);
        }
    }

    pub fn sgd(
        &mut self,
        training_data: &mut [(Array2<f64>, Array2<f64>)],
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        lmbda: f64,
        test_data: &[(Array2<f64>, Array2<f64>)],
    ){
        let n = training_data.len();
        let n_test = test_data.len();

        for i in 0..epochs {
            training_data.shuffle(&mut rng());
            for k in (0..n).step_by(mini_batch_size) {
                let mini_batch = &training_data[k..k + mini_batch_size];
                self.update_mini_batch(mini_batch, eta, lmbda, n);
            }
            println!("Epoch {i}: {} / {}", self.evaluate(test_data), n_test);
        }
    }

    pub fn evaluate(&self, test_data: &[(Array2<f64>, Array2<f64>)]) -> usize {
        test_data
            .iter()
            .filter(|(x, y)| argmax(&self.feedforward(x)) == argmax(y))
            .count()
    }
}
