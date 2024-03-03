use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

use crate::types::Vector;
use crate::types::Matrix;

pub trait Layer {
    fn forward(&mut self, x: &Matrix) -> Matrix;
    fn backward(&mut self, grad: &Matrix, learning_rate: f64) -> Matrix;
}

pub struct Dense {
    pub weights: Matrix,
    pub biases: Vector,
    x: Matrix,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let std_dev = (2.0 / input_size as f64).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        // Weights are initialized with He initialization
        Dense {
            weights: Array::random((input_size, output_size), normal_dist),
            biases: Array::zeros(output_size),
            x: Array::zeros((1, 1)),
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        self.x = x.clone();
        x.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, grad: &Matrix, learning_rate: f64) -> Matrix {
        let dx = grad.dot(&self.weights.t());
        let dw = self.x.t().dot(grad);
        let db = grad.sum_axis(Axis(0));

        self.weights = &self.weights - learning_rate * &dw;
        self.biases = &self.biases - learning_rate * &db;

        dx
    }
}

pub struct Sigmoid {
    y: Matrix,
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
            y: Array::zeros((1, 1)),
        }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        let y = 1.0 / (1.0 + (-x).mapv(f64::exp));
        self.y = y.clone();
        y
    }

    fn backward(&mut self, grad: &Matrix, _: f64) -> Matrix {
        grad * &self.y * (1.0 - &self.y)
    }
}

pub struct ReLU {
    mask: Matrix,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU {
            mask: Array::zeros((1, 1)),
        }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        self.mask = x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        x.mapv(|x| x.max(0.0))
    }

    fn backward(&mut self, grad: &Matrix, _: f64) -> Matrix {
        grad * &self.mask
    }
}
