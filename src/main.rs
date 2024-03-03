mod types;
mod layer;
mod model;

use std::env;

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use types::Matrix;
use model::Model;
use layer::Dense;
use layer::Sigmoid;
use layer::ReLU;

fn generate_xor_data(train_size: usize, test_size: usize) -> (Matrix, Matrix, Matrix, Matrix) {
    let train_x = Matrix::random((train_size, 2), Uniform::new(0.0, 1.0));
    let train_y = Matrix::from_shape_fn((train_size, 1), |(i, _)| {
        let x = train_x[[i, 0]];
        let y = train_x[[i, 1]];
        if (x > 0.5 && y > 0.5) || (x < 0.5 && y < 0.5) {
            0.0
        } else {
            1.0
        }
    });

    let test_x = Matrix::random((test_size, 2), Uniform::new(0.0, 1.0));
    let test_y = Matrix::from_shape_fn((test_size, 1), |(i, _)| {
        let x = test_x[[i, 0]];
        let y = test_x[[i, 1]];
        if (x > 0.5 && y > 0.5) || (x < 0.5 && y < 0.5) {
            0.0
        } else {
            1.0
        }
    });

    (train_x, train_y, test_x, test_y)
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let (train_x, train_y, test_x, test_y) = generate_xor_data(300, 300);

    let mut model = Model::new(vec![
        Box::new(Dense::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Dense::new(4, 1)),
        Box::new(Sigmoid::new()),
    ]);

    std::fs::remove_dir_all("out/train").unwrap();
    std::fs::remove_dir_all("out/test").unwrap();
    model.train(&train_x, &train_y, 500, 8, 0.01);
    model.record(&test_x, &test_y, "test/result");

    // show test result
    let mut correct = 0;
    for (x, y) in test_x.axis_iter(Axis(0)).zip(test_y.axis_iter(Axis(0))) {
        let x = x.to_shape((1, 2)).unwrap().to_owned();
        let y = y.to_owned();
        let out = model.predict(&x);
        println!("Input: {:.1?} ^ {:.1?}, Output: {:.1?}, Expected: {:.1?}", x[[0, 0]].round(), x[[0, 1]].round(), out[[0, 0]].round(), y[0]);
        correct += (out[[0, 0]].round() == y[0]) as usize;
    }

    let loss = model.loss(&test_x, &test_y);
    println!("Loss: {:.4?}", loss.sum());

    println!("Correct: {:?}/{:?}", correct, test_y.len());
}
