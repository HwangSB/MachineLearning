use std::io::Write;
use std::path::Path;

use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;

use crate::types::Matrix;
use crate::layer::Layer;

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Model { layers }
    }

    pub fn predict(&mut self, x: &Matrix) -> Matrix {
        self.layers.iter_mut().fold(x.clone(), |input, layer| layer.forward(&input))
    }

    pub fn train(&mut self, x: &Matrix, y: &Matrix, epochs: usize, batch_size: usize, learning_rate: f64) {
        let mut rng = thread_rng();

        for epoch in 0..epochs {
            // shuffle batch indices
            let train_size = x.shape()[0];
            let indices = (0..train_size).collect::<Vec<_>>();
            let batch_indices = indices.as_slice().choose_multiple(&mut rng, train_size).copied().collect::<Vec<_>>();
            let batches = batch_indices.chunks(batch_size);

            // train with mini-batches
            for batch in batches {
                let x_batch = x.select(Axis(0), batch);
                let y_batch = y.select(Axis(0), batch);

                let out = self.predict(&x_batch);

                let dout = &out - &y_batch;
                self.layers.iter_mut().rev().fold(dout, |grad, layer| layer.backward(&grad, learning_rate));
            }

            // record result
            if epoch % 10 == 0 {
                self.record(x, y, &format!("train/result_{:03}", epoch));
            }
        }
    }

    pub fn loss(&mut self, x: &Matrix, y: &Matrix) -> Matrix {
        // mean squared error
        let y_pred = self.predict(x);
        0.5 * (y_pred - y).mapv(|x| x.powi(2))
    }

    pub fn record(&mut self, x: &Matrix, y: &Matrix, output: &str) {
        let path_name = format!("out/{output}");
        let path = Path::new(&path_name);

        if let Some(dir_path) = path.parent() {
            std::fs::create_dir_all(dir_path).unwrap();
        }

        let mut file = std::fs::File::create(path).unwrap();

        /* 
        File format:
            x1 x2 y
            ...
        */
        for (x, _) in x.axis_iter(Axis(0)).zip(y.axis_iter(Axis(0))) {
            let x = x.to_shape((1, 2)).unwrap().to_owned();
            let out = self.predict(&x);
            file.write_all(&(x[[0, 0]].to_string() + " " + &x[[0, 1]].to_string() + " " + &(out[[0, 0]].round() as i32).to_string() + "\n").as_bytes()).unwrap();
        }
    }
}