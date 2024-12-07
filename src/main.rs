/// A simple Perceptron model for binary classification tasks.
/// 
/// This perceptron takes a fixed number of inputs, each weighted by `weights`,
/// adds a `bias`, and then applies a step activation function to produce an output
/// of either 0 or 1. It can be trained via a basic learning rule to adjust its
/// parameters (`weights` and `bias`).
struct Perceptron {
    /// The weights associated with each input feature.
    weights: Vec<f64>,

    /// A bias term added to the weighted sum before activation.
    bias: f64,

    /// The learning rate controls how quickly the perceptron updates its parameters.
    learning_rate: f64,
}

impl Perceptron {
    /// Creates a new Perceptron with a given number of inputs and a specified learning rate.
    ///
    /// # Arguments
    ///
    /// * `num_inputs` - The number of input features the perceptron will receive.
    /// * `learning_rate` - The rate at which the perceptron updates its parameters.
    ///
    /// # Returns
    ///
    /// A new `Perceptron` instance with zero-initialized weights and bias.
    fn new(num_inputs: usize, learning_rate: f64) -> Self {
        // Initialize weights as a vector of zeros, one for each input.
        let weights = vec![0.0; num_inputs]; 
        // Start with a bias of 0.0.
        let bias = 0.0;
        // Return the new Perceptron.
        Perceptron {
            weights,
            bias,
            learning_rate,
        }
    }

    /// The activation function that decides the perceptron's output based on a given sum.
    ///
    /// This is a step function that returns 1.0 if `sum > 0.0` and 0.0 otherwise.
    ///
    /// # Arguments
    ///
    /// * `sum` - The weighted sum of inputs plus bias.
    ///
    /// # Returns
    ///
    /// * `1.0` if `sum > 0.0`, else `0.0`.
    fn activation(&self, sum: f64) -> f64 {
        // If the sum is positive, return 1.0, otherwise return 0.0.
        if sum > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    /// Predicts the output of the perceptron for a given set of inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of input values for which we want a prediction.
    ///
    /// # Returns
    ///
    /// The perceptron's prediction, either 0.0 or 1.0.
    fn predict(&self, inputs: &[f64]) -> f64 {
        // Calculate the weighted sum of inputs:
        // (w1 * x1) + (w2 * x2) + ... + (wn * xn)
        // Then add the bias.
        let sum: f64 = self.weights.iter()
            .zip(inputs.iter())    // Pair each weight w with its corresponding input i
            .map(|(w, &i)| w * i)  // Multiply them together
            .sum::<f64>() + self.bias;

        // Apply the activation function on the sum to get the perceptron's output.
        self.activation(sum)
    }

    /// Trains the perceptron using a simple learning rule over a given number of epochs.
    ///
    /// For each training example:
    /// 1. Make a prediction.
    /// 2. Compute the error (target - prediction).
    /// 3. Update each weight and the bias based on the error and learning rate.
    ///
    /// # Arguments
    ///
    /// * `training_data` - A slice of (input vector, target) pairs.
    /// * `epochs` - Number of times to iterate over the entire training set.
    fn train(&mut self, training_data: &[(Vec<f64>, f64)], epochs: usize) {
        // Repeat the training process for the specified number of epochs.
        for _ in 0..epochs {
            // For each training example (input vector, target output)
            for (inputs, target) in training_data {
                // Predict the current output based on the inputs.
                let prediction = self.predict(inputs);
                // Compute the error as target - prediction.
                let error = target - prediction;

                // Update each weight: w_i = w_i + (learning_rate * error * input_i)
                for i in 0..self.weights.len() {
                    self.weights[i] += self.learning_rate * error * inputs[i];
                }

                // Update the bias: bias = bias + (learning_rate * error)
                self.bias += self.learning_rate * error;
            }
        }
    }
}

fn main() {
    // Our training data: Each item is (inputs, target_output).
    // Here we are training a perceptron to behave like the AND function.
    let training_data = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 0.0),
        (vec![1.0, 0.0], 0.0),
        (vec![1.0, 1.0], 1.0),
    ];

    // Create a Perceptron with 2 inputs and a learning rate of 0.1.
    let mut perceptron = Perceptron::new(2, 0.1);

    // Train the perceptron for 100 epochs on our training data.
    perceptron.train(&training_data, 100);

    // Evaluate the trained perceptron on the training data.
    // We expect it to have learned the AND logic: only (1,1) should produce 1.0.
    for (inputs, target) in &training_data {
        let prediction = perceptron.predict(inputs);
        println!("Prediction: {}, Target: {}", prediction, target);
    }

    // Test the perceptron with new/unseen data to see how it behaves.
    let new_data = vec![
        vec![10.0, -50.0],
        vec![10.0, 15.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0],
        vec![-50.00151, -50.00151],
        vec![50.00151, 50.00151],
    ];

    for inputs in &new_data {
        let prediction = perceptron.predict(inputs);
        println!("For input {:?}, the perceptron predicts: {}", inputs, prediction);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test the creation of a new Perceptron instance.
    #[test]
    fn test_perceptron_new() {
        let p = Perceptron::new(3, 0.1);
        assert_eq!(p.weights, vec![0.0, 0.0, 0.0]);
        assert_eq!(p.bias, 0.0);
        assert_eq!(p.learning_rate, 0.1);
    }

    // Test the activation function with a positive sum.
    #[test]
    fn test_perceptron_activation_1() {
        let p = Perceptron::new(3, 0.1);
        let sum: f64 = 2.0;
        let output = p.activation(sum);
        assert_eq!(output, 1.0);
    }

    // Test the activation function with a negative sum.
    #[test]
    fn test_perceptron_activation_0() {
        let p = Perceptron::new(3, 0.1);
        let sum: f64 = -1.0;
        let output = p.activation(sum);
        assert_eq!(output, 0.0);
    }

    // Test the predict function for a scenario where the output should be 1.0.
    #[test]
    fn test_perceptron_predict_1() {
        let mut p = Perceptron::new(3, 0.1);
        p.weights = vec![0.5, 0.5, 0.5];
        p.bias = -0.5;
        let input = vec![1.0, 1.0, 1.0];
        let output = p.predict(&input);
        assert_eq!(output, 1.0);
    }

    // Test the predict function for a scenario where the output should be 0.0.
    #[test]
    fn test_perceptron_predict_0() {
        let mut p = Perceptron::new(3, 0.1);
        p.weights = vec![0.5, 0.5, 0.5];
        p.bias = -0.5;
        let input = vec![0.0, 0.0, 0.0];
        let output = p.predict(&input);
        assert_eq!(output, 0.0);
    }

    // Test the train function with a simple dataset.
    #[test]
    fn test_perceptron_train() {
        let mut p = Perceptron::new(3, 0.1);
        let training_data = vec![(vec![1.0, 1.0, 1.0], 1.0), (vec![0.0, 0.0, 0.0], 0.0)];
        let target = 1; // 1 epoch
        p.train(&training_data, target);
        // After one epoch of training on these points, the weights and bias should have changed.
        // This test checks if the final state matches expected values.
        // (Note: These tests may fail if your logic differs from expectations, 
        // but they illustrate the testing process.)
        assert_eq!(p.weights, vec![0.1, 0.1, 0.1]);
        assert_eq!(p.bias, 0.0);
    }

    #[test]
    fn test_perceptron_train_2() {
        let mut p = Perceptron::new(3, 0.1);
        let training_data = vec![
            (vec![1.0, 1.0, 1.0], 1.0),
            (vec![1.0, 1.0, 0.0], 0.0)
        ];
        let target = 1; // 1 epoch of training
        p.train(&training_data, target);
        // Check final weights and bias after training.
        assert_eq!(p.weights, vec![0.0, 0.0, 0.1]);
        assert_eq!(p.bias, 0.0);
    }
}
