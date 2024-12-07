# Rust Perceptron

## Description
This project implements a simple perceptron in Rust, serving as a foundational introduction to neural networks and machine learning concepts. The perceptron takes multiple inputs, each associated with a weight, sums them along with a bias term, and then applies a step activation function to produce a binary output (0 or 1). Through repeated training cycles (epochs) and simple weight-adjustment rules, this perceptron learns to distinguish between two classes of input patterns.

By developing this perceptron from scratch, the project showcases an understanding of:
- Basic neural network principles
- Weight initialization and parameter tuning
- Activation functions and decision boundaries
- Supervised learning with simple datasets
- Writing clean, documented, and testable Rust code

#

## Table of Contents
* [Description](#description)
* [Libraries](#libraries)
* [Classes & Methods](#classes--methods)
* [Functions](#functions)
* [Test Cases](#test-cases)
* [How to use](#how-to-use)
* [When to use](#when-to-use)
* [Skills utilized in this project](#skills-utilized-in-this-project)

#

## Libraries
* **Standard Rust Library**  
  Uses standard collections and I/O functionality. No external crates are required for the perceptron's core logic.

#

## Classes & Methods
* **Perceptron**
    * **new(num_inputs: usize, learning_rate: f64) -> Perceptron**  
      Creates and returns a new Perceptron instance with zero-initialized weights and bias.  
      **Parameters:**  
      - `num_inputs`: The number of input features.  
      - `learning_rate`: The pace at which the perceptron updates weights.
      
    * **activation(&self, sum: f64) -> f64**  
      Applies a step activation function. Returns 1.0 if `sum > 0.0`, else 0.0.
      
    * **predict(&self, inputs: &[f64]) -> f64**  
      Computes the perceptron's output for given inputs by calculating the weighted sum plus bias, then applying the activation function.
      
    * **train(&mut self, training_data: &[(Vec<f64>, f64)], epochs: usize)**  
      Trains the perceptron on the provided dataset. For each input-target pair and for a given number of epochs, it updates the weights and bias based on the prediction error.

#

## Functions
* **main()**  
  Demonstrates the usage of the Perceptron by:
  - Initializing a perceptron for a binary classification task.
  - Training the perceptron to learn an AND logic function using a small dataset.
  - Evaluating the trained perceptron on both the training and some new/unseen data.

#

## Test Cases
* **test_perceptron_new()**  
  Ensures that a new Perceptron is created with the correct initial weights, bias, and learning rate.

* **test_perceptron_activation_1()**  
  Checks that the activation function returns 1.0 for positive input sums.

* **test_perceptron_activation_0()**  
  Checks that the activation function returns 0.0 for non-positive input sums.

* **test_perceptron_predict_1()**  
  Tests that the `predict` method can produce a 1.0 output given certain weights, bias, and inputs.

* **test_perceptron_predict_0()**  
  Tests that the `predict` method can produce a 0.0 output given certain weights, bias, and inputs.

* **test_perceptron_train()**  
  Verifies that after one epoch of training on a given dataset, the perceptron updates its weights (and possibly bias) correctly.

* **test_perceptron_train_2()**  
  Checks the perceptron’s learning behavior on a slightly different dataset to ensure weight and bias updates are consistent.

#

## How to use
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/ZichKoding/RustPerceptron.git
   cd RustPerceptron
   ```

2. **Build the Project:**  
   ```bash
   cargo build
   ```
   
3. **Run the Application:**  
   ```bash
   cargo run
   ```
   This will train the perceptron and display the predictions for the training data as well as for some test inputs.

4. **Run the Tests:**  
   ```bash
   cargo test
   ```
   This will execute the suite of unit tests to ensure the Perceptron logic works as expected.

#

## When to use
Use this perceptron implementation when:
- Learning the fundamentals of neural networks and machine learning.
- Demonstrating binary classification tasks (e.g., logical AND, OR).
- Experimenting with basic learning rules and parameter tuning.
- Gaining familiarity with Rust’s ownership, memory safety, and type system in a simple AI context.
- Creating a baseline model before experimenting with more complex architectures or libraries.

#

## Skills utilized in this project
- **Rust Programming:** Safe, systems-level development with attention to memory and type safety.
- **Machine Learning Fundamentals:** Understanding model initialization, forward pass, backpropagation-like weight updates, and convergence behavior.
- **Algorithmic Thinking:** Implementing the perceptron learning rule and applying a binary decision boundary function.
- **Software Testing:** Writing comprehensive unit tests to verify correctness, stability, and reliability.
- **Code Documentation:** Using doc comments and README formatting to create clear, maintainable, and user-friendly documentation.
- **Version Control & Project Structure:** Adhering to Cargo standards, organizing code in modules, and ensuring reproducible builds and tests.
