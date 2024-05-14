# ** ML Algo Regression :** Neural Network

**Subject:** Implementing a Neural Network library from scratch to build and train models (part 1),  create and fine tune a Neural Network using Torch module, in order to predict California House pricing (part 2).

# Part 1: Neural Network Library Implementation for Regression

## A. Overview:
The **part1_nn_library.py** Python script provides a basic implementation of a neural network library for regression tasks. The neural network is implemented from scratch, covering essential components such as layers, loss functions, and the overall network architecture. This library also include some metrics methods, in order to both predict and evaluate the accuracy and performance of a model.

## B. Steps of usage

1. **Prerequisites**:
   - Python 3.0
   - Required Python libraries (numpy, matplotlib)

  ```bash
  pip install -r requirements.txt
  ```

2. **Directory Structure**:

Ensure your directory structure is organized like this:

Path-to-your-folder
├── part1_nn_library.py
├── README.md
├── iris.dat

You can replace `iris.dat` with your dataset files.

**NB: In this case, please ensure to keep the same name for the files you are uploading**

3. **Data Preparation**:

- Make sure your dataset files are in adapted format with the same structure (file name, attributes and labels).

4. **Running the Script**:
- Open a terminal and navigate to the script's directory.
- Execute the script using the following command:

   ```bash
   python part1_nn_library.py
   ```

5. **Results**:

The main of the Python script will be executed and results will appear in the user's console.

## C. Understanding the Results

- By running the script, the user should expect to see output regarding the training and validation loss during the training process.
- After training, the script prints the training loss and validation loss.
- Finally, it displays the validation accuracy of the trained model.

## D. Modifying for Your Dataset

To use this script with your own dataset, follow these steps:

1. Replace `iris.dat` in the script's directory with your dataset files.

2. You may need to modify other script parameters (e.g., random seed, Network's structure, hyperparameters, ...) to suit your specific analysis.

3. Run the script as described in the "Usage" section to analyze your dataset with the decision tree classifier.


# Part 2: Neural Network fine tuning for housing prediction

## A. Overview:
The **part2_house_value_regression.py** Python script provides a neural network implementation on torch. The neural network is implemented using the Torch module, by creating a Regressor class. additionnaly methods were set up in order to make the fine tuning easier.

## B. Steps of usage

1. **Prerequisites**:

- torch
- pandas
- numpy
- scipy
- scikit-learn

These prerequistes can be easily implemented by running the following command:

  ```bash
  pip install -r requirements.txt
  ```

**Note:**
- Ensure that you have the required dependencies installed before running the code.
- This code assumes a specific structure of the input data (California housing dataset). Adjust preprocessing steps accordingly for different datasets.

2. **Directory Structure**:

Ensure your directory structure is organized like this:

Path-to-your-folder
├── part2_house_value_regression.py
├── README.md
├── housing.csv

You can replace `housing.csv` with your dataset files.

**NB: In this case, please ensure to keep the same name for the files you are uploading**

3. **Data Preparation**:

- Make sure your dataset files are in adapted format with the same structure (file name, attributes and labels).

4. **Running the Script**:
- Open a terminal and navigate to the script's directory.
- Execute the script using the following command:

   ```bash
   python part2_house_value_regression.py
   ```

5. **Results**:

The main of the Python script will be executed and results will appear in the user's console.

## C. Understanding the Results

- By running the script, the user should expect to see output of the score of an optimised model on the test dataset (20% of the dataset).
- The hyperparameters of the model has been fine tuned using a Random Layout Search algorithm, based on both training and validation dataset (80% of the dataset) using cross validation over 5 Fold. Please note that the test data has been removed from the dataset before starting the optimization of the hyperparameters.
- Finally, it displays the score of the model.

## D. Modifying for Your Dataset

To use this script with your own dataset, follow these steps:

1. Replace `housing.csv` in the script's directory with your dataset files.

2. You may need to modify other script parameters (e.g., random seed, Network's structure, hyperparameters, ...) to suit your specific analysis.

3. Run the script as described in the "Usage" section to analyze your dataset with the decision tree classifier.
