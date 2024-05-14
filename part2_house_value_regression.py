# # ** ML Algo Regression :** Artificial Neural Networks Part 2 : Create and train a neural network for regression


# ## ***Table of Contents***
# - [1. Import dependancies]
# - [2. Regressor class]
# - [3. Other required functions]
# - [4. Cross-Validation]
# - [5. Display functions]
# - [6. Main]


# ## **Step 1 - import dependencies**
# Import all libraries required


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import pickle

import numpy as np
import pandas as pd
import scipy
from scipy.stats import sem, t
import random


import sklearn as sk
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#matplotlib imports and display functions will be placed in comments so as not to slow down the code unnecessarily
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from matplotlib.lines import Line2D


# ## **Step 2 - Regressor Class**


class Regressor(nn.Module):
    def __init__(self, x, nb_epoch=None, nb_batches=None, lr=None, neurons=None, activations=None):
        """
        Initialize the Regressor.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
            - nb_epoch {int} -- Number of training epochs.
            - nb_batches {int} -- Number of mini-batches for training.
            - neurons {List[int]} -- List specifying the number of neurons in each layer.
            - activations {List[str]} -- List specifying the activation function for each layer.
        """

        super(Regressor, self).__init__()

        # Initialize parameter_preprocess
        self.parameter_preprocess = []

        # Processes the data
        X, Y = self._preprocessor(x, training=True)

        # Set the parameters
        self.input_size = X.shape[1]
        self.output_size = 1  # it's a regression problem
        self.nb_epoch = nb_epoch if nb_epoch is not None else 50
        self.nb_batches = nb_batches if nb_batches is not None else 128
        self.lr = lr if lr is not None else 0.006420183639503851
        self.criterion = torch.nn.MSELoss()

        if neurons is None: # Set default values if neurons is not provided
            neurons = [self.input_size, 256, 512, 1024, 512, 256, 128, self.output_size]
        if activations is None: # Set default values if activations is not provided
            activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']
        self.neurons = neurons # Use the provided or default values for neurons
        self.activations = activations # Use the provided or default values for activations


        # Set the NN structure
        self._layers = nn.ModuleList()

        # Fill the NN structure
        for i in range(len(neurons) - 1):
            layer = nn.Linear(neurons[i], neurons[i + 1])
            init.eye_(layer.weight)
            self._layers.append(layer)

        # Initialize optimizer after layers are registered
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        
        
        
        

    def forward(self, x):
        """
        Forward pass of the neural network.

        Arguments:
            - x {torch.tensor} -- Input tensor of shape (batch_size, input_size).

        Returns:
            {torch.tensor} -- Output tensor of the neural network.
        """

        for layer in self._layers:
            x = nn.functional.relu(layer(x))

        return x
        
        
        
        
        

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
            size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
            size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        if training:
            # Data extraction
            df_to_process = x.copy()  # Make a copy to avoid modifying the original DataFrame

            # One-hot encoding with LabelBinarizer
            if 'ocean_proximity' in df_to_process.columns:
                lb = LabelBinarizer()
                encode_text = lb.fit_transform(
                    df_to_process['ocean_proximity'])
                encode_text = pd.DataFrame(
                    encode_text, columns=lb.classes_.tolist())
                df_to_process = df_to_process.join(encode_text)
                df_to_process = df_to_process.drop(
                    columns=['ocean_proximity'])  # Drop the textual column
                self.parameter_preprocess.append(lb)  # Store One-hot encoding

            # Fill Na value in each coumn with their median value.
            values_median = df_to_process.median()  # keep median values of each column
            df_to_process = df_to_process.fillna(df_to_process.median())
            self.parameter_preprocess.append(
                values_median)  # Store fillna handling

            # Normalization
            scaler = MinMaxScaler()
            df_to_process[df_to_process.columns] = scaler.fit_transform(
                df_to_process[df_to_process.columns])
            self.parameter_preprocess.append(scaler)  # Store normalization

        else:
            # Preprocessing of test dataset (or validation)
            # Data extraction
            df_to_process = x.copy()  # Make a copy to avoid modifying the original DataFrame

            if 'ocean_proximity' in df_to_process.columns:
                # Check if one-hot encoding was applied during training
                if len(self.parameter_preprocess) > 1:
                    lb = self.parameter_preprocess[0]
                    encode_text = lb.transform(
                        df_to_process['ocean_proximity'])
                    encode_text = pd.DataFrame(
                        encode_text, columns=lb.classes_.tolist())
                    df_to_process = df_to_process.join(encode_text)
                    df_to_process = df_to_process.drop(
                        columns=['ocean_proximity'])  # Drop the textual column

            df_to_process = df_to_process.fillna(self.parameter_preprocess[1])

            if len(self.parameter_preprocess) > 2 and isinstance(self.parameter_preprocess[2], MinMaxScaler):
                scaler = self.parameter_preprocess[2]
                df_to_process[df_to_process.columns] = scaler.transform(
                    df_to_process[df_to_process.columns])

        if y is not None:
            y = torch.tensor(y.astype(np.float32).values)

        X_tensor = torch.tensor(df_to_process.astype(np.float32).values)

        # Return preprocessed x and y, return None for y if it was None
        return X_tensor, (y if training else None)
    
    
    
    
    
    
    
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        X, Y = self._preprocessor(x, y, training=True)


        # Calculate the total number of batches
        total_batches = len(X) // self.nb_batches

        for epoch in range(self.nb_epoch):
            for i in range(total_batches):
                input_batched = X[i * self.nb_batches:(i + 1) * self.nb_batches]
                y_gold_batched = Y[i * self.nb_batches:(i + 1) * self.nb_batches]

                # Reset the gradient
                self.optimizer.zero_grad()
                
                # Predict the labels
                output_batched = self.forward(input_batched)

                assert(y_gold_batched.shape == output_batched.shape), "y_gold_batched and output_batched are not the same size"

                # Compute the loss
                loss = self.criterion(output_batched, y_gold_batched)

                # This limits the norm of the gradients to prevent them from becoming too large
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # Backward pass and update parameters
                loss.backward()
                self.optimizer.step()
            

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        
        
        
        
                
   
        
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        x, _ = self._preprocessor(x, training=False)
        predictions = self.forward(x)

        return predictions.detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        
        
        
        

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        prediction = self.predict(x)

        mse = mean_squared_error(y, prediction, squared=False)

        return mse

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        
        
        
        
        
# ## **Step 3 - Other required functions**        
        

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")



def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model




def regressor_hyperparameter_search_random(x_train, y_train, num_trials=10):
    """
    Performs a hyper-parameter search for fine-tuning the regressor implemented 
    in the Regressor class using a random search layout.

    Arguments:
        x_train {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
        y_train {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        num_trials {int} -- Number of random trials.

    Returns:
        dict -- Dictionary containing the best hyperparameters.
    """

    best_params = {}
    best_score = float('inf')  # Initialize with a high value
    score_list = []
    all_trials = []  # List to store all trial parameters and scores

    learning_rate_range = [0.0001, 0.01]
    batch_size_range = [64, 128, 256, 512]
    num_epochs_range = 75  # Fix at 75
    neurons_list = [[13, 512, 512, 1], [13, 256, 512, 256, 1], [13, 128, 256, 256, 1], [13, 256, 512, 256, 1], [13, 256, 512, 256, 128, 1], [13, 128, 256, 512, 256, 128, 1],
                    [13, 256, 512, 1024, 512, 256, 128, 1], [13, 128, 256, 512, 1024, 512, 256, 128, 1]]
    activations_list = [['relu', 'relu', 'relu', 'relu', 'relu'], ['relu', 'relu', 'relu', 'relu', 'relu'], ['relu', 'relu', 'relu', 'relu', 'relu'], 
                        ['relu', 'relu', 'relu', 'relu', 'relu'],
                        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
                        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
                        ['relu', 'relu', 'relu', 'relu',
                            'relu', 'relu', 'relu', 'relu'],
                        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']]
    # mettre aussi les params avec sigmoid à la fin

    for i in range(num_trials):
        print(f'Trial number {i + 1} / {num_trials}')
        learning_r = random.uniform(
            learning_rate_range[0], learning_rate_range[1])
        batch_size = random.choice(batch_size_range)
        num_epochs = num_epochs_range
        k = random.choice([i for i in range(len(neurons_list))])
        neurons = neurons_list[k]
        activations = activations_list[k]

        regressor = Regressor(x_train, nb_epoch=num_epochs, nb_batches=batch_size, lr=learning_r, neurons=neurons,
                              activations=activations)
        score = cross_validation(regressor, x_train, y_train, 5)

        # Save parameters and score for each trial
        trial_params = {
            'lr': learning_r,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'neurons': neurons,
            'activations': activations
        }
        all_trials.append({'params': trial_params, 'score': score})

        score_list.append(score)

        if score < best_score:
            best_score = score
            best_params = trial_params

    print("Best Hyperparameters:")
    print(f"Learning Rate: {best_params['lr']}")
    print(f"Batch Size: {best_params['batch_size']}")
    print(f"Num Epochs: {best_params['num_epochs']}")
    print(f"Neurons: {best_params['neurons']}")
    print(f"Activations: {best_params['activations']}")
    print(f"Best Score: {best_score}")

    # Return the best parameters and the list of all trials
    return best_params, score_list, all_trials





 ## **Step 4 - Cross-Validation**

def cross_validation(regressor, x, y, n_split=5):
    """
    Perform k-fold cross-validation on the provided Regressor.

    Arguments:
        - regressor {Regressor} -- The regression model to be evaluated.
        - X {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
        - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        - n_splits {int} -- Number of folds for cross-validation.

    Returns:
        {float} -- Mean of the evaluation metric (e.g., RMSE) across all folds.
    """

    list_of_errors = []

    # Create a KFold object with n_split folds
    kf = KFold(n_splits=n_split, shuffle=True, random_state=42)

    # Iterate over the folds
    for fold_idx, (train_index, val_index) in enumerate(kf.split(x)):
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = Regressor(x, nb_epoch=regressor.nb_epoch, nb_batches=regressor.nb_batches,
                          lr=regressor.lr, neurons=regressor.neurons, activations=regressor.activations)
        model.fit(x_train,y_train)
        error = model.score(x_val, y_val)
        list_of_errors.append(error)

    avg = np.mean(list_of_errors)

    return avg




## **Step 5 - Display functions**

"""
def plot_hyperparameter_search_2d(all_trials):

    #Plots the results of a hyperparameter search in a 2D space.
    #Arguments:
    #- all_trials: List of dictionaries, each containing hyperparameters and corresponding scores.

    #Returns:
    #None

    fig, ax = plt.subplots(figsize=(10, 8))

 

    # Extract parameters and scores from all trials

    lr_values = [trial['params']['lr'] for trial in all_trials]

    batch_size_values = [trial['params']['batch_size'] for trial in all_trials]

    scores = [trial['score'] for trial in all_trials]

    neurons_values = [len(trial['params']['neurons']) - 2 for trial in all_trials]

 

    # Scatter plot with switched color and y-axis

    sc = ax.scatter(lr_values, scores, c=batch_size_values, cmap='viridis', s=50)

 

    # Annotate points with the number of neurons

    for i, txt in enumerate(neurons_values):

        ax.annotate(txt, (lr_values[i], scores[i]+500), textcoords="offset points", xytext=(0, 5), ha='center')

 

    # Find the index of the trial with the best score

    best_index = scores.index(min(scores))

 

    # Highlight the point with the best score using a red square

    rect = patches.Rectangle((lr_values[best_index]-0.00015, scores[best_index]-1000), 0.0003, 2000, linewidth=2, edgecolor='red', facecolor='none', label='Best Score')

    ax.add_patch(rect)

   

    # Set labels and title

    ax.set_xlabel('Learning Rate')

    ax.set_ylabel('Score')

    ax.set_title('Hyperparameter Search Results')

 

    # Add colorbar

    cbar = plt.colorbar(sc)

    cbar.set_label('Batch Size')

 

    # Show the plot

    # Add legend for the number of hidden layers

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black', markersize=10), patches.Rectangle((0, 0), 1, 1, edgecolor='red', facecolor='none', label='Best Score')]

    labels = ['Number of Hidden Layers: $n$', 'Best Score']

    ax.legend(handles, labels, loc='upper right')

    plt.show()
    


def plot_prediction_errors(regressor, x_data, y_true):
    
    #Make predictions using a regressor and plot the differences between predicted and true values.

    #Arguments:
        #- regressor: The trained regressor model.
        #- x_data: Input data for prediction.
        #- y_true: True labels for the input data.

    #Returns:
        #None
    
    # Make predictions using the provided regressor
    y_pred = regressor.predict(x_data)

    # Calculate the prediction errors
    errors = y_true.values.flatten() - y_pred.flatten()

    # Plotting histogram of prediction errors
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black')
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Calculate and print additional metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R2): {r2:.2f}')


def fit(self, x, y, x_test=None, y_test=None):
    
    #Regressor training function

    #Arguments:
        #- x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
        #- y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

    #Returns:
        #self {Regressor} -- Trained model.
    

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
        
    X, Y = self._preprocessor(x, y, training=True)

    X_test, _ = self._preprocessor(x_test, y_test, training=False)

    print()

    train_losses = []
    val_losses = []

    total_batches = len(X) // self.nb_batches

    for epoch in range(self.nb_epoch):
        avg_loss = 0
        for i in range(total_batches):
            input_batched = X[i * self.nb_batches:(i + 1) * self.nb_batches]
            y_gold_batched = Y[i * self.nb_batches:(i + 1) * self.nb_batches]

            # Reset the gradient
            self.optimizer.zero_grad()
                
            # Predict the labels
            output_batched = self.forward(input_batched)

            assert(y_gold_batched.shape == output_batched.shape), "y_gold_batched and output_batched are not the same size"

            # Compute the loss
            loss = self.criterion(output_batched, y_gold_batched)

            # This limits the norm of the gradients to prevent them from becoming too large
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # Backward pass and update parameters
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()
            
        avg_loss /= total_batches
        train_losses.append(avg_loss)

        if X_test is not None:
            X_prediction = self.forward(X_test)
            Y_test = torch.tensor(y_test.astype(np.float32).values)
            val_loss = self.criterion(X_prediction, Y_test).item()
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{self.nb_epoch}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Plot the training and validation loss
    plt.plot(range(1, self.nb_epoch + 1), train_losses, label='Training Loss')
    plt.plot(range(1, self.nb_epoch + 1), val_losses, label='Validation Loss')
    plt.axvline(x=100, color='red', linestyle='--', label='Optimal Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve of the Neural Network')
    plt.legend()
    plt.show()

    return train_losses, val_losses

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
"""

# ## **Step 6 - Main**

def example_main():

    data = pd.read_csv("housing.csv")
    output_label = "median_house_value"
    x_train, x_test, y_train, y_test = train_test_split(data.loc[:, data.columns != output_label],data.loc[:, [output_label]],test_size=0.2,random_state=42)

    #best_params, score_list, all_trials=regressor_hyperparameter_search_random(x_train, y_train, num_trials=50)
    #plot_hyperparameter_search_2d(all_trials)

    best_regressor = Regressor(x_train, nb_epoch=80, nb_batches=128, lr=0.00642, neurons=[13, 256, 512, 1024, 512, 256, 128, 1],
                              activations=['relu','relu','relu','relu','relu','relu','relu','relu'])

    best_regressor.fit(x_train, y_train)
    
    save_regressor(best_regressor)   
    
    score = best_regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(score))
    
    #Additonal metrics to assess the performance of our model
    """
 
    predictions = best_regressor.predict(x_test)
    
    # Calculate absolute differences
    differences = (predictions - y_test)
    
    # Calculate mean
    mean_difference = np.mean(differences)
    
    # Calculate standard deviation
    std_difference = np.std(differences)
    
    # Calculate the standard error of the mean
    sem_difference = sem(differences)
    
    # Calculate the margin of error for a 95% confidence interval
    confidence_level = 0.95
    confidence_interval = sem_difference * t.ppf((1 + confidence_level) / 2, len(differences) - 1)
    
    # Calculate the confidence interval
    lower_bound = mean_difference - confidence_interval
    upper_bound = mean_difference + confidence_interval
    
    
    # Calculate minimum and maximum
    min_difference = np.min(differences)
    max_difference = np.max(differences)
    
    # Display the results
    print(f"Mean Absolute Difference: {mean_difference}")
    print(f"Standard Deviation of Absolute Difference: {std_difference}")
    print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")
    print(f"Confidence interval:{confidence_interval}")
    print(f"Minimum Absolute Difference: {min_difference}")
    print(f"Maximum Absolute Difference: {max_difference}")
    
    """
    
    #plot_prediction_errors(best_regressor, x_test, y_test)
    #train_losses, val_losses = best_regressor.fit(x_train, y_train, x_test, y_test) #If we use the fit function to display the loss curve (our test dataset then becomes a validation dataset)
    



if __name__ == "__main__":
    example_main()
