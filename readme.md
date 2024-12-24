
# Linear Regression with One Variable

This project implements a basic linear regression algorithm using gradient descent to find the best-fit line for a given dataset. The project uses Python with `NumPy` and `matplotlib` for calculations and visualization.

## Features
- Implements linear regression with one variable.
- Uses gradient descent to optimize parameters.
- Visualizes the dataset and the best-fit line using `matplotlib`.
- Measures the training time.

## Dataset
The dataset is randomly generated with:
- `x`: integers from 1 to 100.
- `y`: random values between 50 and 150.

## Installation
1. Clone this repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Install required packages:
   ```bash
   pip install numpy matplotlib
   ```

## Usage
1. Run the script to train the model and visualize the results:
   ```bash
   python main.py
   ```

2. The script will:
   - Train the linear regression model using gradient descent.
   - Display the time spent training.
   - Plot the dataset (scatter plot) and the best-fit line.

## Code Overview
### Function: `find_best_cost_function_params`
This function calculates the best-fit parameters `w` and `b` for the equation:

\[ f(x) = wx + b \]

- **Input:**
  - `x_series`: List of x-values.
  - `y_series`: List of y-values.
  - Optional parameters for learning rate, tolerance, and maximum iterations.

- **Output:**
  - Best-fit values of `w` and `b`.

### Visualization
- The scatter plot shows the actual data points.
- The best-fit line (calculated using `w` and `b`) overlays the data.

## Example Output
The console will display updates during the training process and the final values of `w` and `b`. A plot is generated showing:
- Red dots representing the dataset.
- A blue line representing the best-fit line.

## Customization
- Adjust the learning rate (`learning_rate`) in the `find_best_cost_function_params` function to control the gradient descent step size.
- Modify the dataset generation logic to use custom data instead of random values.

## Dependencies
- Python 3.7+
- NumPy
- Matplotlib

## Future Improvements
Here are some ideas for improving the project in the future:
- **Dynamic Learning Rate:** Implement a mechanism to adjust the learning rate dynamically during training for better convergence.
- **Error Plot:** Add a plot to visualize the error reduction over iterations.
- **Batch Gradient Descent:** Modify the algorithm to use batch gradient descent for better performance on larger datasets.
- **Data Normalization:** Normalize the dataset to improve the stability of gradient descent.
- **Model Evaluation:** Add evaluation metrics like Mean Squared Error (MSE) to assess the model's performance.
- **Save Model Parameters:** Allow saving and loading the trained model parameters (w, b) to avoid retraining every time.
- **User Input:** Enable the script to accept user-provided datasets via file upload.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as needed.
