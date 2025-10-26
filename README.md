
# MNIST-Neural-Network-from-scratch
Implementing a Neural Network from scratch (NumPy) to demonstrate a core understanding of backpropagation.
=======


# MNIST Digit Recognizer: Neural Network from Scratch (NumPy)

## 1\. Project Overview`

This project is a complete implementation of a simple 2-layer Neural Network to recognize handwritten digits from the famous MNIST dataset.

The key feature of this project is that it is built **entirely from scratch using only NumPy**. No deep learning frameworks (like TensorFlow or Keras) were used. This approach was taken to demonstrate a fundamental, first-principles understanding of the core mathematical concepts behind deep learning.

## 2\. Core Concepts & Mathematical Knowledge

This project is a practical application of the core mathematics that power all deep learning models:

* **Forward Propagation:** Implementing the flow of data through the network:
    $$Z^{[1]} = W^{[1]}X + b^{[1]}$$
    $$A^{[1]} = g_{\text{ReLU}}(Z^{[1]})$$

* **Backward Propagation (Backpropagation):** Calculating the gradients (derivatives) of the loss function with respect to every weight and bias in the network:
    $$dZ^{[2]} = A^{[2]} - Y$$
    $$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
    $$dB^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$
    $$dZ^{[1]} = W^{[2]T} dZ^{[2]} \cdot g^{[1]\prime}(Z^{[1]})$$
    $$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^{T}$$
    $$dB^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$
  * **Gradient Descent:** Using the calculated gradients to iteratively update the model's parameters and "learn" from the data ($\alpha$ = learning rate).
  * **Activation Functions:** Implementing and understanding the role of `ReLU` (for hidden layers) and `Softmax` (for multi-class classification output).
  * **Linear Algebra:** Heavy use of matrix and vector operations (dot products, transposition, summation) via NumPy, which is the foundation of efficient neural network computation.
  * **Data Preprocessing:** Understanding the importance of data normalization (scaling pixel values from 0-255 to 0-1) and one-hot encoding for categorical labels.

## 3\. Technology Stack

  * **Python:** The core programming language.
  * **NumPy:** Used for all numerical computations, matrix operations, and implementing the network logic.
  * **Pandas:** Used for loading and initially handling the `.csv` data.
  * **Matplotlib:** Used for all data visualization, including plotting the training accuracy curve, visualizing individual predictions, and generating the final confusion matrix.
  * **Kaggle API:** Used to programmatically download and manage the dataset directly from the competition.
  * **`python-dotenv`:** Used for managing environment variables (like file paths) cleanly.
  * **Jupyter (in PyCharm):** Used as the interactive environment for developing, testing, and documenting the code in cells.

## 4\. How to Execute

Follow these steps to set up and run the project locally.

### Step 1: Clone & Set Up Environment

1.  Clone this repository to your local machine.
2.  Create a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install all required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Kaggle API Setup

This project requires the Kaggle API to download the data.

1.  Log in to your Kaggle account.
2.  Go to `Account` \> `API` \> `Create New API Token`. This will download a `kaggle.json` file.
3.  Place this file in the correct location on your system:
      * **Windows:** `C:\Users\<Your-Username>\.kaggle\kaggle.json`
      * **macOS/Linux:** `~/.kaggle/kaggle.json`

### Step 3: Create `.env` File

Create a file named `.env` in the root of the project directory. This file will tell the notebook where to find the data. Add the following line to it:

```
LOCAL_TRAIN_FILE_PATH=train.csv
```

### Step 4: Download the Data

In your terminal, run the following command to download the complete competition dataset:

```bash
kaggle competitions download -c digit-recognizer
```

This will download `digit-recognizer.zip`. Unzip this file (e.g., "Extract Here") to get `train.csv` and `test.csv` in your project folder.

### Step 5: Run the Notebook

Open the `notebook.py` file in PyCharm (or your preferred IDE with Jupyter support) and run the cells sequentially from top to bottom.

## 5\. Results & Model Analysis

The model was trained on 41,000 images and tested on a 1,000-image validation set it had never seen before.

  * **Final Validation Accuracy:** \~84-85%

### Training Accuracy Plot

The plot below shows how the model's accuracy on the training data improved over 500 iterations. The smooth upward curve is a clear indicator that the **gradient descent optimization was successful.**

*(Note: The `notebook.py` file automatically saves this plot as `training_accuracy_plot.png` when you run Cell 6).*

### Validation Confusion Matrix

The confusion matrix below provides a deep analysis of the model's performance on the unseen validation data.

#### Key Insights from the Analysis:

1.  **Strong Performance on '0' and '1':** The model identifies digits '0' and '1' with extremely high accuracy.
2.  **Primary Weakness: Confusing '9' and '4':** The model's single biggest flaw is its confusion between '9's and '4's (and to a lesser extent, '9's and '7's). This is evident from the large off-diagonal numbers at:
      * **True '9', Predicted '4':** 16 errors
      * **True '4', Predicted '9':** 10 errors
3.  **Other Problem Areas:** The digit '2' also shows some confusion, being mistaken for '3', '8', and other digits.

These insights are crucial for future improvement, suggesting that the model would benefit from more data or a more complex architecture (e.g., more layers/neurons or Convolutional layers) to learn the finer features that distinguish these similar-looking digits.

