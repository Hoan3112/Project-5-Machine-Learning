from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax, tanh
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()

        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        return matmul(x, self.w.T).squeeze()
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self.run(x)
        if score.item() >= 0:
            return 1
        else:
            return -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            while True:
                all_correct = True
                for batch in dataloader:
                    x = batch['x']
                    y = batch['label']
                    
                    prediction = self.get_prediction(x)
                    actual = y.item()
                    
                    if prediction != actual:
                        # Update weights: w = w + direction * magnitude
                        # For perceptron, we update by: w = w + y * x
                        self.w.data += x * actual
                        all_correct = False
                
                # If we made a complete pass without any mistakes, we're done
                if all_correct:
                    break



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        
        # Define the architecture
        # Input size: 1, Hidden sizes: 200, 200, Output size: 1
        self.hidden_size1 = 200
        self.hidden_size2 = 200
        
        # Define layers
        self.layer1 = Linear(1, self.hidden_size1)
        self.layer2 = Linear(self.hidden_size1, self.hidden_size2)
        self.layer3 = Linear(self.hidden_size2, 1)
        
        # Learning rate
        self.learning_rate = 0.01



    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Pass through first layer with ReLU activation
        h1 = relu(self.layer1(x))
        
        # Pass through second layer with ReLU activation
        h2 = relu(self.layer2(h1))
        
        # Pass through output layer (no activation)
        output = self.layer3(h2)
        
        return output

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        # Get predictions from the model
        predicted_y = self.forward(x)
        
        # Calculate mean squared error loss
        loss = mse_loss(predicted_y, y)
        
        return loss
 
        

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        # Create dataloader with batch size
        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Train until loss is low enough
        for epoch in range(10000):  # Max epochs
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Calculate loss
                loss = self.get_loss(x, y)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss
            avg_loss = total_loss / num_batches
            
            # Stop if loss is low enough
            if avg_loss < 0.02:
                break

            







class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        
        # Define the architecture
        # Using 2 hidden layers with ReLU activations
        self.hidden_size1 = 200
        self.hidden_size2 = 100
        
        # Define layers
        self.layer1 = Linear(input_size, self.hidden_size1)
        self.layer2 = Linear(self.hidden_size1, self.hidden_size2)
        self.layer3 = Linear(self.hidden_size2, output_size)
        
        # Learning rate
        self.learning_rate = 0.005




    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Pass through first layer with ReLU activation
        h1 = relu(self.layer1(x))
        
        # Pass through second layer with ReLU activation
        h2 = relu(self.layer2(h1))
        
        # Pass through output layer (no activation for logits)
        output = self.layer3(h2)
        
        return output

 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        # Get predictions from the model
        predicted_y = self.run(x)
        
        # Calculate cross entropy loss
        loss = cross_entropy(predicted_y, y)
        
        return loss

    
        

    def train(self, dataset):
        """
        Trains the model.
        """
        # Create dataloader with batch size
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Train until validation accuracy is high enough
        for epoch in range(50):  # Max epochs
            # Training phase
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Calculate loss
                loss = self.get_loss(x, y)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
            
            # Check validation accuracy
            val_accuracy = dataset.get_validation_accuracy()
            print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}")
            
            # Stop if validation accuracy is high enough (slightly above 97% threshold)
            if val_accuracy >= 0.975:
                break



class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel
    before working on this part of the project.)
    """

    def __init__(self):
        # Dataset information
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        super(LanguageIDModel, self).__init__()

        # Model architecture với hidden size 100
        self.hidden_size = 100
        self.output_size = len(self.languages)  # 5 languages

        # Layers cho ký tự đầu tiên (f_initial)
        self.layer1_initial = Linear(self.num_chars, self.hidden_size)
        self.layer2_initial = Linear(self.hidden_size, self.hidden_size)
        
        # Layers cho các ký tự tiếp theo (f)
        self.w_x_recurrent = Linear(self.num_chars, self.hidden_size)
        self.w_hidden_recurrent = Linear(self.hidden_size, self.hidden_size)
        self.layer2_recurrent = Linear(self.hidden_size, self.hidden_size)
        
        # Output layer
        self.output_layer = Linear(self.hidden_size, self.output_size)

        # Learning rate
        self.learning_rate = 0.15


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        xs: list of length L
            each element has shape (batch_size x num_chars)

        Returns:
            Tensor of shape (batch_size x 5) containing logits
        """
        # Xử lý từng ký tự trong chuỗi
        for i in range(len(xs)):
            if i == 0:
                # Ký tự đầu tiên: f_initial(x_0)
                # Z1 = x_0 * W1 + b1
                z1 = self.layer1_initial(xs[i])
                # A1 = ReLU(Z1)
                a1 = relu(z1)
                # h = A1 * W2 + b2 (không có activation)
                h = self.layer2_initial(a1)
            else:
                # Các ký tự tiếp theo: f(h_{i-1}, x_i)
                # Z_one = x_i * W1 + h_{i-1} * W1_hidden + b1_hidden
                z_one = self.w_x_recurrent(xs[i]) + self.w_hidden_recurrent(h)
                # A_one = ReLU(Z_one)
                a_one = relu(z_one)
                # Z_two = A_one * W2_hidden + b2_hidden
                z_two = self.layer2_recurrent(a_one)
                # h = ReLU(Z_two)
                h = relu(z_two)
        
        # Transform final hidden state to language scores
        output = self.output_layer(h)
        
        return output


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        y: one-hot labels with shape (batch_size x 5)
        """
        predicted_y = self.run(xs)
        loss = cross_entropy(predicted_y, y)
        return loss


    def train(self, dataset):
        """
        Trains the model.

        Each batch from the dataloader has shape:
            (batch_size x word_length x num_chars)

        We convert it to:
            (word_length x batch_size x num_chars)
        """
        batch_size = 60
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        for epoch in range(21):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']

                # Reorder dimensions
                x = movedim(x, 0, 1)

                # Convert to list of character tensors
                xs = [x[i] for i in range(len(x))]

                optimizer.zero_grad()
                loss = self.get_loss(xs, y)
                loss.backward()
                optimizer.step()

            val_accuracy = dataset.get_validation_accuracy()
            # print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy >= 0.85:
                break


        





class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        # After convolution of 28x28 with 3x3 kernel, we get 26x26 = 676
        self.hidden_size = 100
        self.layer1 = Linear(676, self.hidden_size)
        self.layer2 = Linear(self.hidden_size, output_size)
        
        # Learning rate
        self.learning_rate = 0.5




    def run(self, x):
        return self(x)
 
    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        # Pass through hidden layer with ReLU
        h = relu(self.layer1(x))
        # Output layer (no activation for logits)
        output = self.layer2(h)
        return output


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        predicted_y = self.run(x)
        loss = cross_entropy(predicted_y, y)
        return loss

     
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        for epoch in range(30):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
            
            val_accuracy = dataset.get_validation_accuracy()
            # print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy >= 0.80:
                break



def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    
    "*** YOUR CODE HERE ***"
    # Get dimensions
    input_height, input_width = input_tensor_dimensions
    kernel_height, kernel_width = weight_dimensions
    
    # Calculate output dimensions
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    # Initialize output tensor list
    output_list = []
    
    # Perform convolution
    for y in range(output_height):
        row = []
        for x in range(output_width):
            # Extract the patch from input
            patch = input[y:y+kernel_height, x:x+kernel_width]
            # Element-wise multiplication and sum (dot product)
            value = tensordot(patch, weight, dims=2)
            row.append(value)
        output_list.append(torch.stack(row))
    
    Output_Tensor = torch.stack(output_list)
    "*** End Code ***"
    return Output_Tensor


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size,layer_size)

        #Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
       
        self.layer_size = layer_size


    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have 
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:
    
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        """YOUR CODE HERE"""
        # Generate Q, K, V matrices by applying linear layers to input
        Q = self.q_layer(input)  # (B, T, C)
        K = self.k_layer(input)  # (B, T, C)
        V = self.v_layer(input)  # (B, T, C)
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        # K^T means transpose last two dimensions
        K_T = movedim(K, 1, 2)  # (B, C, T)
        
        # Matrix multiplication: (B, T, C) @ (B, C, T) = (B, T, T)
        attention_scores = matmul(Q, K_T) / (self.layer_size ** 0.5)
        
        # Apply causal mask
        attention_scores = attention_scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]
        
        # Apply softmax on the last dimension
        attention_weights = softmax(attention_scores, dim=-1)
        
        # Apply attention weights to V: (B, T, T) @ (B, T, C) = (B, T, C)
        output = matmul(attention_weights, V)
        
        return output

     