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
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
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
        
        Hint   : You can use ones(dim) to create a tensor of dimension dim.
        """    
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
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
        "*** YOUR CODE HERE ***"
        return matmul(x, self.w.T).squeeze()

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
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
            "*** YOUR CODE HERE ***"
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
        "*** YOUR CODE HERE ***"
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
        "*** YOUR CODE HERE ***"
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
        "*** YOUR CODE HERE ***"
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
        "*** YOUR CODE HERE ***"
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
        "*** YOUR CODE HERE ***"
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
        """ YOUR CODE HERE """
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
        """ YOUR CODE HERE """
        # Get predictions from the model
        predicted_y = self.run(x)
        
        # Calculate cross entropy loss
        loss = cross_entropy(predicted_y, y)
        
        return loss

    
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
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
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        self.num_languages = len(self.languages)

        # Model architecture
        self.hidden_size = 128

        # Embedding layer (char -> hidden)
        self.embedding = Linear(self.num_chars, self.hidden_size)

        # Recurrent weights (Simple RNN, GRU-like logic)
        self.w_x = Linear(self.hidden_size, self.hidden_size)
        self.w_h = Linear(self.hidden_size, self.hidden_size)

        # Output layer
        self.output_layer = Linear(self.hidden_size, self.num_languages)

        # Learning rate
        self.learning_rate = 0.001

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        batch_size = xs[0].shape[0]

        # Initialize hidden state h_0 = 0
        h = torch.zeros(batch_size, self.hidden_size)

        # Recurrent processing over characters
        for x in xs:
            x_emb = self.embedding(x)                 # (batch_size x hidden)
            h = relu(self.w_x(x_emb) + self.w_h(h))  # h_t

        # Final logits
        logits = self.output_layer(h)
        return logits

    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        logits = self.run(xs)
        loss = cross_entropy(logits, y)
        return loss
        

    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        for epoch in range(20):
            for batch in dataloader:
                x = movedim(batch['x'], 0, 1)  # (L x batch x num_chars)
                y = batch['label']

                xs = [x[i] for i in range(len(x))]

                optimizer.zero_grad()
                loss = self.get_loss(xs, y)
                loss.backward()
                optimizer.step()

            val_accuracy = dataset.get_validation_accuracy()
            print(f"Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy >= 0.85:
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
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    input_h, input_w = input_tensor_dimensions
    weight_h, weight_w = weight_dimensions

    output_h = input_h - weight_h + 1
    output_w = input_w - weight_w + 1

    Output_Tensor = torch.empty((output_h, output_w))
    Output_Tensor.fill_(0)

    for i in range(output_h):
        for j in range(output_w):
            region = input[i:i+weight_h, j:j+weight_w]
            Output_Tensor[i, j] = (region * weight).sum()
    
    "*** End Code ***"
    return Output_Tensor



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
        self.hidden_layer = Linear(26 * 26, 128)
        self.output_layer = Linear(128, output_size)




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
        x = relu(self.hidden_layer(x))
        return self.output_layer(x)


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
        predictions = self.forward(x)
        return cross_entropy(predictions, y)

     
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(12):
            total_loss = 0.0

            for sample in dataloader:
                x = sample['x']
                y = sample['label']

                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader)}")




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
        Q = self.q_layer(input)   # (B, T, C)
        K = self.k_layer(input)   # (B, T, C)
        V = self.v_layer(input)   # (B, T, C)
        
        # Q^T: swap last two dims
        Q_t = movedim(Q, -1, -2)  # (B, C, T)

        # Scaled dot-product attention
        scale = self.layer_size ** 0.5
        scores = matmul(K, Q_t) / scale   # (B, T, T)

        # Apply causal mask
        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float('-inf')
        )[0]

        # Softmax on last dimension
        attn = softmax(scores, dim=-1)    # (T, T)

        # Attention output
        output = matmul(attn, V)          # (T, C)

        return output

     