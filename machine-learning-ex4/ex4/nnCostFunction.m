function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Forward Propagation
n = size(X,2);
X = [ones(m,1) X];
xtranspose = X';
layer2 = Theta1 * xtranspose;
columns_of_layer2 = size(layer2,2);
final_layer2 = sigmoid(layer2);
final_layer2 = [ones(1,columns_of_layer2);final_layer2];
layer3 = Theta2 * final_layer2;
final_layer3 = sigmoid(layer3);  
predictions_matrix = final_layer3';

%predictions matrix is the output of forward propagation. It is a 5000 * 10 matrix

actual_output_matrix = zeros(m,num_labels);
for k=1:m
	l = y(k);
	actual_output_matrix(k,l) = 1;
end;

first_term_coef = -actual_output_matrix;

second_term_coef = 1 - actual_output_matrix;

log_predictions_matrix = log(predictions_matrix);

one_minus_predictions_matrix = 1 - predictions_matrix;

log_one_minus_predictions_matrix = log(one_minus_predictions_matrix);

output_vector = zeros(m,num_labels);

output_vector = (first_term_coef .* log_predictions_matrix) - (second_term_coef .* log_one_minus_predictions_matrix);


final_cost = sum(output_vector,2);
final_cost = sum(final_cost);
final_cost = final_cost/m;
J = final_cost;

% cost computed without regularisation

%Now for regularisation

modified_theta1 = Theta1(:,2:n+1);

modified_theta1 = modified_theta1 .^ 2;

theta1_reg = sum(sum(modified_theta1,2));

col_count = size(final_layer2,1);

modified_theta2 = Theta2(:,2:col_count);

modified_theta2 = modified_theta2 .^ 2;

theta2_reg = sum(sum(modified_theta2,2));

reg = theta1_reg + theta2_reg;

p = 2 * m;

lambda = lambda/p;

reg = lambda * reg;

J = J + reg;

%regularised cost obtained

%Now for gradient computation

actual_output_matrix_transpose = actual_output_matrix';

predictions_matrix_transpose = predictions_matrix';

small_delta_L_equals_3 = predictions_matrix_transpose - actual_output_matrix_transpose;

g_of_one_minus_z_2 = 1 - final_layer2;

g_prime_of_z_2 = final_layer2 .* g_of_one_minus_z_2;

small_delta_2 = (Theta2' * small_delta_L_equals_3) .* g_prime_of_z_2;

small_delta_2 = small_delta_2(2:end,:);

capital_delta_1 = zeros(size(Theta1));

capital_delta_1 = small_delta_2 * X;

capital_delta_2 = zeros(size(Theta2));

capital_delta_2 = small_delta_L_equals_3 * final_layer2';

factor = lambda/m;

%regularised_gradient_D_1 = zeros(size(Theta1));

%regularised_gradient_D_1(:,2:end) = (capital_delta_1(:,2:end)/m) + (factor * Theta1(:,2:end));

unregularised_gradient_D_1 = capital_delta_1 * (1/m);

%regularised_gradient_D_2 = zeros(size(Theta2));

%regularised_gradient_D_2(:,2:end) = (capital_delta_2(:,2:end)/m) + (factor * Theta2(:,2:end));

unregularised_gradient_D_2 = capital_delta_2 * (1/m); 

Theta1_grad = unregularised_gradient_D_1;

Theta2_grad = unregularised_gradient_D_2;


Theta1_grad(:,2:end) = (capital_delta_1(:,2:end) * (1/m)) + (factor * Theta1(:,2:end));

Theta2_grad(:,2:end) = (capital_delta_2(:,2:end) * (1/m)) + (factor * Theta2(:,2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
