function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

xtranspose = X';
thetatranspose = theta';
ytranspose = y';
input_to_sigmoid = thetatranspose * xtranspose;

input_to_log1 = sigmoid(input_to_sigmoid);

elementwise_mul_with_ytranspose = log(input_to_log1);

first_term = ytranspose .* elementwise_mul_with_ytranspose;

one_minus_ytranspose = 1 - ytranspose; 

input_to_log2 = 1 - input_to_log1;

elementwise_mul_with_one_minus_ytranspose = log(input_to_log2);

second_term = one_minus_ytranspose .* elementwise_mul_with_one_minus_ytranspose;

term_in_brackets = first_term + second_term;

term_in_brackets = -1 * term_in_brackets;

term_in_brackets = sum(term_in_brackets);

J = term_in_brackets/m;

J = J';

grad = grad';

sum_term_in_gradient1 = input_to_log1 - ytranspose;

sum_term_in_gradient = sum_term_in_gradient1 * X;

answer = sum_term_in_gradient / m;

grad = answer;

grad = grad';






% =============================================================

end
