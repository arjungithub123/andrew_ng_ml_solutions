function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
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

term_in_brackets =sum(term_in_brackets);

J = term_in_brackets/m;

square_theta = thetatranspose .^ 2;

square_theta(1) = 0;

square_theta_sum = sum(square_theta);

factor1 = 2 * m;

factor = lambda/factor1; 

square_theta_sum = factor * square_theta_sum;

J = J + square_theta_sum;

J = J';

grad = grad';

sum_term_in_gradient1 = input_to_log1 - ytranspose;

sum_term_in_gradient = sum_term_in_gradient1 * X;

answer = sum_term_in_gradient / m;

grad = answer;

final_ans = zeros(1,length(theta));

new_theta = (lambda/m) * thetatranspose;
answer = answer + new_theta;

final_ans = answer;

final_ans(1) = grad(1);

grad = final_ans';











% =============================================================

grad = grad(:);

end
