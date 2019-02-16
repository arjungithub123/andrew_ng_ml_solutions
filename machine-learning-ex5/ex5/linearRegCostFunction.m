function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

m = size(X,1);
n = size(X,2);
%X = [ones(m,1) X];
xtranspose = X';
theta_transpose = theta';
predictions = theta_transpose * xtranspose;
ytranspose = y';
error = predictions - ytranspose;
squared_error = error .^ 2;
first_term = sum(squared_error);
p=2*m;
first_term = first_term/p;
factor = lambda/p;
theta_transpose = theta_transpose(:,2:end);
reg_term = theta_transpose .^ 2;
reg_term = sum(reg_term);
final_reg_term = factor * reg_term;
J = first_term + final_reg_term;

%Cost is calculated

%Now for gradient

error = error';

factor_grad = lambda/m;

for j=1:n
	intermediate = error .* X(:,j);
	intermediate = sum(intermediate);
	intermediate = intermediate/m;
	intermediate = intermediate + factor_grad * theta(j);
	grad(j) = intermediate;
end;

intermediate = error .* X(:,1);
intermediate = sum(intermediate);
intermediate = intermediate/m;

grad(1) = intermediate;


% =========================================================================

grad = grad(:);

end
