function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

test = [0.01;0.03;0.1;0.3;1;3;10;30];
cols = size(test,1);
k = 1;
min_error = 100000000;
C_temp = 0;
sigma_temp = 0;
for i=1:cols
	for j=1:cols
		C = test(i);
		sigma = test(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		err= mean(double(predictions ~= yval));
		if err < min_error
			min_error = err;
			C_temp = C;
			sigma_temp = sigma;
		end;
	end;
end;
C = C_temp;
sigma = sigma_temp;




% =========================================================================

end
