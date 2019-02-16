function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
rows_x1 = size(x1,1);
rows_x2 = size(x2,1);
final_sum = 0;
difference = 0;
sq = 0;
for i=1:rows_x1
	difference = x1(i) - x2(i);
	sq = sq + difference ^ 2;
end;
two_sigma_sq = sigma ^ 2;
two_sigma_sq = 2 * two_sigma_sq;

sq = -sq;
sim = exp(sq/two_sigma_sq);





% =============================================================
    
end
