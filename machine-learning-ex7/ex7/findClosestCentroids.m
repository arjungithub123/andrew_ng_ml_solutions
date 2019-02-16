function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1);
for i=1:m
	x_row = X(i,:);
	dist_vector = zeros(K,1);
	for mu=1:K
		x_row_minus_mu_k = x_row - centroids(mu,:);
		x_row_minus_mu_k_sq = x_row_minus_mu_k .^ 2;
		min_distance = sum(x_row_minus_mu_k_sq);
		min_distance = sqrt(min_distance);
		dist_vector(mu) = min_distance;
	end;
	[val,index] = min(dist_vector);
	idx(i) = index;
end;





% =============================================================

end

