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
C_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i=1:length(C_array)
	for j=1:length(sigma_array)
		model = svmTrain(X, y, C_array(i), @(x1, x2) gaussianKernel(x1, x2, sigma_array(j)));
		predictions = svmPredict(model, Xval);
		predictions_error(i,j) = mean(double(predictions ~= yval));
	end
end

% Convert the predictions_error matrix to a vector and find the minimum value index (in row)
[min_value index_min] = min(predictions_error(:));
% Take the minimum value index (in row) and convert it to the real index of it
[index_min_row, index_min_col] = ind2sub(size(predictions_error),index_min);
% Get the C and sigma values
C = C_array(index_min_row);
sigma = sigma_array(index_min_col);




% =========================================================================

end
