function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% Padding zeros to the first column of input X as bias
X = [ones(m, 1) X];
% Calculate the output of first layer
z1 = X*Theta1';
% Calculate the activation of first layer
a2 = sigmoid(z1);
% Padding zeros to the first column of a2 as bias
a2 = [ones(m, 1) a2];
% Calculate the output of second layer
z2 = a2*Theta2';
% Calculate the activation of second layer
a3 = sigmoid(z2);
% Activation output of second layer as the output
htheta = a3;
% Find the prediction of class
[max_value, p] = max(htheta, [], 2);





% =========================================================================


end
