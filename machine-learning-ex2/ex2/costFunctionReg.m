function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
%Calculate htheta (prediction)
htheta = sigmoid(X*theta);
%Calculate the logistic regression cost for each example
cost = -y.*log(htheta)-(1-y).*log(1-htheta);
%Calculate the regularization term, exclude theta(0)
reg = (lambda/2)*sum(theta(2:length(theta)).^2);
%Calculate the total logistic regression cost
J = (1/m)*(sum(cost)+reg);
%Set the theta(0) = 0 so that, dreg excludes theta(0)
theta_temp = theta;
theta_temp(1) = 0;
%Calculate gradient of regularization term
dreg = lambda*theta_temp;
%Calculate the gradient, exclude theta(0)
grad = (1/m)*(X'*(htheta-y)+dreg);




% =============================================================

end
