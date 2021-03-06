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


for jj = 2:length(theta)

    temp_theta(jj) = theta(jj)^2;
end

J = (((1/ (m)) * (sum((-y .* log(sigmoid(X * theta))) - ((1 .- y) .* log(1-sigmoid(X * theta)))))) + ((lambda/ (2 * m)) * sum(temp_theta))); 



h = sigmoid(X*theta);
shift_theta = theta(2:size(theta));

theta_reg = [0;shift_theta];

% grad_zero = (1/m)*X(:,1)'*(h-y);
% grad_others = (1/m)*(shift_x'*(h - y)+lambda*shift_theta);
% grad      = cat(1, grad_zero, grad_rest);

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);

% =============================================================

end
