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


 %X = [X ones(m,1)];

%J = (1 / ( 2 * m) ) *  sum((( X * theta) - y) .^ 2); 



%for jj = 2:length(theta)

 %   temp_theta(jj-1) = theta(jj)^2;
%end


%J = J + ((lambda/ (2 * m)) * sum(temp_theta));

%X0 = X(:,1);

%grad = (1/m) * ((( X * theta) - y)' * X0);

%the2 = theta(2:end,:);
% for jj = 2:length(theta)
%grad = grad + ((1 / m) * ( sum((( X * theta ) - y) * the2) + sum(the2)));
%end


%diff = X*theta - y;
% calculate penalty
% excluded the first theta value
%theta1 = [0 ; theta(2:end, :)];
%p = lambda*(theta1'*theta1);
%J = (diff'*diff)/(2*m) + p/(2*m);

% calculate grads
%grad = (X'*diff+lambda*theta1)/m;

theta0 = [0; theta(2:end)];

J = (1 / (2*m)) * (sum((X * theta - y) .^ 2) + lambda * theta0' * theta0);
grad = (1/m) * (X' * (X * theta - y) + lambda * theta0);


% =========================================================================

grad = grad(:);

end