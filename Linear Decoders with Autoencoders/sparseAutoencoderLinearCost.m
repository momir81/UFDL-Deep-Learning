function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------      

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

[ndims, m] = size(data);


z2 = zeros(hiddenSize, m);
z3 = zeros(visibleSize, m);
a1 = zeros(ndims, m);
a2 = zeros(size(z2));
a3 = zeros(size(z3));
%autoencode use inputs as target values
y  = zeros(ndims, m);

a1 = data;
y = data;

deltaW1 = zeros(size(W1));
deltab1 = zeros(size(b1));
JW1grad = zeros(size(W1));
Jb1grad = zeros(size(b1));
deltaW2 = zeros(size(W2));
deltab2 = zeros(size(b2));
JW2grad = zeros(size(W2));
Jb2grad = zeros(size(b2));

z2 = W1 * data +repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2,1,m);
a3 = z3;

rho = zeros(hiddenSize, 1);
rho = (1. / m) * sum(a2, 2);
sp = sparsityParam;
sparsity_delta = -sp ./rho+(1-sp)./(1-rho);
delta3 = -(y-a3);
delta2 = (W2' * delta3 +beta * repmat(sparsity_delta,1,m)) .* sigmoidGrad(z2);

deltaW1 = delta2 * a1';
deltab1 = sum(delta2,2);
deltaW2 = delta3 * a2';
deltab2 = sum(delta3,2);

W1grad = (1. / m) * deltaW1 + lambda * W1;
b1grad = (1. / m) * deltab1;
W2grad = (1. / m) * deltaW2 + lambda * W2;
b2grad = (1. / m) * deltab2;

cost = (1. / m) * sum((1. / 2) * sum((a3 - y).^2)) + ...
    (lambda / 2.) * (sum(sum(W1.^2)) + sum(sum(W2.^2))) + ...
    beta * sum( sp*log(sp./rho) + (1-sp)*log((1-sp)./(1-rho)) );


function grad = sigmoidGrad(x)
    e_x = exp(-x);
    grad = e_x ./ ((1 + e_x).^2); 
end
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end
