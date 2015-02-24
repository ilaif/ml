load('ex3data1.mat'); % Loads X - The data-set to predict
load('ex3weights.mat'); % Loads Theta1, Theta2 - A trained Neural Network.

% Compute sigmoid functoon
function g = sigmoid(z)
g = 1.0 ./ (1.0 + exp(-z));
end

% The prediction function
function p = predict(Theta1, Theta2, X)
a1 = [ones(size(X), 1), X]; % Extend X with 'bias' node, this is "a(1)".
z2 = a1 * Theta1'; % Multiply Theta1 by X' to activate first layer.
a2 = [ones(size(z2), 1), sigmoid(z2)]; % Extend X with 'bias' node, this is "a(2)".
z3 = a2 * Theta2'; % Multiply Theta2 by layer_1 to activate second layer.
a3 = sigmoid(z3); % In our case the third layer is actually the final hypothesis h(theta).
h = a3;

[pred_max, index_max] = max(h, [], 2); % Collect the predictions for each sample.
p = index_max; % Return it as a column vector.
end

pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);