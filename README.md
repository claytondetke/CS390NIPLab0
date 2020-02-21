# CS390NIPLab0

# How I implemented the custom NN:
For this part, I simply implemented the sigmoid functions, then added the function for training the model, in which it flattens the input, then for every epoch that it needs to do, it will iterate through the given data, and through backpropogation, it will update the weights every time, or if minibatches are being used, it will store the changes, and update the weights after every minibatch is complete. Then for the classifier, it simply flattens the input, then takes the classifier with the highest probability.

# How I implemented the TF keras NN:
This part was far easier, and runs far quicker, which is probably a major point of this lab, to get us to see how great tensorflow and keras are. For this, I simply made a keras sequential model, and added a layer to flatten the input, a layer of neurons with relu as an activation function, and an output layer with softmax as it's activation function. Then I compiled it with the suggested optimizer and loss type from the in-class slides, set the metric to accuracy, and fit the model to the data. The classifier was almost identical to the one from the previous part.
