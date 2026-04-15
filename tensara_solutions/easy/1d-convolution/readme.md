### implementing convolution in 1d

convolution is where you take an input tensor and a mask tensor and then do a weighted sum.

btw you need to align the centre of the mask to each element you compute convolution for.

input size = output size