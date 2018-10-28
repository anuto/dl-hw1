#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

/*
Maxpooling is another core building block of convolutional neural networks. 
Implementing maxpooling will be similar to implementing convolutions in some 
ways, you will have to iterate over the image, process a window of pixels with 
me fixed size, and in this case find the maximum value to put into the output.

6.1 forward_maxpool_layer
Write the forward method to find the maximum value in a given window size,
 moving by some strided amount between applications. Note: maxpooling happens
  on each channel independently.

6.2 backward_maxpool_layer
The backward method will be similar to forward. Even though the window size
 may be large, only one element contributed to the error in the prediction 
 so we only backpropagate our deltas to a single element in the input per 
 window. Thus, you'll iterate through again and find the maximum value and 
 then backpropagate error to the appropriate element in prev_delta 
 corresponding to the position of the maximum element.
*/
// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

