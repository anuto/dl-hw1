from uwnet import *

def softmax_model():
    l = [make_connected_layer(3072, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(3072, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)

def conv_net():
    # How many operations are needed for a forard pass through this network?
    # Your answer: 
        # 1st layer: 221184
        # 2nd layer: 294912
        # 3rd layer: 294912
        # 4th layer: 294912
        # 5th layer: 2560
        # total: 1108480 operations
        # the comments below are attempts at calculating the 
        # operations per layer, and where we got those numbers 
        # from.  
    
        # the network defined in your_net runs in about 5 million
        # operations instead.  We didn't manage to have enough time
        # to run it with the proper number of operations. But for 
        # the current architecture, 

        # A comparison of conv_net and your_net performance
        # conv_net
        #   training accuracy 61.874
        #   testing accuracy 59.480

        # your_net
        #   training accuracy 65.533
        #   test accuracy 53.450



    l = [ 
            # input to layer - 32x32x3
            # x matrix (post im2col) dimensions are : (w / stride * h / stride X c*size*size;)
                # (32 / 1 * 32 / 1) X (3 * 3 * 3)
            # weight matrix dimensions (size*size*c X sqrt(size*size*c)
                # (3 * 3 * 3) X (8)
            make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),

            # don't worry about maxpooling, no matrix mult
            make_maxpool_layer(32, 32, 8, 3, 2),


            # input to layer - 16x16x8
            # (16 / 1 * 16 / 1) X (8 * 3 * 3)
            # (8 * 3 * 3) X (16)
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
            make_maxpool_layer(16, 16, 16, 3, 2),

            # input to layer - 8x8x16
            # (8 / 1 * 8 / 1) X (16 * 3 * 3)
            # (16 * 3 * 3) X (32)
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
            make_maxpool_layer(8, 8, 32, 3, 2),

            # input to layer - 4x4x32
            # (4 / 1 * 4 / 1) X (32 * 3 * 3)
            # (32 * 3 * 3) X (64)
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
            make_maxpool_layer(4, 4, 64, 3, 2),

            # input to layer - 1x1x256
            # (1) X (256)
            # (256) X (10)
            make_connected_layer(256, 10, SOFTMAX)]

            # 1x1x10
    return make_net(l)

def your_net():
    # Define your network architecture here. It should have 5 layers. How many operations does it need for a forward pass?
    # It doesn't have to be exactly the same as conv_net but it should be close.
    # 5 million operations
    l =[ 
            make_connected_layer(3072, 1500, LRELU),
            make_connected_layer(1500, 500, LRELU),
            make_connected_layer(500, 256, LRELU),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
print("done")
print

print("making model...")

iters = 7500
batch = 140
rate = .01
momentum = .95
decay = .005

m = your_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# Training for the same amount of time (2000 iterations) for the fully connected

