#include <stdio.h>
#include "matrix.h"
#include "matrix.c"
#include "image.h"
#include "image.c"
#include "convolutional_layer.c"
#include "uwnet.h"
#include "activations.c"
#include "maxpool_layer.c"


void print_image(image im) 
{
	int i;
	int j;
	int k;
	for (i = 0; i < im.c; i++) 
	{
		printf("	new channel: %d\n", i);
		image channel = get_channel(im, i);
		for (j = 0; j < im.h; j++)
		{
			for (k = 0; k < im.w; k++) 
			{
				printf("%lf ", channel.data[j * im.w + k]);
			}
			printf("\n");
		}
	}
}

void test_fwd_maxpool() {
    // Params.
//    int channels = 2;
//    int batch = 2;
//    int width = 2, height = 2;

    int channels = 3;
    int batch = 1;
    int width = 7, height = 7;
    int stride = 1, size = 3;
    int cols = width * height * channels;

    // Init.
    layer max_pool = make_maxpool_layer(width, height, channels, size, stride);
    matrix in = make_matrix(batch, cols);
    matrix prev_delta = make_matrix(batch, cols);

    // Raw flat data.
    int counter = 0;
    int len = batch * channels * width * height;
    assert(len == 147);
    float data[147] = {
        0,  1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 48,

        0,  1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 48,

        0,  1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 48,
    };

//    assert(len == 64);
//    float data[64] = {
//        -7,  6, -1,  3,  9,  9,  6, -9,
//         3, -8,  0,  7, 10,  8, -3, 10,
//        -4,  2, -6,  4, -7,  5,  5,  7,
//        -3, -9,  1,  8, -8,  9, -1, -5,
//        -7, 10, -9, -5,  9, -8, -7, 10,
//        -5,  5,  9,  4, 10, -8,  7,  6,
//        -3,  8,  0,  2,  2, -3, -2,  5,
//         4, -6,  7, -3,  1,  4, 10,  0,
//    };

//    assert(len == 2 * 2 * 2 * 2);
//    float data[2 * 2 * 2 * 2] = {
//        1, 2, 3, 4,
//        5, 6, 7, 8,
//
//        1, 2, 3, 4,
//        5, 6, 7, 8,
//    };

    for (int b = 0; b < batch; b++) {
      for (int ch = 0; ch < channels; ch++) {
        for (int r = 0; r < height; r++) {
          for (int c = 0; c < width; c++) {
              assert(counter < len);

              in.data[b * width * height * channels
                + ch * (width * height)
                + r * (width)
                + c] = data[counter];
              counter += 1;
          }
        }
      }
    }

    print_matrix(in);
    printf("------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    matrix a = forward_maxpool_layer(max_pool, in);
    print_matrix(a);
    printf("------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
   backward_maxpool_layer(max_pool, prev_delta);
}

void test_convolutional_layer() {
    int im_size = 5;
    float *data = calloc(im_size * im_size, sizeof(float));
    for (int i = 0; i < im_size * im_size; i++) {
        data[i] = i + 1;
    }
    matrix og = make_matrix(im_size, im_size);
    og.data = data;
    print_matrix(og);
    image im = float_to_image(data, im_size, im_size, 2);
    // print_matrix(make_matrix());
    matrix temp = im2col(im, 3, 1);
    print_matrix(temp);
    col2im(temp, 3, 1, im);
}


int main(int argc, char * argv[])
{
// 	image im = make_image(5, 5, 3);
// 	float temp[] = {1.0, 2.0, 3.0, 4.0, 5.0,
// 				6.0, 7.0, 8.0, 9.0, 10.0,
// 				11.0, 12.0, 13.0, 14.0, 15.0,
// 				16.0, 17.0, 18.0, 19.0, 20.0,
// 				21.0, 22.0, 23.0, 24.0, 25.0,
// 				// 2nd channel
// 				1.01, 2.01, 3.01, 4.01, 5.01,
// 				6.01, 7.01, 8.01, 9.01, 10.01,
// 				11.01, 12.01, 13.01, 14.01, 15.01,
// 				16.01, 17.01, 18.01, 19.01, 20.01,
// 				21.01, 22.01, 23.01, 24.01, 25.0,
// 				// 3rd channel
// 				1.02, 2.02, 3.02, 4.02, 5.02,
// 				6.02, 7.02, 8.02, 9.02, 10.02,
// 				11.02, 12.02, 13.02, 14.02, 15.02,
// 				16.02, 17.02, 18.02, 19.02, 20.02,
// 				21.02, 22.02, 23.02, 24.02, 25.02};
// 	im.data = temp;

// 	print_image(im);

// 	// col_matrix has
// 	//		filter size x filter size x channels rows
// 	// 		image width * image height / stride cols
// 	matrix result = im2col(im, 3, 1);
// 	print_matrix(result);

// 	image im2 = make_image(5, 5, 3);
//     col2im(result, 3, 1, im2);
// 	printf("now printing image 2 after col2im\n");
//     print_image(im2);
	// test_convolutional_layer();
	test_fwd_maxpool();
}




