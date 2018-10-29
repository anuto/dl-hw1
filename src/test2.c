#include <stdio.h>
#include "matrix.h"
#include "matrix.c"
#include "image.h"
#include "image.c"
#include "convolutional_layer.c"
#include "uwnet.h"
#include "activations.c"

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

int main(int argc, char * argv[])
{
	image im = make_image(5, 5, 3);
	float temp[] = {1.0, 2.0, 3.0, 4.0, 5.0,
				6.0, 7.0, 8.0, 9.0, 10.0,
				11.0, 12.0, 13.0, 14.0, 15.0,
				16.0, 17.0, 18.0, 19.0, 20.0,
				21.0, 22.0, 23.0, 24.0, 25.0,
				// 2nd channel
				1.01, 2.01, 3.01, 4.01, 5.01,
				6.01, 7.01, 8.01, 9.01, 10.01,
				11.01, 12.01, 13.01, 14.01, 15.01,
				16.01, 17.01, 18.01, 19.01, 20.01,
				21.01, 22.01, 23.01, 24.01, 25.0,
				// 3rd channel
				1.02, 2.02, 3.02, 4.02, 5.02,
				6.02, 7.02, 8.02, 9.02, 10.02,
				11.02, 12.02, 13.02, 14.02, 15.02,
				16.02, 17.02, 18.02, 19.02, 20.02,
				21.02, 22.02, 23.02, 24.02, 25.02};
	im.data = temp;

	print_image(im);

	// col_matrix has
	//		filter size x filter size x channels rows
	// 		image width * image height / stride cols
	matrix result = im2col(im, 3, 1);
	print_matrix(result);
	image im2 = make_image(5, 5, 3);
    col2im(result, 3, 1, im2);

    printf("now printing image 2 after col2im\n");
    print_image(im2);


}


