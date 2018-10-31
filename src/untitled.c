// go through all the rows
        for(cur_row = 0; cur_row < im.h; cur_row++) {

            // go through a single row
            for(cur_col = 0; cur_col < im.w; cur_col++) {
                // cur_row represents index in the size x size kernel
                // cur_col represents the filter #, starting at 0
                // cur_col can be used to calculate the top left corner of the kernel
                // cur_row can be used to calculate the index from top left corner of kernel
                int im_col = cur_col % im.w - 1;
                int im_row = cur_col / im.w - 1;

                im_col += cur_row % size;
                im_row += cur_row / size;

                int col_array_index = cur_row * cols + cur_col + (size * size * channel * cols);
                if (im_col != -1 && im_row != -1 && im_col != im.w && im_row != im.h) {
                    set_pixel(im, im_col, im_row, channel, get_pixel(im, im_col, im_row, channel) 
                        + col.data[col_array_index]); 
                    // channel_im.data[im_row * im.w + im_col] += col.data[col_array_index];
                }
            }
        }