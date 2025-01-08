#include "Matrix.hpp"

gridEntity Matrix::convolute(gridEntity input_image_section, gridEntity filter, int stride)
{
    int inputHeight = input_image_section.size();
    int inputWidth = input_image_section[0].size();
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();

    // Check if input and filter dimensions are valid
    if (inputHeight < filterHeight || inputWidth < filterWidth) {
        throw std::invalid_argument("Filter dimensions must be smaller than or equal to input dimensions.");
    }

    // Calculate dimensions of the output feature map
    int outputHeight = inputHeight - filterHeight + 1;
    int outputWidth = inputWidth - filterWidth + 1;

    // Initialize the output grid
    gridEntity output(outputHeight, std::vector<double>(outputWidth, 0.0));

    // Perform convolution
    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            double sum = 0.0;

            // Apply filter to the current receptive field
            for (int m = 0; m < filterHeight; ++m) {
                for (int n = 0; n < filterWidth; ++n) {
                    sum += input_image_section[i + m][j + n] * filter[m][n];
                }
            }

            // Store result in the output grid
           

            // I can probaly just apply RELU Here !?????
            output[i][j] = sum;
        }
    }

    return output;
}
