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
double Matrix::genRandomNumber()
{
    std::random_device rd;
    std::mt19937 gen(rd()); // MerseNetworke Twister 19937 generator seeded with rd

    // Define the distribution for floating point numbers between 0 and 1
    std::uniform_real_distribution<float> dis(-3, 3);

    // Generate a random float number between 0 and 1 with 3 decimal digits
    float random_number = dis(gen);
    return random_number;
}
void Matrix::randomize_all_values(gridEntity &mat,int numRows,int numCols)
{
    gridEntity temp;
    for (int i = 0; i < numRows; i++)
    {
        std::vector<double> cols;
        for (int j = 0; j < numCols; j++)
        {
            double r = 0.00;
            r = Matrix::genRandomNumber();
            cols.push_back(r);
        }
        temp.push_back(cols);
    }
    mat = temp;
}


double Matrix::sum_of_all_elements(gridEntity matrix)
{
    double sum = 0.0;
    for (const auto& row : matrix) {
        for (double elem : row) {
            sum += elem;
        }
    }
    return sum;
}

gridEntity Matrix::sum_of_all_matrix_elements(std::vector<gridEntity> all_matrices)
{
    int rows = all_matrices[0].size();
    int cols = all_matrices[0][0].size();

    gridEntity result(rows,std::vector<double>(cols,0));

    for (const auto& mat : all_matrices) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[i][j] += mat[i][j];
            }
        }
    }
    return result;
}