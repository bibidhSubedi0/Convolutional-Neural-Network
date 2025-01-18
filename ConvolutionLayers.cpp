#include "ConvolutionLayers.hpp"
#include "Matrix.hpp"
#include "ImageInput.hpp"

ConvolutionLayers::ConvolutionLayers(gridEntity main_image) : raw_image(main_image) {
	// Define predefined_filters for the network
    this->predefined_filters.push_back(Filters::STRONG_VERTICAL_EDGE_DETECTION);
    this->predefined_filters.push_back(Filters::STRONG_HORIZONTAL_EDGE_DETECTION);
    this->predefined_filters.push_back(Filters::STRONG_DIAGONAL_EDGE_DETECTION);
    
    this->no_of_filters_used = 3;

    // Initilize the input channel to next convolution layer, values will put put after pooling is completed from the first layer
    input_channels.resize(this->no_of_filters_used);



    // Initilize the trainable filters
    // ------------------------------------------------------------------
    // Add a mechnism to intilize the filters to some random values
    // ------------------------------------------------------------------

    // Dimention rows = 3, cols =3, depth = no.of filters
    // i will use 4 filter ub second CL so i will use 4 3x3xN filters where N is the no of. filters in last layer
    
    this->no_of_filters_in_second_CL= 3;


    // initilize the filters to some random values
    for (int i = 0; i < this->no_of_filters_in_second_CL; i++)
    {
        volumetricEntity temp;
        
        // Initilize each filter to a random number
        for (int dep = 0; dep < this->no_of_filters_used; dep++)
        {
            gridEntity sheet;
            CNN_Matrix::Matrix::randomize_all_values(sheet, 3, 3);
            temp.push_back(sheet);
        }

        this->training_filters.push_back(temp);
    }

}

std::vector<gridEntity> ConvolutionLayers::get_all_predefined_filter()
{
    return this->predefined_filters;
}

void ConvolutionLayers::apply_normalaization_universal(gridEntity& to_normailize)
{
    // idk if this will work but fuck it
    int min = 0;
    int max = 800;


        for (int i = 0; i < to_normailize.size(); i++)
        {
            for (int j = 0; j < to_normailize.at(0).size(); j++)
            {
                to_normailize.at(i).at(j) = (to_normailize.at(i).at(j) - min) / (max - min);
            }
        }
    
}


gridEntity ConvolutionLayers::apply_filter_universal(gridEntity image, gridEntity filt, int stride = 1)
{
    gridEntity f_map = CNN_Matrix::Matrix::convolute(image, filt, stride);
    return f_map;

}

void ConvolutionLayers::activate_feature_map_using_RELU_universal(gridEntity &to_activate)
{
    for (int i = 0; i < to_activate.size(); i++)
    {
        for (int j = 0; j < to_activate.at(0).size(); j++)
        {
            if (to_activate.at(i).at(j) < 0)
            {
                to_activate.at(i).at(j) = 0;
            }

        }
    }
    
}

void ConvolutionLayers::activate_feature_map_using_SIGMOID(gridEntity& to_activate)
{
    for (int i = 0; i < to_activate.size(); i++)
    {
        for (int j = 0; j < to_activate.at(0).size(); j++)
        {
            to_activate.at(i).at(j) = (1) / (1 + pow(2.71828, -to_activate.at(i).at(j)));
        }
    }

}

gridEntity ConvolutionLayers::unpool_without_indices( gridEntity& pooled_gradients,  gridEntity& original_filter_map, int stride = 2, int poolHeight = 2, int poolWidth = 2)
{
    // Get dimensions of the original feature map
    int originalHeight = original_filter_map.size();
    int originalWidth = original_filter_map[0].size();

    // Initialize the unpooled gradient matrix with zeros
    gridEntity unpooled_gradients(originalHeight, std::vector<double>(originalWidth, 0.0));

    // Iterate through the pooled gradients
    for (int i = 0; i < pooled_gradients.size(); i++)
    {
        for (int j = 0; j < pooled_gradients[0].size(); j++)
        {
            // Identify the pooling window in the original feature map
            int startRow = i * stride;
            int startCol = j * stride;

            int endRow = std::min(startRow + poolHeight, originalHeight);
            int endCol = std::min(startCol + poolWidth, originalWidth);

            // Find the position of the maximum value in the pooling window
            double max_val = -1e9; // Use a very small number as initial max
            int maxRow = startRow, maxCol = startCol;

            for (int row = startRow; row < endRow; row++)
            {
                for (int col = startCol; col < endCol; col++)
                {
                    if (original_filter_map[row][col] > max_val)
                    {
                        max_val = original_filter_map[row][col];
                        maxRow = row;
                        maxCol = col;
                    }
                }
            }

            // Assign the pooled gradient to the max position in the unpooled gradient
            unpooled_gradients[maxRow][maxCol] += pooled_gradients[i][j];
        }
    }

    return unpooled_gradients;
}



gridEntity ConvolutionLayers::apply_pooling_univeral(gridEntity to_pool, int stride = 2)
{
    // define kernal
    int poolHeight = 2;
    int poolWidth = 2;

    // get dimentions of the feature_map
    int inputHeight = to_pool.size();
    int inputWidth = to_pool[0].size();


    gridEntity pooled;
    // Perform pooling

    for (int i = 0; i < inputHeight; i += stride)
    {

        std::vector<double> pool_row;
        for (int j = 0; j < inputWidth; j += stride)
        {

            double max_val = 0;
            for (int kh = i; kh < i + poolHeight; kh++)
            {
                for (int kw = j; kw < j + poolWidth; kw++)
                {

                    if ((kh < inputHeight) && (kw < inputWidth))
                    {
                        // std::cout << "curr x: " << kh << "  curry y: " << kw << "    :    " << feature.at(kh).at(kw) << std::endl;
                        max_val = to_pool.at(kh).at(kw) > max_val ? to_pool.at(kh).at(kw) : max_val;
                    }
                }
            }
            pool_row.push_back(max_val);

        }
        pooled.push_back(pool_row);
    }
    return pooled;
}



gridEntity ConvolutionLayers::get_raw_input_image()
{
    return this->raw_image;
}

std::vector<gridEntity>& ConvolutionLayers::get_feature_map() {
    return this->feature_maps;
}


std::vector<gridEntity>& ConvolutionLayers::get_pool_map() {
    return this->pool_maps;
}

volumetricEntity& ConvolutionLayers::get_input_channels() {
    return this->input_channels;
}

std::vector<volumetricEntity>& ConvolutionLayers::get_all_training_filter()
{
    return this->training_filters;
}

volumetricEntity &ConvolutionLayers::get_output_feature_maps()
{
    return this->ouput_feature_maps;
}


volumetricEntity& ConvolutionLayers::get_final_pool_maps()
{
    return this->final_pool_maps;
}

std::vector<volumetricEntity> ConvolutionLayers::compute_filter_gradients(
    const std::vector<gridEntity>& inputChannels,        // Input to the convolutional layer
    const std::vector<gridEntity>& outputGradients,      // Gradients w.r.t output (from unpooling)
    int stride = 1                                       // Stride used during convolution
) {
    std::vector<volumetricEntity> filterGradients;

    int numFilters = outputGradients.size();
    int numChannels = inputChannels.size();

    // Loop through each filter
    for (int filterIdx = 0; filterIdx < numFilters; ++filterIdx) {
        volumetricEntity filterGradient(numChannels);

        // Loop through each input channel
        for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
            // Perform convolution between the input channel and output gradient
            gridEntity gradient = apply_filter_universal(
                inputChannels[channelIdx],
                outputGradients[filterIdx],
                stride
            );

            // Store the resulting gradient
            filterGradient[channelIdx] = gradient;
        }

        // Append the filter gradient (volumetric entity)
        filterGradients.push_back(filterGradient);
    }

    return filterGradients;
}




void ConvolutionLayers::update_filters_with_gradients(std::vector<volumetricEntity>& filters,const std::vector<volumetricEntity>& gradients,double learningRate)
{
    // Loop through each filter (volumetric entity)
    for (size_t filterIdx = 0; filterIdx < filters.size(); ++filterIdx)
    {
        volumetricEntity& filterTensor = filters[filterIdx];                // Current filter tensor
        const volumetricEntity& gradientTensor = gradients[filterIdx];     // Corresponding gradient tensor

        // Ensure the tensor dimensions match
        if (filterTensor.size() != gradientTensor.size())
        {
            std::cerr << "Error: Mismatch in tensor dimensions for filter " << filterIdx << "\n";
            continue;
        }

        // Update each 2D filter (gridEntity) in the tensor
        for (size_t channelIdx = 0; channelIdx < filterTensor.size(); ++channelIdx)
        {
            gridEntity& filter = filterTensor[channelIdx];                 // Current 2D filter
            const gridEntity& gradient = gradientTensor[channelIdx];       // Corresponding 2D gradient

            // Ensure 2D dimensions match
            if (filter.size() != gradient.size())
            {
                std::cerr << "Error: Mismatch in filter dimensions for channel " << channelIdx << " of filter " << filterIdx << "\n";
                continue;
            }

            // Update filter values
            for (size_t i = 0; i < filter.size(); ++i)
            {
                for (size_t j = 0; j < filter[i].size(); ++j)
                {
                    filter[i][j] -= learningRate * gradient[i][j];
                }
            }
        }
    }
}

