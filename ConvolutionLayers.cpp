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

std::vector<volumetricEntity> ConvolutionLayers::get_all_training_filter()
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
