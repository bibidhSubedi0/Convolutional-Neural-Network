#include "ConvolutionLayers.hpp"
#include "Matrix.hpp"
#include "ImageInput.hpp"

ConvolutionLayers::ConvolutionLayers(gridEntity main_image) : raw_image(main_image) {
	// Define Filters for the network
    
    // THIS IS NOT THE MOST OPTIMAL PLACE TO KEEP THESE FILTERS BUT FUCK IT WE BALL
    
    // Purpose: Detect strong vertical edges in a larger receptive field
    gridEntity filter1 = {
    {1, 0, -1, 0, 1},
    {1, 0, -1, 0, 1},
    {1, 0, -1, 0, 1},
    {1, 0, -1, 0, 1},
    {1, 0, -1, 0, 1}  
    };
    
    // Purpose: Detect horizontal edges with a broader coverage
    gridEntity filter2 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0},
        {-1, -1, -1, -1, -1},
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}  
    };

    // Purpose: Detect diagonal edges (top-left to bottom-right)
    gridEntity filter3 = {
        {1, 1, 0, -1, -1},
        {1, 1, 0, -1, -1},
        {0, 0, 0, 0, 0},
        {-1, -1, 0, 1, 1},
        {-1, -1, 0, 1, 1}  
    };


    this->filters.push_back(filter1);
    this->filters.push_back(filter2);
    this->filters.push_back(filter3);

}

gridEntity ConvolutionLayers::get_filter(int index)
{
    return this->filters.at(index);
}


void ConvolutionLayers::apply_filter(gridEntity filt,int stride=1)
{
    // Write the convlution for filter and image and return the un-activaed feature map
    // fuck this is going to be long
    gridEntity f_map = Matrix::convolute(raw_image,filt);
    this->feature_maps.push_back(f_map);
}



void ConvolutionLayers::activate_feature_maps_using_RELU()
{
    for(int xx=0;xx<this->feature_maps.size();xx++)
    {
        for (int i = 0; i < this->feature_maps.at(xx).size(); i++)
        {
            for (int j = 0;j< this->feature_maps.at(xx).at(0).size(); j++)
            {
                if (this->feature_maps.at(xx).at(i).at(j) < 0)
                {
                    this->feature_maps.at(xx).at(i).at(j) = 0;
                }

            }
        }
    }
}


void ConvolutionLayers::apply_normalaization()
{
    // idk if this will work but fuck it
    int min = 0;
    int max = 800;

    for (int xx = 0; xx < this->feature_maps.size(); xx++)
    {
        for (int i = 0; i < this->feature_maps.at(xx).size(); i++)
        {
            for (int j = 0; j < this->feature_maps.at(xx).at(0).size(); j++)
            {
                this->feature_maps.at(xx).at(i).at(j) = (this->feature_maps.at(xx).at(i).at(j) - min) / (max - min);
            }
        }
    }
}



void ConvolutionLayers::apply_pooling(gridEntity feature, int stride = 2)
{
    // define kernal
    int poolHeight = 2;
    int poolWidth = 2;

    // get dimentions of the feature_map
    int inputHeight = feature.size();
    int inputWidth = feature[0].size();


    gridEntity pooled;
    // Perform pooling

    for (int i = 0; i < inputHeight; i+=stride)
    {

        std::vector<double> pool_row;
        for (int j = 0; j < inputWidth; j+=stride)
        {
           
            double max_val = 0;
            for (int kh = i; kh < i+poolHeight; kh++)
            {
                for (int kw = j; kw < j+poolWidth; kw++)
                {
                    
                    if ((kh < inputHeight) && (kw < inputWidth))
                    {
                        // std::cout << "curr x: " << kh << "  curry y: " << kw << "    :    " << feature.at(kh).at(kw) << std::endl;
                        max_val = feature.at(kh).at(kw) > max_val ? feature.at(kh).at(kw) : max_val;
                    }
                }
            }

            std::cout << max_val << std::endl;
            pool_row.push_back(max_val);

        }
        pooled.push_back(pool_row);
    }
   


    this->pool_maps.push_back(pooled);
}







std::vector<gridEntity> ConvolutionLayers::getFeatureMaps()
{
    return this->feature_maps;
}

std::vector<gridEntity> ConvolutionLayers::getPoolMaps()
{
    return this->pool_maps;
}


















//for (int row = 0; row < raw_image.size(); row++)
//{
//    for (int col = 0; col < raw_image[row].size(); col + stride)
//    {
//        // Covolute the 2 things
//
//        // Get the part of the origianl image to convolute:
//        // Fuck off aakrist, i know there is a better way to do this, i just like it this way
//        gridEntity imageSection;
//        for (int i = row; i < filter_size; i++)
//        {
//            std::vector<double> temp;
//            for (int j = col; j < filter_size; j++)
//            {
//                temp.push_back(this->raw_image.at(i).at(j));
//            }
//            imageSection.push_back(temp);
//        }
//
//    }
//}
