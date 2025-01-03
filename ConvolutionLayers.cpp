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

    // Purpose: Detect circular or blob-like features
    gridEntity filter4 = {
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {1, 1, 1, 1, 1},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0}  
    };

    // Purpose: Detect star-like or cross patterns
    gridEntity filter5 = {
        {1, 0, 0, 0, 1},
        {0, 1, 0, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 0, 1, 0},
        {1, 0, 0, 0, 1}  
    };

    this->filters.push_back(filter1);
    this->filters.push_back(filter2);
    this->filters.push_back(filter3);
    this->filters.push_back(filter4);
    this->filters.push_back(filter5);

}

gridEntity ConvolutionLayers::get_filter(int index)
{
    return this->filters.at(index);
}


void ConvolutionLayers::apply_filter(gridEntity filt,int stride=1)
{
    // Write the convlution for filter and image and return the un-activaed feature map
    // fuck this is going to be long
    int filter_size = filt.size();
    gridEntity f_map = Matrix::convolute(raw_image,filt);
    this->feature_maps.push_back(f_map);
}

std::vector<gridEntity> ConvolutionLayers::getFeatureMaps()
{
    return this->feature_maps;
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
