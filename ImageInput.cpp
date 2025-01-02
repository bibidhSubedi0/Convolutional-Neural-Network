#include "ImageInput.hpp"
#include "all_includes.hpp"



ImageInput ::ImageInput(std::string img_name, int mode){
    this->image = cv::imread(img_name, mode);
    this->matrixifyPixelValues();
}
ImageInput::ImageInput(){}

void ImageInput::setImage(std::string img_name, int mode)
{
    this->image = cv::imread(img_name, mode);
    this->matrixifyPixelValues();
}


void ImageInput::showImage()
{
    cv::imshow("Image", this->image);
    cv::waitKey(0);
}

void ImageInput::matrixifyPixelValues()
{
    for (int i = 0; i < this->image.rows; ++i) {
        std::vector<int> temp;
        for (int j = 0; j < this->image.cols; ++j) {
            temp.push_back((int)this->image.at<uchar>(i, j));
        }
        this->pixel_values.push_back(temp);
    }
}

std::vector<std::vector<int>> ImageInput::getMatrixifiedPixelValues()
{
    return this->pixel_values;
}


