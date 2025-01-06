#include "ImageInput.hpp"
#include "all_includes.hpp"



ImageInput ::ImageInput(std::string img_name, int mode){
    this->image = cv::imread(img_name, mode);
    this->matrixifyPixelValues();
}
ImageInput::ImageInput(){}

ImageInput::ImageInput(std::vector<std::vector<double>> raw_pixels)
{
    this->image = cv::Mat(raw_pixels.size(), raw_pixels[0].size(), CV_64F);

    for (size_t i = 0; i < raw_pixels.size(); ++i) {
        for (size_t j = 0; j < raw_pixels[i].size(); ++j) {
            this->image.at<double>(i, j) = raw_pixels[i][j];
        }
    }

    this->matrixifyPixelValues();

}

void ImageInput::setImage(std::string img_name, int mode)
{
    this->image = cv::imread(img_name, mode);
    this->matrixifyPixelValues();
}


void ImageInput::showImage()
{
    cv::Mat largeImage;

    // Show image by magnifiying 10 times
    resize(this->image, largeImage, cv::Size(this->image.rows * 10, this->image.cols * 10), 0, 0, cv::INTER_NEAREST);
    cv::imshow("Image", largeImage);
    // cv::imshow("Image", this->image);
    cv::waitKey(0);
}

void ImageInput::matrixifyPixelValues()
{
    for (int i = 0; i < this->image.rows; ++i) {
        std::vector<double> temp;
        for (int j = 0; j < this->image.cols; ++j) {
            temp.push_back((int)this->image.at<uchar>(i, j));
        }
        this->pixel_values.push_back(temp);
    }
}

std::vector<std::vector<double>> ImageInput::getMatrixifiedPixelValues()
{
    return this->pixel_values;
}


