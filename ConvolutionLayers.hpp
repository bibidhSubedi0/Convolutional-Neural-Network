#pragma once
#include "all_includes.hpp"


class ConvolutionLayers
{
	// The raw image we get from input layer
	gridEntity raw_image;

	// All the filters we need to apply to the main image
	// I can extract the no of filters, filter dimentions from this data
	// Do note that, the filers are 2d filters as grid entity is a (2D) Matrix
	std::vector<gridEntity> predefined_filters;

	// The results after appying the filters to the main image
	// no. of filters = no. of feature_map
	std::vector<gridEntity> feature_maps;


	// Pool maps or smth, idk what they are called
	std::vector<gridEntity> pool_maps;


	// Define Filters for 1'st Convolutoin layer [Deterministic] -> Still there are better places to put filter
	gridEntity STRONG_VERTICAL_EDGE_DETECTION = {
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 } };
	gridEntity STRONG_HORIZONTAL_EDGE_DETECTION = {
		{1, 1, 1, 1, 1},
		{0, 0, 0, 0, 0},
		{-1, -1, -1, -1, -1},
		{0, 0, 0, 0, 0},
		{1, 1, 1, 1, 1} };
	gridEntity STRONG_DIAGONAL_EDGE_DETECTION = {
		{1, 1, 0, -1, -1},
		{1, 1, 0, -1, -1},
		{0, 0, 0, 0, 0},
		{-1, -1, 0, 1, 1},
		{-1, -1, 0, 1, 1} };

	// Define Filters for 2'nd Convolution layer [Non-Determinstic/The shit to train]
	std::vector<gridEntity> trained_filters;
	// Assume i need to apply 4 filters( 3 x 3 x P) , P because i applied P filters in the last convolution layer to get P feature maps so it became P chanal input for my new convolution layer
	volumetricEntity input_channels;
	volumetricEntity training_filters;
	volumetricEntity ouput_features;



	// Informations
	int no_of_filters_used; // = no_of_channels

public:
	ConvolutionLayers(gridEntity);


	// std::vector<gridEntity> getFeatureMaps();
	// std::vector<gridEntity> getPoolMaps();

	// Returns all the filters used int the first convolution layer
	std::vector<gridEntity> get_all_predefined_filter();

	// Returns the raw image in form fo gridEntity
	gridEntity get_raw_input_image();

	// Return by reference as we may need to Insert into feature map
	std::vector<gridEntity>& get_feature_map();

	// Return by reference as we may need to Insert into pool map
	std::vector<gridEntity>& get_pool_map();

	// Takes image to apply filter to, the filter to apply and stride and returns the feature_map
	gridEntity apply_filter_universal(gridEntity, gridEntity, int);

	// Takes the reference to the feature map and activates it
	void activate_feature_map_using_RELU_universal(gridEntity&);

	// Takes the reference to the feature map and applies normaization to the feature map
	void apply_normalaization_universal(gridEntity&);

	// Takes a feature map and returns the pool map
	gridEntity apply_pooling_univeral(gridEntity, int);


};

