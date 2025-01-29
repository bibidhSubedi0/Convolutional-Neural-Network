#pragma once


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<string>
#include <vector>
#include <stdexcept>

using gridEntity = std::vector<std::vector<double>>;
using volumetricEntity = std::vector<std::vector<std::vector<double>>>;



namespace Filters{

	static gridEntity STRONG_VERTICAL_EDGE_DETECTION = {
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 } };
	static gridEntity STRONG_HORIZONTAL_EDGE_DETECTION = {
		{1, 1, 1, 1, 1},
		{0, 0, 0, 0, 0},
		{-1, -1, -1, -1, -1},
		{0, 0, 0, 0, 0},
		{1, 1, 1, 1, 1} };
	static gridEntity STRONG_DIAGONAL_EDGE_DETECTION = {
		{1, 1, 0, -1, -1},
		{1, 1, 0, -1, -1},
		{0, 0, 0, 0, 0},
		{-1, -1, 0, 1, 1},
		{-1, -1, 0, 1, 1} };
}


