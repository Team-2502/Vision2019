#include "opencv2/opencv.hpp"
#include <iostream>
#include <ctime>

int main() {
	cv::VideoCapture cap(4);

	while(1) {
		cv::Mat frame;
		std::time_t result = std::time(nullptr);
		cap >> frame;
		std::cout << 1/(std::time(nullptr) - result) << "\n";

	}
	return 1;
}
