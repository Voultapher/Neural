#include <cstdio>
#include <iostream>
#include <vector>
#include <string>

#include "Net.h"
#include "TrainingData.h"

#define MAX_ITERATION 1e4
#define MAX_ERROR 1e-5
#define UPPER_BOUND 10.0

void printResult(std::string label, std::vector<double> data, double scalingFactor = 1.0);
std::vector<double> getUserInput(int targetOutPutsize);

int main(){
	std::vector<unsigned int> topology{ 2, 16, 4, 1 }; // create the Neural Net topology

	std::string filePath = "inputData.txt";
	std::vector<DataSet> dataSets;
	std::vector<double> resultVals;

	TrainingData traningData;
	if (traningData.init(filePath)){
		Net myNet(topology, UPPER_BOUND);
		traningData.max(UPPER_BOUND); // only do this if the data relation is linear
		dataSets = traningData.getDataSets();

		for (std::size_t n = 0; n < MAX_ITERATION; ++n){
			for (auto& dataSet : dataSets){
				myNet.feedForward(dataSet.inputValues);
				myNet.backPropagate(dataSet.targetValues);
			}
		}

		for (auto& dataSet : dataSets){ // print all at the end
			myNet.feedForward(dataSet.inputValues);
			printResult("Input", dataSet.inputValues);
			printResult("Target", dataSet.targetValues);
			myNet.getResults(resultVals);
			printResult("Output", resultVals, myNet.getScalingFactor());
			myNet.backPropagate(dataSet.targetValues);
			printf("\n");
		}

		std::vector<double> testInput = {-1.0, 3.0 };
		myNet.feedForward(testInput);
		myNet.getResults(resultVals);
		printResult("Output", resultVals, myNet.getScalingFactor());
	}

	printf("Press ENTER to close.\n");
	char tmp = std::cin.get();

	return 0;
}

void printResult(std::string label, std::vector<double> data, double scalingFactor){
	for (auto value : data){
		printf(" %s: %.5f ", label.c_str(), value * scalingFactor);
	}
	printf("\n");
}