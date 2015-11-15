#pragma once

#include <string>
#include <vector>

struct DataSet{
	std::vector<double> inputValues;
	std::vector<double> targetValues;
};

class TrainingData
{
public:
	TrainingData();
	~TrainingData();

	bool init(std::string filePath);
	void max(double netBound); // will only work if the trainingsData has a linear ralation

	double getUpperBound() const { return _upperBound; }
	std::vector<DataSet> getDataSets() const { return _dataSets; }

private:
	void analyseFileData();
	void normalizeData();

	double _upperBound;
	std::string _filePath;
	std::vector<unsigned char> _fileData;
	std::vector<DataSet> _dataSets;
};

