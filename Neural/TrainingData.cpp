#include "TrainingData.h"

#include "IOManager.h"
#include "Errors.h"


TrainingData::TrainingData()
{
}


TrainingData::~TrainingData()
{
}

bool TrainingData::init(std::string filePath){
	_filePath = filePath;
	
	if (IOManager::readFileToBuffer(filePath, _fileData)){
		analyseFileData();
		normalizeData();
		return true;
	}
	else{
		return false;
	}
}

void TrainingData::analyseFileData(){ // data sets with data from the file
	double value = 0.0;
	std::string valueString = "";
	char ident, lastIdent = 'N';
	_fileData.push_back('\r');
	_dataSets.push_back(DataSet()); // add first data set

	for (std::size_t n = 0; n < _fileData.size(); ++n){
		ident = _fileData[n] == 'I' || _fileData[n] == 'T' ? _fileData[n] == 'I' ? 'I' : 'T' : ident; // ident will mark input(I) or target(T), only change if data is I or T
		if (lastIdent == 'T' && ident == 'I'){
			_dataSets.push_back(DataSet()); // add new data set
		}

		if (_fileData[n] != ' ' && _fileData[n] != 'I' && _fileData[n] != 'T' && _fileData[n] != '\r' && _fileData[n] != '\n'){ // loop through one number
			valueString += _fileData[n];
		}
		else if (valueString.size() > 0){
			try{
				value = std::stod(valueString);
			}
			catch (...){
				fatalError("Input file not conform");
			}
			valueString = "";
			ident == 'I' ? _dataSets.back().inputValues.push_back(value) : _dataSets.back().targetValues.push_back(value);
		}
		lastIdent = ident;
	}
}

void TrainingData::normalizeData(){
	double upperBound = 0.0;
	for (auto& dataSet : _dataSets){ // for all data
		for (auto data : dataSet.inputValues){
			if (std::abs(data) > upperBound){
				upperBound = std::abs(data);
			}
		}
		for (auto data : dataSet.targetValues){
			if (std::abs(data) > upperBound){
				upperBound = std::abs(data);
			}
		}
	}
	_upperBound = upperBound;
}

void TrainingData::max(double netBound){
	double scaling = netBound / _upperBound;
	for (auto& dataSet : _dataSets){ // for all data
		for (auto& data : dataSet.inputValues){
			data *= scaling;
		}
		for (auto& data : dataSet.targetValues){
			data *= scaling;
		}
	}
}