#include "TrainingData.h"

bool TrainingData::isEof()
{
	return trainingDataFile.eof();
}

TrainingData::TrainingData(const std::string filename)
{
	trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getTargetOutputs(std::vector<double>& targetOutputVals)
{
	targetOutputVals.clear();

	std::string line;
	getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
		{
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}

unsigned TrainingData::getNextInputs(std::vector<double>& inputVals)
{
	inputVals.clear();

	std::string line;
	getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
		{
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

void TrainingData::getTopology(std::vector<unsigned>& topology)
{
	std::string line;
	std::string label;

	std::getline(trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
}