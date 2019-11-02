#pragma once
#include <vector>
#include <fstream>
#include <sstream>

class TrainingData
{
	public :

		TrainingData(const std::string filename);
		bool isEof();
		void getTopology(std::vector<unsigned>& topology);

		unsigned getNextInputs(std::vector<double>& inputVals);
		unsigned getTargetOutputs(std::vector<double>& targetOutputVals);

	private :

		std::ifstream trainingDataFile;
};