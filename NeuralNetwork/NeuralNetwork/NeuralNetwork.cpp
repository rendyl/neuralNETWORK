#include <iostream>
#include <vector>

#include "TrainingData.h"
#include "Network.h"

void createTrainingData()
{
	std::ofstream myfile;
	myfile.open("data.txt");
	myfile << "topology: 2 4 1\n";
	for (int i = 2000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t = n1 ^ n2;
		myfile << "in: " << n1 << ".0 " << n2 << ".0 \n";
		myfile << "out: " << t << ".0 \n";
	} 
	myfile.close();
}

void showVectorVals(std::string str, std::vector<double> vec)
{
	std::cout << str << " ";

	for (unsigned i = 0; i < vec.size(); ++i)
	{
		std::cout << vec[i] << " ";
	}

	std::cout << std::endl;
}

int main()
{
	createTrainingData();
	TrainingData trainData("data.txt");

	// 3 entrées 2 neurones 1 sortie
	std::vector<unsigned> topology;
	trainData.getTopology(topology);

	Network myNetwork(topology);
	
	int kaka;
	std::cin >> kaka;

	std::vector<double> inputValues;
	std::vector<double> targetValues;
	std::vector<double> resultValues;

	int trainingPass = 0;
	std::cout << "Starting Training" << std::endl;

	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << std::endl << "Pass " << trainingPass;

		if (trainData.getNextInputs(inputValues) != topology[0]) break;

		showVectorVals(": Inputs:", inputValues);
		myNetwork.feedForward(inputValues);

		myNetwork.getResults(resultValues);
		showVectorVals("Outputs:", resultValues);

		trainData.getTargetOutputs(targetValues);
		showVectorVals("Targets:", targetValues);

		assert(targetValues.size() == topology.back());

		myNetwork.backProp(targetValues);

		std::cout << "Net recent avg error: " << myNetwork.getRecentAverageError() << std::endl;
	}
	
	std::cout << std::endl << "Training Done" << std::endl;
	
	std::cout << std::endl << "TEST XOR : 0 0 " << std::endl;
	myNetwork.feedForward({0.0, 0.0});
	myNetwork.getResults(resultValues);
	std::cout << abs(round(resultValues[0])) << std::endl;

	std::cout << std::endl << "TEST XOR : 0 1 " << std::endl;
	myNetwork.feedForward({0.0, 1.0});
	myNetwork.getResults(resultValues);
	std::cout << abs(round(resultValues[0])) << std::endl;

	std::cout << std::endl << "TEST XOR : 1 0 " << std::endl;
	myNetwork.feedForward({1.0, 0.0});
	myNetwork.getResults(resultValues);
	std::cout << abs(round(resultValues[0])) << std::endl;

	std::cout << std::endl << "TEST XOR : 1 1 " << std::endl;
	myNetwork.feedForward({1.0, 1.0});
	myNetwork.getResults(resultValues);
	std::cout << abs(round(resultValues[0])) << std::endl;
}
