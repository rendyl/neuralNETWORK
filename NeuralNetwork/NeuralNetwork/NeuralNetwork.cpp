#include <iostream>
#include <vector>

#include "TrainingData.h"
#include "Network.h"

void createTrainingDataXOR()
{
	std::ofstream myfile;
	myfile.open("Data/Training/testXOR.txt");
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

void createTrainingDataTEXT()
{
	std::ofstream myfile;
	myfile.open("Data/Training/testLG.txt");
	myfile << "topology: 26 4 5\n";

	for (int i = 0; i < 3000; i++)
	{
		// Anglais
		myfile << "in: 0.08167 0.01492 0.02782 0.04253 0.12702 0.02228 0.02015 0.06094 0.06966 0.00153 0.00772 0.04025 0.02406 0.06749 0.07507 0.01929 0.00095 0.05987 0.06327 0.09056 0.02758 0.00978 0.0236 0.0015 0.01974 0.00074\n";
		myfile << "out: 1 0 0 0 0\n";

		// Francais
		myfile << "in: 0.07636 0.00901 0.0326 0.03669 0.14715 0.01066 0.00866 0.00737 0.07529 0.00613 0.00074 0.05456 0.02968 0.07095 0.05796 0.02521 0.01362 0.06693 0.07948 0.07244 0.06311 0.01838 0.00049 0.00427 0.00128 0.00326\n";
		myfile << "out: 0 1 0 0 0\n";

		// Espagnol
		myfile << "in: 0.11525 0.02215 0.04019 0.0501 0.12181 0.00692 0.01768 0.00703 0.06247 0.00493 0.00011 0.04967 0.03157 0.06712 0.08683 0.0251 0.00877 0.06871 0.07977 0.04632 0.02927 0.01138 0.00017 0.00215 0.01008 0.00467\n";
		myfile << "out: 0 0 1 0 0\n";

		// Italien
		myfile << "in: 0.11745 0.00927 0.04501 0.03736 0.11792 0.01153 0.01644 0.00636 0.10143 0.00011 0.00009 0.0651 0.02512 0.06883 0.09832 0.03056 0.00505 0.06367 0.04981 0.05623 0.03011 0.02097 0.00033 0.00003 0.0002 0.01181\n";
		myfile << "out: 0 0 0 1 0\n";

		// Neerlandais
		myfile << "in: 0.07486 0.01584 0.01242 0.05933 0.1891 0.00805 0.03403 0.0238 0.06499 0.0146 0.02248 0.03568 0.02213 0.10032 0.06063 0.0157 0.00009 0.06411 0.0373 0.0679 0.0199 0.0285 0.0152 0.00036 0.00035 0.0139\n";
		myfile << "out: 0 0 0 0 1\n";
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

std::vector<double> getFrequencyFromText(std::string filename)
{
	std::fstream inFile;
	char oneChar;
	char ASCII[128] = { 0 }; //127 characters of the ASCII +1 
	int frequency[128];
	char code;

	double cptLetter = 0.0;

	for (int ASCII = 0; ASCII < 128; ASCII++) //Loop through all the ASCII characters to get the frequency of each
	{
		frequency[ASCII] = 0;
	}

	inFile.open(filename, std::ios::in); //Reads file

	if (inFile.fail()) //Check for error opening the file
	{
		std::cout << "Error: File not found!" << std::endl << std::endl;
	}
	else
	{
		oneChar = inFile.get();

		while (inFile.eof() == false) //End of file
		{
			if (oneChar != ' ')
			{
				if ((oneChar >= 'a' && oneChar <= 'z') || (oneChar >= 'A' && oneChar <= 'Z'))
				{
					frequency[oneChar]++; //When not empty character
					cptLetter++;
				}
			}
			oneChar = inFile.get();
		}
	}

	inFile.close();

	std::vector<double> letterFreq;
	for (char caps = 'A'; caps <= 'Z'; caps++) letterFreq.push_back((double)frequency[caps]);
	int i = 0;
	for (char lower = 'a'; lower <= 'z'; lower++)
	{
		letterFreq[i] += (double)frequency[lower];
		letterFreq[i] /= cptLetter;
		i++;
	}

	return letterFreq;
}

int main()
{
	int choix;
	int iteMax = 0;
	std::string filename = "Data/Training/testLG.txt";

	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "Choose your training : XOR or TEXT ? 0 or 1 > ";
	std::cin >> choix;

	if (choix == 0)
	{
		std::cout << "You chose : XOR" << std::endl;
		createTrainingDataXOR();
		filename = "Data/Training/testXOR.txt";
		iteMax = 2000;
	}

	else
	{
		std::cout << "You chose : TEXT" << std::endl;
		createTrainingDataTEXT();
		filename = "Data/Training/testLG.txt";
		iteMax = 15000;
	}

	int nbIter;
	std::cout << "How many displays do you want ? > ";
	std::cin >> nbIter;
	iteMax = iteMax - nbIter;

	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "Creating the Network : " << std::endl;

	TrainingData trainData(filename);

	std::vector<unsigned> topology;
	trainData.getTopology(topology);

	Network myNetwork(topology);

	std::vector<double> inputValues;
	std::vector<double> targetValues;
	std::vector<double> resultValues;

	int trainingPass = 0;
	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "Starting Training : " << std::endl;

	while (!trainData.isEof())
	{
		++trainingPass;
		if (trainingPass > iteMax) std::cout << std::endl << "Pass " << trainingPass << std::endl;

		if (trainData.getNextInputs(inputValues) != topology[0]) break;

		if (trainingPass > iteMax) showVectorVals("Inputs:", inputValues);
		myNetwork.feedForward(inputValues);

		myNetwork.getResults(resultValues);
		if (trainingPass > iteMax) showVectorVals("Outputs:", resultValues);

		trainData.getTargetOutputs(targetValues);
		if (trainingPass > iteMax) showVectorVals("Targets:", targetValues);

		assert(targetValues.size() == topology.back());

		myNetwork.backProp(targetValues);

		if (trainingPass > iteMax) std::cout << "Net recent avg error: " << myNetwork.getRecentAverageError() << std::endl;
	}
	
	std::cout << "Training Done" << std::endl;
	
	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "Testing Time : ";


	// TEST XOR
	if (choix == 0)
	{
		std::cout << "XOR" << std::endl << std::endl;
		std::cout << "Inputs: 0 0 " << std::endl;
		myNetwork.feedForward({ 0.0, 0.0 });
		myNetwork.getResults(resultValues);
		std::cout << "Outputs: " << abs(round(resultValues[0])) << std::endl;

		std::cout << std::endl << "Inputs: 0 1 " << std::endl;
		myNetwork.feedForward({ 0.0, 1.0 });
		myNetwork.getResults(resultValues);
		std::cout << "Outputs: " << abs(round(resultValues[0])) << std::endl;

		std::cout << std::endl << "Inputs: 1 0 " << std::endl;
		myNetwork.feedForward({ 1.0, 0.0 });
		myNetwork.getResults(resultValues);
		std::cout << "Outputs: " << abs(round(resultValues[0])) << std::endl;

		std::cout << std::endl << "Inputs: 1 1 " << std::endl;
		myNetwork.feedForward({ 1.0, 1.0 });
		myNetwork.getResults(resultValues);
		std::cout << "Outputs: " << abs(round(resultValues[0])) << std::endl;
	}

	// TEST TEXT
	else
	{
		std::cout << "TEXT" << std::endl << std::endl;
		std::cout << "Texte Anglais : " << std::endl;
		std::vector<double> letterFreq = getFrequencyFromText("Data/Testing/textENG.txt");
		myNetwork.feedForward(letterFreq);
		showVectorVals("Inputs:", letterFreq);
		myNetwork.getResults(resultValues);
		showVectorVals("Outputs:", resultValues);
		std::cout << "ENG FR ESP ITA NL" << std::endl << std::endl;

		std::cout << "Texte Francais : " << std::endl;
		letterFreq = getFrequencyFromText("Data/Testing/textFR.txt");
		myNetwork.feedForward(letterFreq);
		showVectorVals("Inputs:", letterFreq);
		myNetwork.getResults(resultValues);
		showVectorVals("Outputs:", resultValues);
		std::cout << "ENG FR ESP ITA NL" << std::endl;
	}

	std::cout << std::endl << "End" << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;
}
