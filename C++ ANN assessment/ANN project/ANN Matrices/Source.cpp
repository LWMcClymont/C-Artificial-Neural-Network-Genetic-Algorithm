#include <iostream>
using namespace std;

#include "ArtificialNeuralNetwork.h"

const int RoundI(double x)
{
	if (x >= 0.5)
	{
		return 1;
	}

	return 0;
}

void main()
{
	vector<Data> dataSet;

	dataSet.push_back(Data(Matrix(4,2), Matrix(4, 1)));
	dataSet.back().input[0][0] = 0;
	dataSet.back().input[0][1] = 0;
	dataSet.back().expectedOutput[0][0] = 0;

	dataSet.back().input[1][0] = 1;
	dataSet.back().input[1][1] = 1;
	dataSet.back().expectedOutput[1][0] = 0;

	dataSet.back().input[2][0] = 1;
	dataSet.back().input[2][1] = 0;
	dataSet.back().expectedOutput[2][0] = 1;

	dataSet.back().input[3][0] = 0;
	dataSet.back().input[3][1] = 1;
	dataSet.back().expectedOutput[3][0] = 1;

	dataSet.push_back(Data(Matrix(8, 3), Matrix(8, 3)));
	Matrix & input = dataSet.back().input;
	input[0][0] = 0;
	input[0][1] = 0;
	input[0][2] = 0;

	input[1][0] = 0;
	input[1][1] = 0;
	input[1][2] = 1;

	input[2][0] = 0;
	input[2][1] = 1;
	input[2][2] = 0;

	input[3][0] = 0;
	input[3][1] = 1;
	input[3][2] = 1;

	input[4][0] = 1;
	input[4][1] = 0;
	input[4][2] = 0;

	input[5][0] = 1;
	input[5][1] = 0;
	input[5][2] = 1;

	input[6][0] = 1;
	input[6][1] = 1;
	input[6][2] = 0;

	input[7][0] = 1;
	input[7][1] = 1;
	input[7][2] = 1;

	Matrix & output = dataSet.back().expectedOutput;

	output[0][0] = 0;
	output[0][1] = 0;
	output[0][2] = 1;

	output[1][0] = 0;
	output[1][1] = 1;
	output[1][2] = 0;

	output[2][0] = 0;
	output[2][1] = 1;
	output[2][2] = 1;

	output[3][0] = 1;
	output[3][1] = 0;
	output[3][2] = 0;

	output[4][0] = 1;
	output[4][1] = 0;
	output[4][2] = 1;

	output[5][0] = 1;
	output[5][1] = 1;
	output[5][2] = 0;

	output[6][0] = 1;
	output[6][1] = 1;
	output[6][2] = 1;

	output[7][0] = 0;
	output[7][1] = 0;
	output[7][2] = 0;

	vector<unsigned int> structure;

	structure.push_back(3); // inputs
	structure.push_back(3); // hidden Layer
	structure.push_back(3); // Extra hidden layer
	structure.push_back(3); // outputs

	GANeuralNetwork nn = GANeuralNetwork(structure, 200, 1000000);
	
	nn.Train(dataSet[1]);

	NeuralNetwork n = nn.GetFittestNN().m_net;
	n.FeedForward(dataSet[1].input);
	
	cout << "Input: " << endl << dataSet[1].input << endl;
	cout << "Output: " << endl;

	Matrix & result = n.GetResult();

	for (int r = 0; r < result.data.size(); r++)
	{
		for (int c = 0; c < result.data[0].size() - 1; c++)
		{
			cout << RoundI(result[r][c]) << " ";
		}
		cout << endl;
	}

	system("pause");
}