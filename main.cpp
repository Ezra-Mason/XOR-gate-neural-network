// xor-neural-net.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include "MetropolisXOR.h"


int main()
{
	const int TruthTable[32] = { 0,0,0,0, 0,0,1,1, 0,1,0,1, 0,1,1,0, 1,0,0,1, 1,0,1,0, 1,1,0,0, 1,1,1,1 };

	// generate the random starting weights between -0.5 and +0.5
	srand(time(0));
	float InitialWeights[16] = { 0 };
	for (int i = 0; i < 16; i++) {
		*(InitialWeights + i) = (float)rand() / RAND_MAX - 0.5;
		std::cout << "init weight " << i << " = " << InitialWeights[i] << std::endl;
	}

	MetropolisXOR MXor = MetropolisXOR(TruthTable, InitialWeights);
	MXor.Run();
	MXor.LogResults();

}
