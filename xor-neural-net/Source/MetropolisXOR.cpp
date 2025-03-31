#include "MetropolisXOR.h"
#include <iostream>
#include <iterator>
#include <chrono>
#include <math.h>

MetropolisXOR::MetropolisXOR(const int* TruthArr, float* InitWeights) :
	TruthTable(TruthArr)
{
	for (size_t i = 0; i < 16; i++)
	{
		*(CurrentWeights + i) = InitWeights[i];
	}
}

MetropolisXOR::~MetropolisXOR()
{
}

void MetropolisXOR::Run()
{
	std::cout << "Running Metropolis Algorithm:" << std::endl;
	auto StopwatchStart = std::chrono::high_resolution_clock::now();
	srand(time(NULL));

	const int NMax = 1000000;
	for (size_t i = 0; i < NMax; i++) {
		// inverse temperature increases over time, simulating the system cooling
		Beta = Beta + 1000.f / NMax;

		// copy the current weights to the new weights array
		std::copy(std::begin(CurrentWeights), std::end(CurrentWeights), std::begin(NewWeights));

		// pick a random weight in the array and set it to a new value between -10 and 10
		const int RandIndex = (float(rand()) / (float(RAND_MAX))) * 16;
		NewWeights[RandIndex] = (float(rand()) / (float(RAND_MAX)) - 0.5) * 20;

		// get the global errors for each of the weight configurations
		const float CurrentError = GetGobalError(CurrentWeights);
		const float NewError = GetGobalError(NewWeights);

		//keep the configuration if it reduces error
		if (NewError - CurrentError < 0.f)
		{
			std::copy(std::begin(NewWeights), std::end(NewWeights), std::begin(CurrentWeights));
		}
		else
		{
			// take the new config even if it doesnt reduce error if the probability of a random jump is high enough
			const float JumpProbability = exp((CurrentError - NewError) * Beta);
			const float Rand = float(rand()) / (float(RAND_MAX));
			if (Rand < JumpProbability)
			{
				std::copy(std::begin(NewWeights), std::end(NewWeights), std::begin(CurrentWeights));
			}
		}
	}
	auto StopwatchEnd = std::chrono::high_resolution_clock::now();
	auto TimeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(StopwatchEnd - StopwatchStart);
	std::cout << "Completed Metropolis Algorithm in " << TimeElapsed.count() << " ms" << std::endl;
}

void MetropolisXOR::LogResults()
{
	for (size_t i = 0; i < 8; i++)
	{
		const float H4 = SigmoidThreshold(CurrentWeights[0] + CurrentWeights[4] * TruthTable[i * 4] + CurrentWeights[5] * TruthTable[i * 4 + 1] + CurrentWeights[6] * TruthTable[i * 4 + 2]);
		const float H5 = SigmoidThreshold(CurrentWeights[1] + CurrentWeights[7] * TruthTable[i * 4] + CurrentWeights[8] * TruthTable[i * 4 + 1] + CurrentWeights[9] * TruthTable[i * 4 + 2]);
		const float H6 = SigmoidThreshold(CurrentWeights[2] + CurrentWeights[10] * TruthTable[i * 4] + CurrentWeights[11] * TruthTable[i * 4 + 1] + CurrentWeights[12] * TruthTable[i * 4 + 2]);
		const float O7 = SigmoidThreshold(CurrentWeights[3] + CurrentWeights[13] * H4 + CurrentWeights[14] * H5 + CurrentWeights[15] * H6);
		std::cout << "xor(" << TruthTable[i * 4] << TruthTable[i * 4 + 1] << TruthTable[i * 4 + 2] << ") - expected: " << TruthTable[i * 4 + 3] << ", neural net: " << round(O7) << std::endl;
	}

}

float MetropolisXOR::GetGobalError(float* Weights)
{
	float GlobalError = 0.f;
	for (size_t i = 0; i < 8; i++)
	{
		const float H4 = SigmoidThreshold(Weights[0] + Weights[4] * TruthTable[i * 4] + Weights[5] * TruthTable[i * 4 + 1] + Weights[6] * TruthTable[i * 4 + 2]);
		const float H5 = SigmoidThreshold(Weights[1] + Weights[7] * TruthTable[i * 4] + Weights[8] * TruthTable[i * 4 + 1] + Weights[9] * TruthTable[i * 4 + 2]);
		const float H6 = SigmoidThreshold(Weights[2] + Weights[10] * TruthTable[i * 4] + Weights[11] * TruthTable[i * 4 + 1] + Weights[12] * TruthTable[i * 4 + 2]);
		const float O7 = SigmoidThreshold(Weights[3] + Weights[13] * H4 + Weights[14] * H5 + Weights[15] * H6);
		GlobalError += (TruthTable[i * 4 + 3] - O7) * (TruthTable[i * 4 + 3] - O7);
	}

	return GlobalError;
}

float MetropolisXOR::SigmoidThreshold(float Input)
{
	return 1.f / (1.f + exp(-2.f * Input));
}
