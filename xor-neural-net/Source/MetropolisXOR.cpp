#include "MetropolisXOR.h"
#include <iostream>
#include <iterator>
#include <numeric>
#include <chrono>
#include <math.h>

MetropolisXOR::MetropolisXOR(
	const int InputCount,
	const std::vector<std::vector<int>>& InputArr,
	const int* TruthArr,
	const int HiddenCount,
	float* InitWeights)
	:
	Inputs(InputArr),
	InputNodeCount(InputCount),
	TruthTable(TruthArr),
	HiddenNodeCount(HiddenCount)
{

	for (size_t i = 0; i < 16; i++)
	{
		*(CurrentWeights + i) = InitWeights[i];
	}
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
		std::vector<float> H;
		for (size_t j = 0; j < HiddenNodeCount; j++)
		{
			const int StartIndex = HiddenNodeCount * (j + 1) + 1;
			const int EndIndex = StartIndex + HiddenNodeCount;
			H.emplace_back(SigmoidThreshold(CurrentWeights[j] + std::inner_product(CurrentWeights + StartIndex, CurrentWeights + EndIndex, Inputs[i].begin(), 0.f)));
		}
		const int OutputBeginIndex = HiddenNodeCount * (HiddenNodeCount + 1) + 1;
		const int OutputEndIndex = OutputBeginIndex + HiddenNodeCount;
		const float Output = SigmoidThreshold(CurrentWeights[HiddenNodeCount] + std::inner_product(CurrentWeights + OutputBeginIndex, CurrentWeights + OutputEndIndex, H.begin(), 0.f));

		printf("Input: (%d, %d, %d),  expected: %d, neural net: %f \n", Inputs[i][0], Inputs[i][1], Inputs[i][2], TruthTable[i * 4 + 3], Output);
	}
}

float MetropolisXOR::GetGobalError(float* Weights)
{
	float GlobalError = 0.f;
	std::vector<float> HiddenOutputs;
	for (size_t i = 0; i < 8; i++)
	{
		HiddenOutputs.clear();
		for (size_t j = 0; j < HiddenNodeCount; j++)
		{
			const int StartIndex = HiddenNodeCount * (j + 1) + 1;
			const int EndIndex = StartIndex + HiddenNodeCount;
			HiddenOutputs.emplace_back(SigmoidThreshold(Weights[j] + std::inner_product(Weights + StartIndex, Weights + EndIndex, Inputs[i].begin(), 0.f)));
		}
		const int OutputBeginIndex = HiddenNodeCount * (HiddenNodeCount + 1) + 1;
		const int OutputEndIndex = OutputBeginIndex + HiddenNodeCount;
		const float Output = SigmoidThreshold(Weights[HiddenNodeCount] + std::inner_product(Weights + OutputBeginIndex, Weights + OutputEndIndex, HiddenOutputs.begin(), 0.f));

		GlobalError += (TruthTable[i * 4 + 3] - Output) * (TruthTable[i * 4 + 3] - Output);
	}

	return GlobalError;
}

float MetropolisXOR::SigmoidThreshold(float Input)
{
	return 1.f / (1.f + exp(-2.f * Input));
}
