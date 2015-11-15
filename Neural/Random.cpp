#include "Random.h"


double randomWeight(){
	static std::mt19937 generator(1337); // fixed seed for result reproducability
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);

	return distribution(generator);
}
