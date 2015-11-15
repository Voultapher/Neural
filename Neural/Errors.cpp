#include <iostream>
#include <cstdlib>

#include "Errors.h"

void fatalError(std::string errorString){

	printf("\n%s\n", errorString.c_str());
	printf("\nEnter any key to exit...\n");
	char tmp = std::cin.get();
	exit(42);

}
