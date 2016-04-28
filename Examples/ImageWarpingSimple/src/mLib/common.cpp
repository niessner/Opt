#include <string>

namespace ml
{


void warningFunctionMLIB(const std::string &description)
{
	std::cout << description << std::endl;
    //DEBUG_BREAK;
}

void errorFunctionMLIB(const std::string &description)
{
	std::cout << description << std::endl;
	DEBUG_BREAK;
}

void assertFunctionMLIB(bool statement, const std::string &description)
{
	if(!statement)
	{
		std::cout << description << std::endl;
#ifdef _DEBUG
#ifdef _WIN32
		DEBUG_BREAK;
#endif
#endif
	}
}

}  // namespace ml