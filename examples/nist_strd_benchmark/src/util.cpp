
#include "util.h"

bool Utility::FileExists(const std::string &filename)
{
	std::ifstream file(filename);
	return (!file.fail());
}

std::vector<std::string> Utility::GetFileLines(const std::string &filename, unsigned int minLineLength)
{
	if (!Utility::FileExists(filename))
	{
		std::cout << "Required file not found: " << filename << '\n';
		exit(1);
	}
	std::ifstream file(filename);
	std::vector<std::string> result;
	std::string curLine;
	while (!file.fail())
	{
		std::getline(file, curLine);
		if (!file.fail() && curLine.length() >= minLineLength)
		{
			if (curLine.at(curLine.length() - 1) == '\r')
				curLine = curLine.substr(0, curLine.size() - 1);
			result.push_back(curLine);
		}
	}
	return result;
}

std::vector<std::string> Utility::PartitionString(const std::string &s, const std::string &separator)
{
	std::vector<std::string> result;
	std::string curEntry;
	for (unsigned int outerCharacterIndex = 0; outerCharacterIndex < s.length(); outerCharacterIndex++)
	{
		bool isSeperator = true;
		for (unsigned int innerCharacterIndex = 0; innerCharacterIndex < separator.length() && outerCharacterIndex + innerCharacterIndex < s.length() && isSeperator; innerCharacterIndex++)
		{
			if (s[outerCharacterIndex + innerCharacterIndex] != separator[innerCharacterIndex])
			{
				isSeperator = false;
			}
		}

		if (isSeperator)
		{
			if (curEntry.length() > 0)
			{
				result.push_back(curEntry);
				curEntry.clear();
			}
			outerCharacterIndex += (int)separator.length() - 1;
		}
		else
		{
			curEntry.push_back(s[outerCharacterIndex]);
		}
	}
	if (curEntry.length() > 0)
	{
		result.push_back(curEntry);
	}
	return result;
}

int Utility::StringToInt(const std::string &s)
{
	std::stringstream stream(std::stringstream::in | std::stringstream::out);
	stream << s;

	int result;
	stream >> result;
	return result;
}

float Utility::StringToFloat(const std::string &s)
{
	std::stringstream stream(std::stringstream::in | std::stringstream::out);
	stream << s;

	float result;
	stream >> result;
	return result;
}

std::vector<int> Utility::StringToIntegerList(const std::string &s, const std::string &prefix)
{
	std::string subString = Utility::PartitionString(s, prefix)[0];
	auto parts = Utility::PartitionString(subString, " ");

	std::vector<int> result(parts.size());
	for (unsigned int resultIndex = 0; resultIndex < result.size(); resultIndex++)
	{
		result[resultIndex] = Utility::StringToInt(parts[resultIndex]);
	}
	return result;
}

std::vector<float> Utility::StringToFloatList(const std::string &s, const std::string &prefix)
{
	std::string subString = Utility::PartitionString(s, prefix)[0];
	auto parts = Utility::PartitionString(subString, " ");

	std::vector<float> result(parts.size());
	for (unsigned int resultIndex = 0; resultIndex < result.size(); resultIndex++)
	{
		result[resultIndex] = Utility::StringToFloat(parts[resultIndex]);
	}
	return result;
}
