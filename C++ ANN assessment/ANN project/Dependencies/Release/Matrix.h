#pragma once
#include <vector>
#include <iostream>
using namespace std;

class Matrix
{
	typedef vector<double> Column;
public:
	Matrix(unsigned int rows, unsigned int columns);
	Matrix();
	~Matrix();
	int GetSize();
	vector<double>& operator [] (unsigned int index);
	Matrix operator * (Matrix &m);
	void operator = (Matrix &m);
	vector<Column> data;

	friend std::ostream& operator << (std::ostream& stream, const Matrix& m)
	{
		for (int r = 0; r < m.data.size(); r++)
		{
			for (int c = 0; c < m.data[r].size(); c++)
			{
				stream << m.data[r][c] << " ";
			}

			stream << endl;
		}

		return stream;
	}

private:
};