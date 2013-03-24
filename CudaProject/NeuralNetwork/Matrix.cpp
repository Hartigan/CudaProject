#include "Matrix.h"

Matrix::Matrix(int dimX, int dimY)
{
	_values = new float[dimX * dimY];
	_dimX = dimX;
	_dimY = dimY;
}

Matrix::~Matrix()
{
	delete[] _values;
}

float* Matrix::getArrayPointer()
{
	return _values;
}

float Matrix::getValue(int x, int y)
{
	return _values[x * _dimX + y];
}

void Matrix::setValue(int x, int y, float value)
{
	_values[x * _dimX + y] = value;
}

int Matrix::getDimX()
{
	return _dimX;
}

int Matrix::getDimY()
{
	return _dimY;
}