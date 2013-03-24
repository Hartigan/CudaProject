#ifndef MATRIX
#define MATRIX

class Matrix
{
	int _dimX, _dimY;
	float* _values;
public:
	Matrix(int dimX, int dimY);
	~Matrix();
	int getDimX();
	int getDimY();
	float* getArrayPointer();
	float getValue(int x, int y);
	void setValue(int x, int y, float value);
};

#endif