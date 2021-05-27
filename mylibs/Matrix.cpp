#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <assert.h>

template <class mtype>
class Matrix{	
protected:
	std::vector<mtype> matrix;
	int rows;
	int cols;
	bool t;
public:
	Matrix();
	Matrix(int, int);
	Matrix(const Matrix&);
	Matrix(std::vector<mtype>);
	Matrix(int, int, std::vector<mtype>);
	void operator =(const Matrix&);
	int getrows() const;
	int getcols() const;
	int size() const;
	mtype &operator()(int, int);
	mtype operator()(int, int) const;
	mtype &operator()(int);
	mtype operator()(int) const;
	Matrix operator* (const Matrix&);
	Matrix tmult (const Matrix&);
	Matrix operator* (mtype);
	Matrix operator+ (const Matrix&);
	Matrix operator- (const Matrix&);
	Matrix operator- ();
	void operator+=(const Matrix&);
	void operator-=(const Matrix&);
	Matrix operator! ();
	void transpose();
	void clear();
	void resize(int, int);
	void print();
	virtual ~Matrix(){};
};

template <class mtype>
Matrix<mtype>::Matrix()
: rows(0), cols(0), t(0){}

template <class mtype>
Matrix<mtype>::Matrix(int rows, int cols) 
: rows(rows), cols(cols), t(0){	matrix.resize(rows * cols); }
template <class mtype>
Matrix<mtype>::Matrix(const Matrix<mtype>& mtx) 
: matrix(mtx.matrix), rows(mtx.rows), cols(mtx.cols), t(mtx.t){}

template <class mtype>
Matrix<mtype>::Matrix(std::vector<mtype> mtx)
:  matrix(mtx), rows(mtx.size()), cols(1), t(0){}

template <class mtype>
Matrix<mtype>::Matrix(int rows, int cols, std::vector<mtype> mtx)
:  matrix(mtx), rows(rows), cols(cols), t(0){}

template <class mtype>
void Matrix<mtype>::operator=(const Matrix<mtype>& mtx){
	matrix = mtx.matrix;
	rows = mtx.rows;
	cols = mtx.cols;
	t = mtx.t;
}

template <class mtype>
int Matrix<mtype>::getrows() const{ return t ? cols : rows; }

template <class mtype>
int Matrix<mtype>::getcols() const{ return t ? rows : cols; }

template <class mtype>
int Matrix<mtype>::size() const{ return rows*cols; }

template <class mtype>
mtype &Matrix<mtype>::operator()(int y, int x) {
	return t ? matrix[cols*x+y] : matrix[cols*y+x]; 
}
template <class mtype>
mtype Matrix<mtype>::operator()(int y, int x) const {
	return t ? matrix[cols*x+y] : matrix[cols*y+x]; 
}
template <class mtype>
mtype &Matrix<mtype>::operator()(int x) {
	return t ? matrix[cols*(x%rows) + x/rows] : matrix[x]; 
}

template <class mtype>
mtype Matrix<mtype>::operator()(int x) const {
	return t ? matrix[cols*(x%rows) + x/rows] : matrix[x];
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::operator*(const Matrix<mtype> &mtx){
	assert(this->getcols() == mtx.getrows());
	Matrix<mtype> tmp(this->getrows(), mtx.getcols());
	for(int i = 0; i < this->getrows(); i++){
		for(int j = 0; j < mtx.getcols(); j++){
			for(int k = 0; k < this->getcols(); k++){
				tmp(i, j) += (*this)(i, k) * mtx(k, j); 
			}
		}
	}
	return tmp;
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::tmult(const Matrix<mtype> &mtx){
	t = !t;
	Matrix<mtype> tmp = (*this) * mtx;
	t = !t;
	return tmp;
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::operator*(mtype scalar){
	Matrix<mtype> tmp = *this;
	for(int i = 0; i < rows*cols; i++){
		tmp(i) *= scalar;
	}
	return tmp;
}

template <class mtype, class stype>
Matrix<mtype> operator*(stype scalar, Matrix<mtype> mtx){
	return mtx * (mtype) (scalar);	
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::operator+(const Matrix<mtype> &mtx){
	assert(this->getrows() == mtx.getrows() && this->getcols() == mtx.getcols());
	Matrix<mtype> tmp(rows, cols);
	for(int i = 0; i < rows*cols; i++){
		tmp(i) = (*this)(i) + mtx(i);
	}
	return tmp;
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::operator- (const Matrix<mtype> &mtx){
	return (*this) + (-1*mtx);
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::operator- (){
	return (*this) * -1;
}

template <class mtype>
void Matrix<mtype>::operator+=(const Matrix<mtype> &mtx){
	(*this) = (*this) + mtx; 
}

template <class mtype>
void Matrix<mtype>::operator-=(const Matrix<mtype> &mtx){
	(*this) = (*this) - mtx; 
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::operator! (){
	t = !t;
	Matrix<mtype> tmp(*this);
	t = !t;
	return tmp;
}

template <class mtype>
void Matrix<mtype>::transpose(){ t = !t; }

template <class mtype>
void Matrix<mtype>::clear(){
	for(int i = 0; i < this->size(); i++)
		(*this)(i) = 0;
}

template <class mtype>
void Matrix<mtype>::print(){
	for(int i = 0; i < cols*rows;){
		std::cout << (*this)(i) << " "; i++;
		if(i%this->getcols() == 0) std::cout << std::endl; 
	}
	std::cout << std::endl;
}

#endif