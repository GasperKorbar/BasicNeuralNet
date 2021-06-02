#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <assert.h>
#include <immintrin.h>
#include "timer.cpp"

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
	Matrix fastmult_vvt(const Matrix&);
	Matrix fastmult_mv(const Matrix&);
	Matrix operator* (mtype);
	Matrix operator+ (const Matrix&);
	Matrix operator- (const Matrix&);
	Matrix operator- ();
	void operator+=(const Matrix&);
	void operator-=(const Matrix&);
	Matrix pointwisemult(const Matrix&);
	Matrix pointwiseoperator(mtype (*)(mtype));
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
	if(this->size() > 6000 && mtx.getcols() == 1)return this->fastmult_mv(mtx);
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

template <class mtype>
Matrix<mtype> Matrix<mtype>::fastmult_mv(const Matrix<mtype> &mtx){
	assert(this->getcols() == mtx.getrows() && mtx.getcols() == 1);
	Matrix<mtype> tmp(this->getrows(), mtx.getcols());
	// int vecsize = 8;
	int size = this->getcols()%8 == 0 ? this->getcols()/8 : this->getcols()/8+1;
	union {__m256 a8; float a[8];}u;
	std::vector<float> extra(8);
	__m256 vec, m;
	int overhang = this->getcols()%8;
	for(int j = 0; j < (int) tmp.size(); j++){
		u.a8 = _mm256_setzero_ps();
		for(int i = 0; i < size; i++){
			if(i == size-1 && this->getcols()%8 != 0){
				for(int j = 0; j < overhang; j++) extra[j] = mtx(8*i+j);
				vec = _mm256_loadu_ps(&extra[0]);
			} else {
				vec = _mm256_loadu_ps(&mtx.matrix[8*i]);
			}
			if(i == size-1 && this->getcols()%8 != 0){
				for(int k = 0; k < overhang; k++) extra[k] = (*this)(j, 8*i+k);
				m = _mm256_loadu_ps(&extra[0]);
			} else{
				// m = _mm256_loadu_ps(&(*this)(j, 8*i));
				m = _mm256_setr_ps((*this)(j, 8*i), (*this)(j,8*i+1), (*this)(j,8*i+2), (*this)(j,8*i+3), (*this)(j,8*i+4), (*this)(j,8*i+5), (*this)(j,8*i+6), (*this)(j,8*i+7));
			}
			u.a8 = _mm256_add_ps(u.a8, _mm256_mul_ps(m, vec));
		}
		float sum = 0;
		for(int h = 0; h < 8; h++){
			sum += u.a[h];
		}
		tmp(j) += sum;
	}
	return tmp;
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::fastmult_vvt(const Matrix<mtype> &mtx){
	assert(this->getcols() == 1 && mtx.getcols() == 1);
	Matrix<mtype> tmp(this->getrows(), mtx.getrows());
	__m256 allnumvec, vec2;
	for(int j = 0; j < (int) this->size(); j++){
		allnumvec = _mm256_set1_ps((*this)(j));
		for(int i = 8; i < mtx.size(); i+=8){
			vec2 = _mm256_loadu_ps(&mtx.matrix[i-8]);
			_mm256_storeu_ps(&tmp(j, i-8), _mm256_mul_ps(allnumvec, vec2));
		}
		if(mtx.size()%8 != 0){
			int maxoffset = 8*(mtx.size()/8);
			for(int i = 0; i < mtx.size()%8; i++) tmp(j, maxoffset+i) = (*this)(j) * mtx(maxoffset+i);
		}
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
	assert(this->getrows() == mtx.getrows() && this->getcols() == mtx.getcols());
	Matrix<mtype> tmp(rows, cols);
	for(int i = 0; i < rows*cols; i++){
		tmp(i) = (*this)(i) - mtx(i);
	}
	return tmp;
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::operator- (){
	return (*this) * -1;
}

template <class mtype>
void Matrix<mtype>::operator+=(const Matrix<mtype> &mtx){
	assert(this->getrows() == mtx.getrows() && this->getcols() == mtx.getcols());
	for(int i = 0; i < rows*cols; i++){
		(*this)(i) += mtx(i);
	}
}

template <class mtype>
void Matrix<mtype>::operator-=(const Matrix<mtype> &mtx){
	assert(this->getrows() == mtx.getrows() && this->getcols() == mtx.getcols());
	for(int i = 0; i < rows*cols; i++){
		(*this)(i) -= mtx(i);
	}
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::operator! (){
	t = !t;
	Matrix<mtype> tmp(*this);
	t = !t;
	return tmp;
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::pointwisemult(const Matrix<mtype> &mtx){
	assert(mtx.getrows() == this->getrows() && mtx.getcols() == this->getcols());
	Matrix<mtype> tmp(this->getrows(), this->getcols());
	for(int i = 0; i < tmp.size(); i++){
		tmp(i) = mtx(i) * (*this)(i);
	}
	return tmp;
}

template <class mtype>
Matrix<mtype> Matrix<mtype>::pointwiseoperator(mtype (*func)(mtype)){
	Matrix<mtype> tmp(this->getrows(), this->getcols());
	for(int i = 0; i < this->size(); i++){
		tmp(i) = func((*this)(i)); 
	}
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