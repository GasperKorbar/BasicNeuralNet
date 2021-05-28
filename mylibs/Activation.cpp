#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Matrix.cpp>
#include <cmath>
class Activation : public Matrix<double>{
public:
	Activation(const Matrix<double> &mtx) : Matrix<double>(mtx){}
	virtual void applyactivation(){
		for(int i = 0; i < cols*rows; i++) matrix[i] = activationfunc(matrix[i]);
	}
	virtual void applyderivative(Matrix<double> &m){
		for(int i = 0; i < cols*rows; i++) m(i) *= activationderivative((*this)(i));
	}
	virtual double activationfunc(double) = 0;
	virtual double activationderivative(double) = 0;
	virtual ~Activation(){}
};

class Sigmoid : public Activation{
public:
	Sigmoid(const Matrix<double> &mtx) : Activation(mtx){}
	double activationfunc(double a){
		double tmp = exp(a);
		return tmp/(1+tmp);
	}
	double activationderivative (double a){
		double tmp = activationfunc(a);
		return tmp*(1-tmp);
	}
};

class ReLU : public Activation {
public:
	ReLU(const Matrix<double> &mtx) : Activation(mtx){}
	double activationfunc(double a){
		if(a > 0) return a;
		else return 0;
	}
	double activationderivative (double a){
		if(a > 0) return 1;
		else return 0; 
	}
};

//only used at the end of neural network, with one hot encoded predictions
class Softmax : public Activation{
public:
	Softmax(const Matrix<double> &mtx) : Activation(mtx){}
	void applyactivation(){
		double sum = 0;
		double m = 0;
		for(int i = 0; i < this->size(); i++) m = std::max(m, (*this)(i));
		for(int i = 0; i < this->size(); i++) sum += exp((*this)(i) - m);
		for(int i = 0; i < this->size(); i++){
			(*this)(i) = exp((*this)(i)-m-log(sum));
		}
	}
	void applyderivative(Matrix<double> &m){
 		for(int i = 0; i < cols*rows; i++) m(i) = (*this)(i);
	}
	double activationfunc(double a){
		std::cout << "I am never used" << std::endl;
		double sum = 0;
		for(int i = 0; i < getrows(); i++) sum += exp((*this)(i));
		return exp(a)/sum;
	}
	double activationderivative (double a){
		std::cout << "I am never used" << std::endl;
		return a;
	}
};

#endif