#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Matrix.cpp>
#include <cmath>
class Activation : public Matrix<float>{
public:
	Activation(const Matrix<float> &mtx) : Matrix<float>(mtx){}
	virtual void applyactivation(const Matrix<float>&m){
		for(int i = 0; i < cols*rows; i++) matrix[i] = activationfunc(m(i));
	}
	virtual void applyderivative(Matrix<float> &m){
		for(int i = 0; i < cols*rows; i++) m(i) *= activationderivative((*this)(i));
	}
	virtual float activationfunc(float) = 0;
	virtual float activationderivative(float) = 0;
	virtual ~Activation(){}
};

class Sigmoid : public Activation{
public:
	Sigmoid(const Matrix<float> &mtx) : Activation(mtx){}
	float activationfunc(float a){
		float tmp = exp(a);
		return tmp/(1+tmp);
	}
	float activationderivative (float a){
		return a*(1-a);
	}
};
class fastSigmoid : public Activation{
public:
	fastSigmoid(const Matrix<float> &mtx) : Activation(mtx){}
	float activationfunc(float a){
		float tmp = exp(a);
		return a/(2*(1+abs(a))) + 0.5;
	}
	float activationderivative (float a){
		return a*(1-a);
	}
};

class ReLU : public Activation {
public:
	ReLU(const Matrix<float> &mtx) : Activation(mtx){}
	float activationfunc(float a){
		if(a > 0) return a;
		else return 0;
	}
	float activationderivative (float a){
		if(a > 0) return 1;
		else return 0; 
	}
};

//only used at the end of neural network, with one hot encoded predictions
class Softmax : public Activation{
public:
	Softmax(const Matrix<float> &mtx) : Activation(mtx){}
	void applyactivation(const Matrix<float>&mtx){
		float sum = 0;
		float m = 0;
		for(int i = 0; i < this->size(); i++) m = std::max(m, (mtx)(i));
		for(int i = 0; i < this->size(); i++) sum += exp((mtx)(i) - m);
		if(sum < 0) std::cout << "nan is happening" << std::endl;
		for(int i = 0; i < this->size(); i++){
			(*this)(i) = exp((mtx)(i)-m-log(sum+1e-6));
		}
	}
	void applyderivative(Matrix<float> &m){
 		for(int i = 0; i < cols*rows; i++) m(i) = (*this)(i);
	}
	float activationfunc(float a){
		std::cout << "I am never used" << std::endl;
		float sum = 0;
		for(int i = 0; i < getrows(); i++) sum += exp((*this)(i));
		return exp(a)/sum;
	}
	float activationderivative (float a){
		std::cout << "I am never used" << std::endl;
		return a;
	}
};

#endif