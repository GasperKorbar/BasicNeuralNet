#ifndef NN_H
#define NN_H

#include <Matrix.cpp>
#include <Activation.cpp>
#include <ctime>
#include <cmath>
#include <fstream>
#include <limits>
#define PI 3.14159265358979

float box_muller(float mu = 0, float sigma = 1){
	constexpr float epsilon = std::numeric_limits<float>::epsilon();
	float u1, u2;
	do{
		u1 = (float)rand()/RAND_MAX;
		u2 = (float)rand()/RAND_MAX;
	} while(u1 <= epsilon);
	return (sqrt(-2*log(u1))*cos(2*PI*u2) * sigma + mu);
}

class NN{
protected:
	std::vector<Matrix<float>*> w, layers, activations, deltas, bias, gradient, biasgradient;
	float alpha;
	int minibatchsize;
public:
	NN(std::vector<int> topology, float alpha = 0.03, int minibatchsize = 64){
		assert(topology.size() > 1);
		srand(time(NULL));
		int n = topology.size();
		this->alpha = alpha;
		this->minibatchsize = minibatchsize;
		for(int i = 0; i < n; i++){
			layers.push_back(new Matrix<float>(topology[i], 1));
			activations.push_back(layers[i]);
			deltas.push_back(new Matrix<float>(topology[i], 1));
			if(i > 0) {
				w.push_back(new Matrix<float>(topology[i-1], topology[i]));
				bias.push_back(new Matrix<float>(topology[i], 1));
				gradient.push_back(new Matrix<float>(topology[i-1], topology[i]));
				biasgradient.push_back(new Matrix<float>(topology[i], 1));
			}
		}

	}
	template <class activationtype>
	void addActivation(std::vector<int> index){
		assert((std::is_base_of<Activation, activationtype>::value));
		for(int i = 0; i < (int) index.size(); i++){
			if(index[i] >= (int)(layers.size()) || index[i] == 0) continue;
			if(activations[index[i]] != layers[index[i]])
				delete activations[index[i]];
			activations[index[i]] = new activationtype(Matrix<float>(layers[index[i]]->getrows(), layers[index[i]]->getcols()));
		}
	}

	void weightinit(){
		float sd;
		for(int i = 0; i < (int) layers.size()-1; i++){
			sd = sqrt((float)2/layers[i]->size());
			for(int j = 0; j < w[i]->size(); j++){
				(*w[i])(j) = box_muller(0, sd);
			}
		}
	}

	void loadinput(const std::vector<float> &input){
		assert(input.size() == layers[0]->size());
		for(int i = 0; i < (int) input.size(); i++)
			(*layers[0])(i) = input[i];
	}

	virtual void setlastdeltas(Matrix<float> &label){
		(*deltas[deltas.size()-1]) = (*activations[activations.size()-1]) - label;
	}
	void forwardProp(){
		for(int i = 0; i < (int)layers.size()-1; i++){
			*layers[i+1] = w[i]->tmult(*activations[i]) + *bias[i];
			if(activations[i+1] != layers[i+1]){
				Activation* tmp = dynamic_cast<Activation*>(activations[i+1]);
				tmp->applyactivation(*layers[i+1]);

			}
		}
	}
	void backProp(){
 		for(int i = (int)layers.size()-1; i > 0; i--){
 			*deltas[i-1] = (*w[i-1]) * (*deltas[i]);
 			if(activations[i-1] != layers[i-1]) {
 				Activation* tmp = dynamic_cast<Activation*>(activations[i-1]);
 				tmp->applyderivative(*deltas[i-1]);
 			}
 		}
	}

	virtual float test(std::vector<std::vector<float>> &positions, std::vector<Matrix<float>> &labels){
		int n = positions.size();
		float correct = 0;
		for(int i = 0; i < n; i++){
			loadinput(positions[i]);
			forwardProp();
			int l = activations.size();
			float m = (*activations[l-1])(0), labelm = labels[i](0);

			int index = 0, labelindex = 0;
			for(int a = 0; a < activations[l-1]->size(); a++){
				if((*activations[l-1])(a) > m){
					index = a;
					m = (*activations[l-1])(a);
				}
				if(labels[i](a) > labelm){
					labelindex = a;
					labelm = (labels[i])(a);
				}
			}
			if(index == labelindex) correct++;
		}
		return 100*correct/n;
	}
	virtual float lossfunction(std::vector<std::vector<float>> &input, std::vector<Matrix<float>> &solution){
		float sum = 0;
		int l = input.size();
		int n = layers.size();
		for(int i = 0; i < l; i++){
			loadinput(input[i]);
			forwardProp();
			Matrix<float> tmp = *activations[n-1];
			for(int i = 0; i < tmp.getrows(); i++) tmp(i) = log(tmp(i)+1e-6);
			for(int i = 0; i < tmp.size(); i++)
				if(!std::isnormal(tmp(i))) std::cout << "nan problem again: " <<  tmp(i) <<std::endl; 
			float logloss = -solution[i].dotproduct(tmp);
			sum += logloss;
		}
		return sum;
	}
	Matrix<float> prediction(const std::vector<float> &vec){
		loadinput(vec);
		forwardProp();
		return *activations[activations.size()-1];
	}
	
	void update(int epochs){
		int n = gradient.size();
		float tmpalpha = alpha*exp(-epochs*0.1)/minibatchsize;
		float l2 = 0.01;
		for(int i = 0; i < n; i++){
			(*w[i]) -= (tmpalpha)*(*gradient[i]) + (tmpalpha * l2 * w[i]->pointwisemult(*w[i]));
			(*bias[i]) -= (tmpalpha)*(*biasgradient[i]) + (tmpalpha * l2 * bias[i]->pointwisemult(*bias[i]));
		}
	}
	void trainNetwork(std::vector<std::vector<float>> &positions, std::vector<Matrix<float>> &labels, int epochs){
		int n = positions.size();
		std::vector<int> indexes(n);
		for(int i = 0; i < n; i++) indexes[i] = i;
		for(int i = 0; i < epochs; i++){
			std::cout << "Epoch: " << i << ", test score: " << test(positions, labels)  << " loss function: " << lossfunction(positions, labels)<< std::endl; 
			TIMER global;
			for(int j = 0; j < n; j++){
				std::swap(indexes[j], indexes[rand()%n]);
			}
			for(int j = 0; j < (int)w.size(); j++){
				gradient[j]->clear();
				biasgradient[j]->clear();
			}
			long long forproptime = 0, backproptime = 0, gradientupdatetime = 0, updatetime = 0;
			for(int j = 0; j < n; j+=minibatchsize){
				for(int k = 0; k < (int)w.size(); k++){
			 		float momentum_param = 0.9;
					*gradient[k] = momentum_param*(*gradient[k]);
					*biasgradient[k] = momentum_param*(*biasgradient[k]);
				}
				std::cout << j/minibatchsize << " ";
			 	for(int k = j; k < n && k < j+minibatchsize; k++){
			 		loadinput(positions[indexes[k]]);
				 	{TIMERret tim(forproptime);forwardProp();}
				 	// if(i == 28 && ){
				 	// 	(!(*activations[activations.size()-1])).print();
				 	// }
				 	setlastdeltas(labels[indexes[k]]);
				 	{TIMERret tim(backproptime);backProp();}
				 	for(int a = 0; a < (int) w.size(); a++){
				 		{TIMERret tim(gradientupdatetime);*gradient[a] += activations[a]->fastmult_vvt(*deltas[a+1]);}
				 		*biasgradient[a] += *deltas[a+1];
				 	}
			 	}
			 	{TIMERret tim(updatetime); update(i);}
			 }
			 std::cout << std::endl;
			 std::cout << "forproptime " << forproptime << std::endl;
			 std::cout << "backproptime: "<<backproptime << std::endl;
			 std::cout << "gradientupdatetime: "<<gradientupdatetime << std::endl;
			 std::cout << "updatetime: "<<updatetime << std::endl;
		}
		

	}

	void writeNNtofile(std::string filename){
		std::ofstream file(filename.c_str());
		if(file.is_open()){
			for(int i = 0; i < (int) w.size(); i++){
				for(int j = 0; j < w[i]->getrows()*w[i]->getcols(); j++){
					file << (*w[i])(j) << " ";
				}
			}
			for(int i = 0; i < (int) w.size(); i++){
				for(int j = 0; j < bias[i]->size(); j++){
					file << (*bias[i])(j) << " ";
				}
			}
		} else {
			std::cout << "file: " << filename << " couldn't open" << std::endl;
		}
		file.close();

	}
	void readNNfromfile(std::string filename){
		std::ifstream file(filename.c_str());
		if(file.is_open()){
			for(int i = 0; i < (int) w.size(); i++){
				for(int j = 0; j < w[i]->getrows()*w[i]->getcols(); j++){
					file >> (*w[i])(j);
				}
			}
			for(int i = 0; i < (int) w.size(); i++){
				for(int j = 0; j < bias[i]->size(); j++){
					file >> (*bias[i])(j);
				}
			}
		} else {
			std::cout << "file: " << filename << " couldn't open" << std::endl;
		}
		file.close();
	}

	~NN(){
		for(int i = 0; i < (int)layers.size(); i++) {delete layers[i]; delete deltas[i];}
		for(int i = 0; i < (int)activations.size() && activations[i]!=layers[i]; i++) delete activations[i];
		for(int i = 0; i < (int)w.size(); i++) {
			delete w[i]; delete bias[i]; delete gradient[i]; delete biasgradient[i];
		}
	}

};

class NN_MSE : public NN{
public:
	NN_MSE(std::vector<int> topology, float alpha = 0.03, int minibatchsize = 64) : NN(topology, alpha, minibatchsize){}
	float test(std::vector<std::vector<float>> &positions, std::vector<Matrix<float>> &labels){
		int n = positions.size();
		float correct = 0;
		for(int i = 0; i < n; i++){
			loadinput(positions[i]);
			forwardProp();
			int l = activations.size();
			if(std::abs((*activations[l-1])(0) - labels[i](0)) < 0.1)
				correct++;
		}
		return 100*correct/n;
	}
	void setlastdeltas(Matrix<float> &label){
		int lastindex = deltas.size()-1;
		*deltas[lastindex] = (*activations[lastindex]) - label;
		Activation* tmp = dynamic_cast<Activation*>(activations[lastindex]);
		tmp->applyderivative(*deltas[lastindex]);
	}
	float lossfunction(std::vector<std::vector<float>> &input, std::vector<Matrix<float>> &solution){
		float sum = 0;
		int l = input.size();
		int n = layers.size();
		for(int i = 0; i < l; i++){
			loadinput(input[i]);
			forwardProp();
			float tmp = (*activations[n-1])(0);
			float mse = (tmp-solution[i](0))*(tmp-solution[i](0));
			sum += mse;
		}
		return sum/(2*n);
	}
};

#endif