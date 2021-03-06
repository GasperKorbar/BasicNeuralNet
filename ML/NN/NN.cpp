#include <Matrix.cpp>
#include "Activation.cpp"
#include <ctime>
#include <cmath>
#include <fstream>
#include <limits>
#include "readMNISTfiles.cpp"
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
public:
	NN(std::vector<int> topology, float alpha = 0.06){
		assert(topology.size() > 1);
		srand(time(NULL));
		int n = topology.size();
		this->alpha = alpha;
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

	void loadinput(std::vector<float> &input){
		assert(input.size() == layers[0]->size());
		for(int i = 0; i < (int) input.size(); i++)
			(*layers[0])(i) = input[i];
	}

	void setlastdeltas(Matrix<float> &label){
		(*deltas[deltas.size()-1]) = (*activations[activations.size()-1]) - label;
	}
	void forwardProp(){
		for(int i = 0; i < (int)layers.size()-1; i++){
			*layers[i+1] = w[i]->tmult(*activations[i]) + *bias[i];
			if(activations[i+1] != layers[i+1]){
 				// *activations[i+1] = *layers[i+1];
				Activation* tmp = dynamic_cast<Activation*>(activations[i+1]);
				tmp->applyactivation(*layers[i+1]);

			}
		}
	}
	float test(std::vector<std::vector<float>> &images, std::vector<Matrix<float>> &labels){
		int n = images.size();
		float correct = 0;
		for(int i = 0; i < n; i++){
			loadinput(images[i]);	
			forwardProp();
			int l = activations.size();
			float m = (*activations[l-1])(0);
			int index = 0;
			for(int a = 0; a < activations[l-1]->size(); a++){
				if((*activations[l-1])(a) > m){
					index = a;
					m = (*activations[l-1])(a);
				}
			}
			if(labels[i](index) == 1) correct++;
		}
		return 100*correct/n;
	}
	float lossfunction(std::vector<std::vector<float>> &input, std::vector<Matrix<float>> &solution){
		float sum = 0;
		int l = input.size();
		int n = layers.size();
		for(int i = 0; i < l; i++){
			loadinput(input[i]);
			forwardProp();
			Matrix<float> tmp = *activations[n-1];
			for(int i = 0; i < tmp.getrows(); i++) tmp(i) = log(tmp(i));
			float logloss = -solution[i].tmult(tmp)(0);
			sum += logloss;
		}
		return sum;
	}
	int prediction(std::vector<float> vec){
		loadinput(vec);
		forwardProp();
		int l = activations.size();
		float m = (*activations[l-1])(0);
		int index = 0;
		for(int a = 0; a < activations[l-1]->size(); a++){
			if((*activations[l-1])(a) > m){
				index = a;
				m = (*activations[l-1])(a);
			}
		}
		return index;	
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
	void update(int epochs){
		int n = gradient.size();
		float tmpalpha = alpha*exp(-epochs*0.1);
		for(int i = 0; i < n; i++){
			(*w[i]) -= (tmpalpha/64)*(*gradient[i]);
			(*bias[i]) -= (tmpalpha/64)*(*biasgradient[i]);
		}
	}
	void trainNetwork(std::vector<std::vector<float>> &images, std::vector<Matrix<float>> &labels, int epochs){
		int minibatchsize = 64;
		int n = images.size();
		std::vector<int> indexes(n);
		for(int i = 0; i < n; i++) indexes[i] = i;
		weightinit();
		writeNNtofile("C:\\projects\\ML\\NN\\weights.txt");
		for(int i = 0; i < epochs; i++){
			if(i != 0)std::cout << "Epoch: " << i << " test score: " << test(images, labels)  << " " << lossfunction(images, labels)<< std::endl; 
			TIMER global;
			for(int j = 0; j < n; j++){
				std::swap(indexes[j], indexes[rand()%n]);
			}
			for(int j = 0; j < n; j+=minibatchsize){
				TIMER timerrr;
				std::cout << j/minibatchsize << " ";
			 	for(int i = 0; i < (int)w.size(); i++){
					gradient[i]->clear();
					biasgradient[i]->clear();
				}
			 	for(int k = j; k < n && k < j+minibatchsize; k++){
			 		loadinput(images[indexes[k]]);
				 	forwardProp();
				 	setlastdeltas(labels[indexes[k]]);
				 	backProp();
				 	for(int a = 0; a < (int) w.size(); a++){
				 		int gradcols = gradient[a]->getcols();
				 		for(int h = 0; h < gradient[a]->size(); h++){
				 			(*gradient[a])(h) += (*activations[a])(h/gradcols)*(*deltas[a+1])(h%gradcols);
				 		}
				 		// very slow code: 50 msec per batch *gradient[a] += (*activations[a])*(!(*deltas[a+1]));
				 		*biasgradient[a] += *deltas[a+1];
				 	}
			 	}
			 	update(i);
			 }
			 writeNNtofile("C:\\projects\\ML\\NN\\weights.txt");
			 std::cout << std::endl;
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
		} else {
			std::cout << "file: " << filename << " couldn't open" << std::endl;
		}

	}
	void readNNfromfile(std::string filename){
		std::ifstream file(filename.c_str());
		if(file.is_open()){
			for(int i = 0; i < (int) w.size(); i++){
				for(int j = 0; j < w[i]->getrows()*w[i]->getcols(); j++){
					file >> (*w[i])(j);
				}
			}
		} else {
			std::cout << "file: " << filename << " couldn't open" << std::endl;
		}
	}

	~NN(){
		for(int i = 0; i < (int)layers.size(); i++) {delete layers[i]; delete deltas[i];}
		for(int i = 0; i < (int)activations.size() && activations[i]!=layers[i]; i++) delete activations[i];
		for(int i = 0; i < (int)w.size(); i++) {
			delete w[i]; delete bias[i]; delete gradient[i]; delete biasgradient[i];
		}
	}

};

using namespace std;
void printmnistnumber(std::vector<float> vec){
	for(int i = 0; i < 28*28; i++){
		if((i)%28 == 0) cout << endl;
		if(vec[i]>0) cout << "#";
		else cout << " ";
	}
	cout << endl;
}

int main(){
	srand(time(NULL));
	NN network({784, 300, 10});
	network.addActivation<ReLU>({1});
	network.addActivation<Softmax>({2});
	std::vector<std::vector<float>> images = read_mnist_images("C:\\projects\\ML\\NN\\train-images.idx3-ubyte");
	std::vector<Matrix<float>> labels = read_mnist_labels("C:\\projects\\ML\\NN\\train-labels.idx1-ubyte");
	int n = images.size();
	network.trainNetwork(images, labels, 10);
	// network.readNNfromfile("C:\\projects\\ML\\NN\\weights.txt");
	cout << network.test(images, labels) << endl;
	// for(int i = 0; i< 100; i++){
	// 	printmnistnumber(images[n-i-1]);
	// 	int pred = network.prediction(images[n-i-1]);
	// 	if(labels[n-i-1](network.prediction(images[n-i-1])) == 1) cout << "pravilno " << pred << endl;
	// 	else cout << "napacno " << pred << endl;
	// }
	
	return 0;

}