#ifndef DIRICHLET_H
#define DIRICHLET_H
#include <random>
#include <vector>
// copied from https://github.com/gcant/dirichlet-cpp/blob/master/dirichlet.h
template <class RNG>
class dirichlet_distribution{
	public:
		dirichlet_distribution(const std::vector<float>&);
		void set_params(const std::vector<float>&);
		std::vector<float> get_params();
		std::vector<float> operator()(RNG&);
	private:
		std::vector<float> alpha;
		std::vector<std::gamma_distribution<>> gamma;
};

template <class RNG>
dirichlet_distribution<RNG>::dirichlet_distribution(const std::vector<float>& alpha){
	set_params(alpha);
}

template <class RNG>
void dirichlet_distribution<RNG>::set_params(const std::vector<float>& new_params){
	alpha = new_params;
	std::vector<std::gamma_distribution<>> new_gamma(alpha.size());
	for (int i=0; i<(int)alpha.size(); ++i){
		std::gamma_distribution<> temp(alpha[i], 1);
		new_gamma[i] = temp;
	}
	gamma = new_gamma;
}

template <class RNG>
std::vector<float> dirichlet_distribution<RNG>::get_params(){
	return alpha;
}

template <class RNG>
std::vector<float> dirichlet_distribution<RNG>::operator()(RNG& generator){
	std::vector<float> x(alpha.size());
	float sum = 0.0;
	for (int i=0; i<(int)alpha.size(); ++i){
		x[i] = gamma[i](generator);
		sum += x[i];
	}
	for (float &xi : x) xi = xi/sum;
	return x;
}

#endif