#include <iostream>
#include <vector>
#include <utility>
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

// tttNN copied from NN/NN.cpp
class tttNN{
protected:
	std::vector<Matrix<float>*> w, layers, activations, deltas, bias, gradient, biasgradient, accumulategrad, accumulatebias;
	float alpha;
public:
	tttNN(std::vector<int> topology, float alpha = 0.06){
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
				accumulategrad.push_back(new Matrix<float>(topology[i-1], topology[i]));
				accumulatebias.push_back(new Matrix<float>(topology[i], 1));
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
	float test(std::vector<std::vector<float>> &positions, std::vector<Matrix<float>> &labels){
		int n = positions.size();
		float correct = 0;
		for(int i = 0; i < n; i++){
			loadinput(positions[i]);	
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
	Matrix<float> prediction(std::vector<float> vec){
		loadinput(vec);
		forwardProp();
	
		int l = activations.size();
		return *activations[l-1];
		// float m = (*activations[l-1])(0);
		// int index = 0;
		// for(int a = 0; a < activations[l-1]->size(); a++){
		// 	if((*activations[l-1])(a) > m){
		// 		index = a;
		// 		m = (*activations[l-1])(a);
		// 	}
		// }
		// return index;	
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
		float tmpalpha = alpha;
		float decayfact = 0.8;
		for(int i = 0; i < n; i++){
			*accumulategrad[i] = (decayfact * (*accumulategrad[i])) + ((1-decayfact)* (gradient[i]->pointwisemult(*gradient[i]))); 
			*accumulatebias[i] = (decayfact * (*accumulatebias[i])) + ((1-decayfact)* (bias[i]->pointwisemult(*bias[i]))); 
			(*w[i]) -= accumulategrad[i]->pointwiseoperator([](float l) -> float{return 1/sqrt(l+0.0000001);}).pointwisemult((tmpalpha/128)*(*gradient[i]));
			(*bias[i]) -= accumulatebias[i]->pointwiseoperator([](float l) -> float{return 1/sqrt(l+0.0000001);}).pointwisemult((tmpalpha/128)*(*biasgradient[i]));
		}
		//if(epochs>0)w[n-1]->print();
	}
	void trainNetwork(std::vector<std::vector<float>> &positions, std::vector<Matrix<float>> &labels, int epochs){
		int minibatchsize = 128;
		int n = positions.size();
		std::vector<int> indexes(n);
		for(int i = 0; i < n; i++) indexes[i] = i;
		weightinit();
		for(int i = 0; i < epochs; i++){
			TIMER ti;
			std::cout << "Epoch: " << i << " test score: " << test(positions, labels)  << " " << lossfunction(positions, labels)<< std::endl; 
			for(int j = 0; j < n; j++){
				std::swap(indexes[j], indexes[rand()%n]);
			}
			long long forproptime = 0, bproptime = 0, gradupdatetime = 0;
			for(int j = 0; j < n; j+=minibatchsize){
				std::cout << j/minibatchsize << " ";
			 	for(int i = 0; i < (int)w.size(); i++){
					gradient[i]->clear();
					biasgradient[i]->clear();
				}
			 	for(int k = j; k < n && k < j+minibatchsize; k++){
			 		loadinput(positions[indexes[k]]);
				 	{TIMERret forpropr(forproptime); forwardProp();}
				 	setlastdeltas(labels[indexes[k]]);
				 	{TIMERret lll(bproptime);backProp();}
				 	for(int a = 0; a < (int) w.size(); a++){
				 		{TIMERret gradupdatett(gradupdatetime);
				 		// int gradcols = gradient[a]->getcols();
				 		// for(int h = 0; h < gradient[a]->size(); h++){
				 		// 	(*gradient[a])(h) += (*activations[a])(h/gradcols)*(*deltas[a+1])(h%gradcols);
				 		// }
				 		*gradient[a] += activations[a]->fastmult_vvt(*deltas[a+1]);
				 		}
				 		*biasgradient[a] += *deltas[a+1];}
				 	}
			 	update(i);
			 }
			 writeNNtofile(".\\weights.txt");
			 std::cout << std::endl;
			 std::cout << "forproptime " <<forproptime << std::endl;
			 std::cout << "bproptime " << bproptime << std::endl;
			 std::cout << "gradupdatetime " << gradupdatetime << std::endl;
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
	}

	~tttNN(){
		for(int i = 0; i < (int)layers.size(); i++) {delete layers[i]; delete deltas[i];}
		for(int i = 0; i < (int)activations.size() && activations[i]!=layers[i]; i++) delete activations[i];
		for(int i = 0; i < (int)w.size(); i++) {
			delete w[i]; delete bias[i]; delete gradient[i]; delete biasgradient[i]; delete accumulategrad[i];delete accumulatebias[i];
		}
	}

};

using namespace std;

class ttt{
	int board[3][3];
	int side;
public:
	ttt();
	ttt(const ttt&);
	int end();
	int if_win();
	int getside();
	void move(int, int);
	void remove(int, int);
	void print_ttt();
	void playgame(vector<pair<int, int>>);
	char XorO(int);
	vector<float> retboard();
	vector<pair<int, int>> listofmoves();
};

ttt::ttt(){
	for(int i = 0; i < 9; i++){
		board[i/3][i%3] = 0;
	}
	side = 1;
}

ttt::ttt(const ttt& copyttt){
	for(int i = 0; i < 9; i++)
		board[i/3][i%3] = copyttt.board[i/3][i%3];
	side = copyttt.side; 
}

int ttt::getside(){ return side; }

vector<float> ttt::retboard(){
	vector<float> tmp(9);
	for(int i = 0; i < 9; i++){
		tmp[i] = board[i/3][i%3];
	}
	return tmp;
}

void ttt::move(int y, int x){
	if(y > 2 || x > 2) return;
	if(board[y][x]== 0){
		board[y][x] = side;
		side *= -1;
	}
}

void ttt::remove(int y, int x){
	if(y > 2 || x > 2) return;
	if(board[y][x] != 0){
		board[y][x] = 0;
		side *= -1;
	}

}
char ttt::XorO(int s){
	if(s == 1) return 'X';
	else if(s == -1) return 'O';
	else return ' ';
}

int ttt::if_win(){
	for(int i = 0; i < 3; i++)
		if(abs(board[i][0]+board[i][1]+board[i][2]) == 3) return board[i][0];
	for(int i = 0; i < 3; i++)
		if(abs(board[0][i]+board[1][i]+board[2][i]) == 3) return board[0][i];
	if(abs(board[0][0] + board[1][1] + board[2][2]) == 3) return board[1][1];
	if(abs(board[2][0] + board[1][1] + board[0][2]) == 3) return board[1][1];
	return 0;
}

// 1 if x wins, -1 if o wins, 0 if not over, 2 if draw
int ttt::end(){
	int ifwin = if_win();
	if(ifwin) return ifwin;
	for(int i = 0; i < 9; i++){
		if(board[i/3][i%3] == 0) return 0;
	}
	return 2;
}

void ttt::print_ttt(){
	for(int i = 0; i < 3; i++){
		cout << "+-+-+-+" << endl;
		for(int j = 0; j < 3; j++){
			cout << "|" << XorO(board[i][j]);
		}
		cout << "|" << endl;
	}
	cout << "+-+-+-+" << endl;
}
void ttt::playgame(vector<pair<int, int>> game){
	for(int i = 0; i < (int) game.size(); i++){
		move(game[i].first, game[i].second);
	}
}

vector<pair<int, int>> ttt::listofmoves(){
	vector<pair<int, int>> tmp;
	if(end()) return tmp;
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			if(board[i][j] == 0){
				tmp.push_back(make_pair(i, j));
			}
		}
	}
	return tmp;
}
void printlistofmoves(vector<pair<int,int>> l){
	for(int i = 0; i < (int) l.size(); i++){
		cout << "(" << l[i].first << ", " << l[i].second << ")";
		if(i != l.size()-1) cout << ", ";
	}
	cout << endl;
}

pair<int, int> bestmove(ttt board, tttNN *player, float rnd = -1){
	vector<pair<int, int>> moves = board.listofmoves();
	vector<float> scores;
	int side = board.getside();
	for(int i = 0; i < (int) moves.size(); i++){
		ttt tmpboard(board);
		tmpboard.move(moves[i].first, moves[i].second);
		Matrix<float> currscore(player->prediction(tmpboard.retboard()));
		// (!currscore).print();
		if(side == -1){
			float tmp = currscore(0);
			currscore(0) = currscore(2);
			currscore(2) = tmp;
		}
		if(currscore(0)-currscore(1)-currscore(2) > 0){
			scores.push_back(currscore(0)-currscore(2));
		} else {
			scores.push_back(currscore(1)-currscore(2));
		}
	}
	int count = moves.size();
	int index;
	while(1) {
		float m = scores[0];
		index = 0;
		for(int i = 0; i < count; i++){
			if(scores[i] > m){
				m = scores[i];
				index = i;
			}
		}
		// std::cout << index << std::endl;
		if(((float)rand()/RAND_MAX * rnd > 0.5) || rnd == -1) return moves[index];
		swap(scores[index], scores[count-1]);
		swap(moves[index], moves[count-1]);
		count--;
		if(count == 0) break;

	}
	return moves[rand()%(moves.size())];
}

vector<pair<int, int>> simulategame(tttNN *p1=NULL, tttNN *p2=NULL, float rnd = 0, float rnd1 = 1) {
	vector<pair<int, int>> history;
	ttt board;
	float count = 0;
	while(!board.end()){
		pair<int, int> move(-1, -1);
		if(board.getside() == 1 && p1 != NULL)
			move = bestmove(board, p1, rnd+count*rnd1);
		else if(board.getside() == -1 && p2 != NULL)
			move = bestmove(board, p2, rnd+count*rnd1);
		else {
			vector<pair<int, int>> moveslist = board.listofmoves();
			move = moveslist[rand()%(moveslist.size())];
		}
		history.push_back(move);
		board.move(move.first, move.second);
		count++;
	}

	return history;
}

void gamestats(vector<vector<pair<int, int>>> &games, int player = 1){
	vector<int> stats = {0, 0, 0};
	float n = games.size();
	for(int i = 0; i < n; i++){
		ttt board;
		board.playgame(games[i]);
		int result = board.end();
		if(result == player) stats[0]++;
		else if(result == -player) stats[2]++;
		else if(result == 2)stats[1]++;
	}

	cout << "win percent: " << (float)100*stats[0]/n << endl;
	cout << "draw percent: " << (float)100*stats[1]/n << endl;
	cout << "loss percent: " << (float)100*stats[2]/n << endl;
}
void preparedata(vector<vector<pair<int, int>>> &data, vector<vector<float>>& positions, vector<Matrix<float>> &labels){
	positions.clear();
	labels.clear();
	for(int i = 0; i < (int) data.size(); i++){	
		ttt board;
		int m = data[i].size();
		for(int j = 0; j < m; j++){
			int side = board.getside();
			board.move(data[i][j].first, data[i][j].second);
			vector<float> vecboard = board.retboard();
			positions.push_back(vecboard);
		}
		for(int j = 0; j < m; j++){
			Matrix<float> l(3, 1);
			l(abs(board.end()-1)) = 1;
			labels.push_back(l);
		}
	}

}

// https://medium.com/swlh/tic-tac-toe-and-deep-neural-networks-ea600bc53f51

int main(){
	srand(time(NULL));
	tttNN network({9, 200, 125, 75, 25, 3}, 0.003);
	network.addActivation<ReLU>({1, 2, 3, 4});
	network.addActivation<Softmax>({5});
	// network.readNNfromfile(".\\weights.txt");
	vector<vector<pair<int, int>>> randomgames;
	for(int i = 0; i < 10000; i++){
		randomgames.push_back(simulategame());
	}
	vector<vector<float>> positions;
	vector<Matrix<float>> labels;
	preparedata(randomgames, positions, labels);
	network.trainNetwork(positions, labels, 10);

	vector<vector<pair<int, int>>> randomgamestest;
	for(int i = 0; i < 1000; i++){
		randomgamestest.push_back(simulategame(NULL, &network, -1));
	}
	gamestats(randomgamestest);

	// ttt board;
	// float counter = 0;
	// while(!board.end()){
	// 	int x, y;
	// 	pair<int, int> machinemove = bestmove(board, &network, counter);
	// 	board.move(machinemove.first, machinemove.second);
	// 	board.print_ttt();
	// 	if(board.end()) break;
	// 	cin >> y >> x;
		
	// 	board.move(y, x);
	// 	board.print_ttt();
	// 	if(counter != -1) counter+=1;
	// 	if(counter == 3) counter = -1;
	// }


	return 0;
}