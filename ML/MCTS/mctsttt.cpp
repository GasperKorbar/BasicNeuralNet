#include <iostream>
#include <vector>
#include <utility>
#include <ctime>
#include <cmath>
#include <timer.cpp>
#include <algorithm>
#include "ttt.cpp"
//mcts for tictactoe
class State{
	ttt board;
	int visitcount;
	double wincount;
public:
	State(){
		visitcount = 1;
		wincount = 0;
	}
	State(ttt board) : board(board){
		visitcount = 1;
		wincount = 0;
	}
	State(const State& copyState) 
	: board     (copyState.board),
	  visitcount(copyState.visitcount),
	  wincount (copyState.wincount) {}
	int getvisitcount() { return visitcount; }
	double getwincount() { return wincount; }
	void visitcountincr() { visitcount++; }
	void wincountincr(double increment = 1) { wincount+=increment; }
	ttt &getboard() { return board; }
	std::vector<std::pair<int, int>> listofmoves() { return board.listofmoves(); }
	void playrandommove(){
		std::vector<std::pair<int, int>> possiblemoves = board.listofmoves();
		std::pair<int, int> m = possiblemoves[rand()%possiblemoves.size()];
		board.move(m.first, m.second);
	}
};


class Node{
	State state;
	Node *parent;
	std::vector<Node*> children;
	int number_of_possible_moves;
	std::pair<int, int> move;
public:
	Node(){
		this->parent = NULL;
		number_of_possible_moves = state.getboard().listofmoves().size();
	}
	Node(ttt board) : state(board){
		this->parent = NULL;
		number_of_possible_moves = board.listofmoves().size();
	}
	Node(Node* parent, std::pair<int, int> move) :state(parent->getstate().getboard()){
		this->getstate().getboard().move(move.first, move.second);
		this->move = move;
		this->parent = parent;
		number_of_possible_moves = state.getboard().listofmoves().size(); 
	}

	std::vector<Node*> &getchildren(){
		return children;
	}
	Node *getparent(){ return parent; }
	State& getstate(){	return state; }
	int get_number_of_moves() { return number_of_possible_moves; }
	std::pair<int, int> getmove() { return move; }
	void cleanup(){
		for(int i = 0; i < (int) children.size(); i++){
			if(children[i] != NULL) children[i]->cleanup();
		}
		delete this;
	}

};

class MonteCarloTreeSearch{
	Node *root;
public:
	MonteCarloTreeSearch(ttt board){
		// std::cout << "constructor" << std::endl;
		root = new Node(board);
		srand(time(NULL));
	}
	
	/* MCTS part */
	//--------------------------------------------
	std::pair<int, int> runMCTSinTime(long long timeinmicro = 1000000){
		long long t = 0;
		Node *leafnode;
		while(t < timeinmicro){
			TIMERret _(t);
			leafnode = Selection();
			Expansion(leafnode);
			Node *newnode;
			if(leafnode->get_number_of_moves() != 0)
				newnode = leafnode->getchildren()[leafnode->getchildren().size()-1];
			else newnode = leafnode;
			Backprop(newnode, Simulation(newnode));
		}
		int moveindex = getbestmove();
		return root->getchildren()[moveindex]->getmove();

	}
	std::pair<int, int> runMCTSforIters(int iters){
		Node *leafnode;
		int counter = 0;
		while(counter < iters){
			leafnode = Selection();
			Expansion(leafnode);
			Node *newnode;
			if(leafnode->get_number_of_moves() != 0)
				newnode = leafnode->getchildren()[leafnode->getchildren().size()-1];
			else newnode = leafnode;
			Backprop(newnode, Simulation(newnode));
			counter++;

		}
		int moveindex = getbestmove();
		return root->getchildren()[moveindex]->getmove();

	}


	Node* Selection(){
		Node *currnode = root;
		currnode->getstate().visitcountincr();
		int counter = 0;
		while((int)currnode->getchildren().size() == currnode->get_number_of_moves() && currnode->get_number_of_moves() != 0){
			currnode = currnode->getchildren()[choosemove(currnode)];
			currnode->getstate().visitcountincr();
			counter++;
		}
		return currnode;
	}
	void Expansion(Node *leaf){
		std::vector<std::pair<int, int>> possiblemoves = leaf->getstate().getboard().listofmoves();
		if(possiblemoves.size() == 0) return;
		for(int i = 0; i < (int) leaf->getchildren().size(); i++){
			for(int j = 0; j < (int) possiblemoves.size()-i; j++){
				if(possiblemoves[j].first == leaf->getchildren()[i]->getmove().first && possiblemoves[j].second == leaf->getchildren()[i]->getmove().second){
					std::swap(possiblemoves[possiblemoves.size()-i-1], possiblemoves[j]);
					break;
				}
			}
		}
		int numberofchoise = possiblemoves.size()-leaf->getchildren().size();
		std::pair<int, int> tmp = possiblemoves[rand()%numberofchoise];
		leaf->getchildren().push_back(new Node(leaf, possiblemoves[rand()%numberofchoise]));
	}
	int Simulation(Node* leaf){
		ttt boardpos(leaf->getstate().getboard());
		while(!boardpos.end()) boardpos.playrandommove();
		return boardpos.end();
	}
	void Backprop(Node* leaf, int result){
		if(result == 2){
			Node *currnode = leaf;
			do {
				currnode->getstate().wincountincr(0.5);
				currnode = currnode->getparent();
			} while(currnode->getparent());

		}
		Node *currnode = leaf;
		if(-1*currnode->getstate().getboard().getside() == result) currnode->getstate().wincountincr();

		do {
			currnode = currnode->getparent();
			if(-1*currnode->getstate().getboard().getside() == result) currnode->getstate().wincountincr();
		} while(currnode->getparent() != NULL);
	}
	int getbestmove(){
		// it is choosen by most visited node in selection
		int index = 0;
		int maxvisitcount = root->getchildren()[0]->getstate().getvisitcount();
		for(int i = 0; i < (int)root->getchildren().size(); i++){
			if(root->getchildren()[i]->getstate().getvisitcount() > maxvisitcount){
				maxvisitcount = root->getchildren()[i]->getstate().getvisitcount();
				index = i;
			}
		}
		return index;
	}
	//--------------------------------------------
	/* UCT part */
	//--------------------------------------------
	int choosemove(Node * currnode){
		double UCTscore = UCTvalue(currnode->getchildren()[0]);
		int index = 0;
		for(int i = 0; i < currnode->getchildren().size(); i++){
			double currUCTscore = UCTvalue(currnode->getchildren()[i]);
			if(currUCTscore > UCTscore){
				UCTscore = currUCTscore;
				index = i;
			}
		}
		return index;
	}
	double UCTvalue(Node * currnode){
		return (currnode->getstate().getwincount() / currnode->getstate().getvisitcount()
				+ sqrt(2.0*log(currnode->getparent()->getstate().getvisitcount())/currnode->getstate().getvisitcount()));
	}
	//--------------------------------------------
	/* tree related methods */
	//--------------------------------------------
	int treesize(Node* start){
		int s = 1;
		for(int i = 0; i < (int) start->getchildren().size(); i++){
			s += treesize(start->getchildren()[i]);
		}
		return s;
	}
	int wholetreesize(){
		return treesize(root);
	}
	void createSubTree(int index){
		Node*tmp = root->getchildren()[index];
		root->getchildren()[index] = NULL;
		root->cleanup();
		root = tmp;
	}
	//--------------------------------------------
	/* minimax stuff */
	//--------------------------------------------
	std::pair<int, int> runMinimax(){
		ttt currpos(root->getstate().getboard());
		int side = currpos.getside();
		std::vector<std::pair<int, int>> possiblemoves = currpos.listofmoves();
		std::vector<int> scores(possiblemoves.size());
		for(int i = 0; i < (int)possiblemoves.size(); i++) {
			scores[i] = side*Minimax(currpos.retmove(possiblemoves[i]));
		}
		int index = 0;
		int m = scores[0];
		for(int i = 0; i < (int)possiblemoves.size(); i++){
			std::cout << scores[i] << std::endl;
			if(scores[i] > m){
				m = scores[i];
				index = i;
			}
		}
		return possiblemoves[index];

	}
	int Minimax(ttt board){
		int side = board.getside();
		int m = -side;
		if(board.end()){
			if(board.end() == 2) return 0;
			return board.end();
		} else {
			std::vector<std::pair<int, int>> possiblemoves = board.listofmoves();
			for(int i = 0; i < (int) possiblemoves.size(); i++)
				m = side*std::max(side*m, side* Minimax(board.retmove(possiblemoves[i])));
		}
		return m;
	}

	//--------------------------------------------

	~MonteCarloTreeSearch(){
		root->cleanup();
	}
};


std::vector<std::pair<int, int>> simulategame(){
	ttt board;
	std::vector<std::pair<int, int>> game;
	while(!board.end()){
		MonteCarloTreeSearch mcts(board);
		std::pair<int, int> machinemove = mcts.runMCTSforIters(17);
		board.move(machinemove.first, machinemove.second);
		game.push_back(std::make_pair(machinemove.first, machinemove.second));
		if(board.end()) break;
		MonteCarloTreeSearch minimax(board); 
		machinemove = minimax.runMinimax();
		board.move(machinemove.first, machinemove.second);
		game.push_back(std::make_pair(machinemove.first, machinemove.second));
	}
	return game;
}
void stats(std::vector<std::vector<std::pair<int, int>>> &games){
	int results[] = {0,0,0};
	for(int i = 0; i < (int)games.size(); i++){
		ttt board;
		board.playgame(games[i]);
		results[std::abs(board.end()-1)]++;
	}
	std::cout << "winning: " << results[0] << std::endl;
	std::cout << "drawing: " << results[1] << std::endl;
	std::cout << "losing: " << results[2] << std::endl;
}