#include <iostream>
#include <vector>
#include <utility>
#include <ctime>
#include <cmath>
#include <timer.cpp>
#include <algorithm>
#include "connectfour.cpp"
//mcts for tictactoe
class State{
	CFBoard board;
	int visitcount;
	double wincount;
public:
	State(){
		visitcount = 1;
		wincount = 0;
	}
	State(CFBoard board) : board(board){
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
	CFBoard &getboard() { return board; }
	std::vector<int> listofmoves() { return board.listofmoves(); }
};


class Node{
	State state;
	Node *parent;
	std::vector<Node*> children;
	int number_of_possible_moves;
	int move;
public:
	Node(){
		this->parent = NULL;
		number_of_possible_moves = state.getboard().listofmoves().size();
	}
	Node(CFBoard board) : state(board){
		this->parent = NULL;
		number_of_possible_moves = board.listofmoves().size();
	}
	Node(Node* parent, int move) :state(parent->getstate().getboard()){
		this->getstate().getboard().move(move);
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
	int getmove() { return move; }
	void cleanup(){
		for(int i = 0; i < (int) children.size(); i++){
			if(children[i] != NULL) children[i]->cleanup();
		}
		delete this;
	}

};

class MCTSCF{
protected:
	Node *root;
public:
	MCTSCF(CFBoard board){
		// std::cout << "constructor" << std::endl;
		root = new Node(board);
		srand(time(NULL));
	}
	
	/* MCTS part */
	//--------------------------------------------
	int runMCTSinTime(long long timeinmicro = 1000000){
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
	int runMCTSforIters(int iters){
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
		std::vector<int> possiblemoves = leaf->getstate().getboard().listofmoves();
		if(possiblemoves.size() == 0) return;
		for(int i = 0; i < (int) leaf->getchildren().size(); i++){
			for(int j = 0; j < (int) possiblemoves.size()-i; j++){
				if(possiblemoves[j] == leaf->getchildren()[i]->getmove()){
					std::swap(possiblemoves[possiblemoves.size()-i-1], possiblemoves[j]);
					break;
				}
			}
		}
		int numberofchoise = possiblemoves.size()-leaf->getchildren().size();
		int tmp = possiblemoves[rand()%numberofchoise];
		leaf->getchildren().push_back(new Node(leaf, possiblemoves[rand()%numberofchoise]));
	}
	int Simulation(Node* leaf){
		CFBoard boardpos(leaf->getstate().getboard());
		while(!boardpos.if_win()) boardpos.playrandommove();
		return boardpos.if_win();
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
		if(-1*currnode->getstate().getboard().getturn() == result) currnode->getstate().wincountincr();

		do {
			currnode = currnode->getparent();
			if(-1*currnode->getstate().getboard().getturn() == result) currnode->getstate().wincountincr();
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
	void print_stats(){

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

	~MCTSCF(){
		root->cleanup();
	}
};


std::vector<int> simulategame(){
	CFBoard board;
	std::vector<int> game;
	while(!board.if_win()){
		MCTSCF mcts(board);
		int machinemove = mcts.runMCTSforIters(17);
		board.move(machinemove);
		game.push_back(machinemove);
		if(board.if_win()) break;
		MCTSCF minimax(board); 
		machinemove = minimax.runMinimax();
		board.move(machinemove);
		game.push_back(machinemove);
	}
	return game;
}
void stats(std::vector<std::vector<int>> &games){
	int results[] = {0,0,0};
	for(int i = 0; i < (int)games.size(); i++){
		CFBoard board;
		board.playgame(games[i]);
		results[std::abs(board.if_win()-1)]++;
	}
	std::cout << "winning: " << results[0] << std::endl;
	std::cout << "drawing: " << results[1] << std::endl;
	std::cout << "losing: " << results[2] << std::endl;
}