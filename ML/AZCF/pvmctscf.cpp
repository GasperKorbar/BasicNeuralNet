#ifndef PVMCTSCF_H
#define PVMCTSCF_H

#include <iostream>
#include <vector>
#include <utility>
#include <ctime>
#include <cmath>
#include <timer.cpp>
#include <algorithm>
#include "connectfour.cpp"
#include "nns.cpp"
#include "dirichlet.h"
//mcts for tictactoe
class State{
	CFBoard board;
	int visitcount;
	float winscore;
	std::vector<float> policy;
public:
	State(){
		visitcount = 1;
		winscore = 0;
	}
	State(CFBoard board) : board(board){
		visitcount = 1;
		winscore = 0;
	}
	State(const State& copyState) 
	: board     (copyState.board),
	  visitcount(copyState.visitcount),
	  winscore (copyState.winscore) {}
	int getvisitcount() { return visitcount; }
	float getwinscore() { return winscore; }
	void visitcountincr() { visitcount++; }
	void winscoreincr(float increment = 1) { winscore+=increment; }
	std::vector<float> &getpolicy() { return policy; }
	CFBoard &getboard() { return board; }
};


class Node{
	State state;
	Node *parent;
	std::vector<Node*> children;
	int number_of_possible_moves;
	int move;
	float policyprobability;
public:
	Node(){
		this->parent = NULL;
		number_of_possible_moves = state.getboard().listofmoves().size();
	}
	Node(CFBoard board) : state(board){
		this->parent = NULL;
		number_of_possible_moves = board.listofmoves().size();
	}
	Node(Node* parent, int move, float policyprob) :state(parent->getstate().getboard()){
		this->getstate().getboard().move(move);
		this->move = move;
		this->parent = parent;
		policyprobability = policyprob;
		number_of_possible_moves = state.getboard().listofmoves().size(); 
	}

	std::vector<Node*> &getchildren(){
		return children;
	}
	float getpolicyprobability() { return policyprobability; }
	void setpolicyprobability(float policyprob) { policyprobability = policyprob; }
	Node *getparent(){ return parent; }
	void noparent(){ parent = NULL; }
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

class PVMCTSCF{
	Node *root;
	NN* policynet;
	NN_MSE* valuenet;
public:		

	PVMCTSCF(NN *policynetwork, NN_MSE *valuenetwork) {
		root = new Node();
		policynet = policynetwork;
		valuenet = valuenetwork;
		setpolicy(root);
		srand(time(NULL));
	}

	PVMCTSCF(CFBoard board, NN *policynetwork, NN_MSE *valuenetwork) {
		root = new Node(board);
		policynet = policynetwork;
		valuenet = valuenetwork;
		setpolicy(root);
		srand(time(NULL));
	}
	Node *retroot(){ return root; }
	
	/* MCTS part */
	//--------------------------------------------
	int runMCTSforIters(int iters, int modeofchoosing = 0){
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
		int moveindex;
		if(modeofchoosing == 0) moveindex = getbestmove();
		else moveindex = getsampledmove();
		int newmove = root->getchildren()[moveindex]->getmove();
		return newmove;
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
		int newmove = possiblemoves[rand()%numberofchoise];
		Node *newnode = new Node(leaf, newmove, leaf->getstate().getpolicy()[newmove]);
		setpolicy(newnode);
		leaf->getchildren().push_back(newnode);
	}
	float Simulation(Node* leaf){
		float winstatus = leaf->getstate().getboard().if_win();
		if(winstatus != 0) {
			if(winstatus == 2) return 0.5;
			leaf->setpolicyprobability(1.0);
			setpolicy(leaf->getparent(), leaf->getmove());
			return (winstatus+1)/2;
		}
		return valuenet->prediction(leaf->getstate().getboard().getnetform())(0);
		
	}
	void Backprop(Node* leaf, float result){
		Node *currnode = leaf;
		do {
			if(-1*currnode->getstate().getboard().getturn() == 1) currnode->getstate().winscoreincr(result);
			else if(-1*currnode->getstate().getboard().getturn() == -1) currnode->getstate().winscoreincr(1-result);
			currnode = currnode->getparent();
		} while(currnode != NULL);
	}

	int getbestmove(){
		// it is choosen by most visited node
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

	int getsampledmove(){
		// it is choosen randomly but proportionaly to number of visits to the node 
		float randomvar = rand()%RAND_MAX;
		float probsum = 0;
		for(int i = 0; i < (int)root->getchildren().size(); i++){
			float proportion = (float)root->getchildren()[i]->getstate().getvisitcount()/root->getstate().getvisitcount();
			if(probsum+proportion >= randomvar)
				return i;
		}
		return root->getchildren().size()-1;
	}

	Matrix<float> retnewpolicy(){
		Matrix<float> tmp(std::vector<float>(7));
		for(int i = 0; i < (int) root->getchildren().size(); i++){
			tmp(root->getchildren()[i]->getmove()) = (float)root->getchildren()[i]->getstate().getvisitcount()/root->getstate().getvisitcount();
		}
		return tmp;
	}
	std::vector<float> retnetform(){
		return root->getstate().getboard().getnetform();
	}

	void print_stats(){
		for(int i = 0; i < (int)root->getchildren().size(); i++){
			std::cout << "move, visitcount, winscore: ";
			std::cout << root->getchildren()[i]->getmove() << " ";
			std::cout << root->getchildren()[i]->getstate().getvisitcount() << " ";
			std::cout << root->getchildren()[i]->getstate().getwinscore() << std::endl;
		}
	}
	//--------------------------------------------
	/* UCT part */
	//--------------------------------------------
	int choosemove(Node * currnode){
		float UCTscore = UCTvalue(currnode->getchildren()[0]);
		int index = 0;
		for(int i = 0; i < currnode->getchildren().size(); i++){
			float currUCTscore = UCTvalue(currnode->getchildren()[i]);
			if(currUCTscore > UCTscore){
				UCTscore = currUCTscore;
				index = i;
			}
		}
		return index;
	}

	float UCTvalue(Node * currnode){
		return (currnode->getstate().getwinscore() / currnode->getstate().getvisitcount()
				+ currnode->getpolicyprobability()*sqrt(2.0*currnode->getparent()->getstate().getvisitcount()) /currnode->getstate().getvisitcount());
	}
	void setpolicy (Node *currnode){
		Matrix<float> nodepolicy = policynet->prediction(currnode->getstate().getboard().getnetform());
		std::vector<int> illegalmoves = currnode->getstate().getboard().listofillmoves();
		for(int i = 0; i < (int) illegalmoves.size(); i++) nodepolicy(illegalmoves[i]) = 0;
		nodepolicy = nodepolicy*(1.0/sqrt(nodepolicy.dotproduct(nodepolicy)+1e-6));
		currnode->getstate().getpolicy() = nodepolicy.retvector();
	}

	void setpolicy(Node *currnode, int move){
		std::fill(currnode->getstate().getpolicy().begin(), currnode->getstate().getpolicy().end(), 0);
		currnode->getstate().getpolicy()[move] = 1;
		for(int i = 0; i < (int) currnode->getchildren().size(); i++){
			if(currnode->getchildren()[i]->getmove() == move)currnode->getchildren()[i]->setpolicyprobability(1);
			else currnode->getchildren()[i]->setpolicyprobability(0);
		}
	}

	void addnoisetopolicy(Node *rootnode = NULL){
		if(rootnode == NULL) rootnode = root;
		std::mt19937 gen(time(NULL));
		dirichlet_distribution<std::mt19937> d(std::vector<float>(7, 0.2));
		std::vector<float> noise = d(gen);
		for(int i = 0; i < 7; i++){
			rootnode->getstate().getpolicy()[i] = rootnode->getstate().getpolicy()[i] * 0.75 + noise[i] * 0.25;
		}
		//normalize vector
		if(rootnode->getstate().getboard().listofillmoves().size() != 0){
			Matrix<float> nodepolicy(rootnode->getstate().getpolicy());
			std::vector<int> illegalmoves = rootnode->getstate().getboard().listofillmoves();
			for(int i = 0; i < (int) illegalmoves.size(); i++) nodepolicy(illegalmoves[i]) = 0;
			nodepolicy = nodepolicy*(1.0/sqrt(nodepolicy.dotproduct(nodepolicy) +1e-6));
			rootnode->getstate().getpolicy() = nodepolicy.retvector();
		}

		for(int i = 0; i < (int)rootnode->getchildren().size(); i++){
			rootnode->getchildren()[i]->setpolicyprobability(rootnode->getstate().getpolicy()[rootnode->getchildren()[i]->getmove()]);
		}
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
		root->noparent();
	}
	void createSubTreeOfmove(int move){
		for(int i = 0; i < (int) root->getchildren().size(); i++){
			if(root->getchildren()[i]->getmove() == move) {
				createSubTree(i);
				return;
			}
		}
		Node *newnode = new Node(root, move, 0);
		setpolicy(newnode);
		root->getchildren().push_back(newnode);
		createSubTreeOfmove(move);

	}
	//--------------------------------------------

	~PVMCTSCF(){
		root->cleanup();
	}
};


#endif