#ifndef ttt_H
#define ttt_H

#include <iostream>
#include <vector>
#include <utility>

class ttt{
	int board[3][3];
	int side;
public:
	ttt();
	ttt(const ttt&);
	ttt retmove(std::pair<int, int>);
	int end();
	int if_win();
	int getside();
	void move(int, int);
	void remove(int, int);
	void print_ttt();
	void playgame(std::vector<std::pair<int, int>>);
	char XorO(int);
	void playrandommove();
	std::vector<float> retboard();
	std::vector<std::pair<int, int>> listofmoves();
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

ttt ttt::retmove(std::pair<int, int> move){
	ttt tmp(*this);
	tmp.move(move.first, move.second);
	return tmp;
}

void ttt::playrandommove(){
	std::vector<std::pair<int, int>> possiblemoves = listofmoves();
	std::pair<int, int> m = possiblemoves[rand()%possiblemoves.size()];
	move(m.first, m.second);
}

int ttt::getside(){ return side; }

std::vector<float> ttt::retboard(){
	std::vector<float> tmp(9);
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
		std::cout << "+-+-+-+" << std::endl;
		for(int j = 0; j < 3; j++){
			std::cout << "|" << XorO(board[i][j]);
		}
		std::cout << "|" << std::endl;
	}
	std::cout << "+-+-+-+" << std::endl;
}
void ttt::playgame(std::vector<std::pair<int, int>> game){
	for(int i = 0; i < (int) game.size(); i++){
		move(game[i].first, game[i].second);
	}
}

std::vector<std::pair<int, int>> ttt::listofmoves(){
	std::vector<std::pair<int, int>> tmp;
	if(end()) return tmp;
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			if(board[i][j] == 0){
				tmp.push_back(std::make_pair(i, j));
			}
		}
	}
	return tmp;
}
void printlistofmoves(std::vector<std::pair<int,int>> l){
	for(int i = 0; i < (int) l.size(); i++){
		std::cout << "(" << l[i].first << ", " << l[i].second << ")";
		if(i != l.size()-1) std::cout << ", ";
	}
	std::cout << std::endl;
}

#endif
	