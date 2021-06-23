#include <iostream>
#include <ctime>
#include "AZCF.cpp"

using namespace std;
std::vector<int> simulaterandomgame(){
	CFBoard board;
	std::vector<int> gamelog;
	while(!board.if_win()){
		std::vector<int> possiblemoves = board.listofmoves();
		int x = possiblemoves[rand()%possiblemoves.size()];
		board.move(x);
		gamelog.push_back(x);
	}
	return gamelog;
}

std::string retfilename(int iternum, int vernum, bool ifpolicy){
	std::string retstring;
	if(ifpolicy)retstring = ".\\weights\\AZver"+ std::to_string(vernum) + "\\AZver" + std::to_string(vernum)+ "_policy" +std::to_string(iternum)+ ".txt";
	else retstring = ".\\weights\\AZver" +std::to_string(vernum) + "\\AZver" + std::to_string(vernum)+ "_value" +std::to_string(iternum)+ ".txt";
	return retstring;
}

// retfilename(iternum, vernum, 1), retfilename(iternum, vernum, 0)
int main(){
	srand(time(NULL));
	int iternum = 10;
	int vernum = 6;
	std::string vernumstr = std::to_string(vernum);
	// AlphaZeroCF alphazero;
	// alphazero.training(20, "AZver"+vernumstr+"\\AZver"+vernumstr+"_");
	AlphaZeroCF alphazero(retfilename(iternum, vernum, 1), retfilename(iternum, vernum, 0));
	for(int j = 0; j <= iternum; j+=2){
		std::vector<std::vector<int>> games;
		for(int i = 0; i < 100; i++) {
			std::cout << i << " ";
			games.push_back(alphazero.simulategame(retfilename(j, vernum, 1), retfilename(j, vernum, 0)));
		}
		std::cout << std::endl;
		std::cout << "test between: " << j << "-" << iternum << std::endl;
		stats(games);
	}
	
	// stats(games);

	//stats
	// std::vector<std::vector<int>> randomgames;
	// for(int i = 0; i < 1000; i++)
	// 	randomgames.push_back(simulaterandomgame());
	// stats(randomgames);
	// alphazero.playgame();

	return 0;
}