#include <iostream>
#include "mctsttt.cpp"

using namespace std;
int main(){
	// std::vector<std::vector<std::pair<int, int>>> gamesMCTSMinimax;
	// for(int i = 0; i < 10; i++){
	// 	gamesMCTSMinimax.push_back(simulategame());
	// }
	// stats(gamesMCTSMinimax);
	ttt board;
	while(!board.end()){
		int x, y;
		MonteCarloTreeSearch mcts(board);
		cout << "hello2" << endl;
		std::pair<int, int> machinemove = mcts.runMCTSinTime(1000);
		board.move(machinemove.first, machinemove.second);
		std::cout << "hello3" << std::endl;
		std::cout << "tree size: " << mcts.wholetreesize() << std::endl;
	
		board.print_ttt();
		if(board.end()) break;
		std::cin >> y >> x;
		
		board.move(y, x);
		board.print_ttt();
	}
 	return 0;
} 