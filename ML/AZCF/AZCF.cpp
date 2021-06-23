#ifndef AZCF_H
#define AZCF_H

#include "pvmctscf.cpp"

#define TEST 0

bool if_file_exist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

class AlphaZeroCF{
	NN policynetwork;
	NN_MSE valuenetwork;
public:
	AlphaZeroCF()
	: policynetwork({43, 128, 256, 128, 64, 7}), valuenetwork({43, 128, 256, 128, 64, 1}){
		policynetwork.addActivation<ReLU>({1, 2, 3, 4});
		policynetwork.addActivation<Sigmoid>({5});
		valuenetwork.addActivation<ReLU>({1, 2, 3, 4});
		valuenetwork.addActivation<Sigmoid>({5});
		policynetwork.weightinit();
		valuenetwork.weightinit();

	}
	AlphaZeroCF(std::string pnfilename, std::string vnfilename)
	: policynetwork({43, 128, 256, 128, 64, 7}), valuenetwork({43, 128, 256, 128, 64, 1}){
		policynetwork.addActivation<ReLU>({1, 2, 3, 4});
		policynetwork.addActivation<Softmax>({5});
		valuenetwork.addActivation<ReLU>({1, 2, 3, 4});
		valuenetwork.addActivation<Sigmoid>({5});
		policynetwork.readNNfromfile(pnfilename);
		valuenetwork.readNNfromfile(vnfilename);
	}
	NN*     retpolicynet() { return &policynetwork; }
	NN_MSE* retvaluenet () { return &valuenetwork; } 

	void training(int policyiters, std::string filename){
		int numberofgames = 200, epochs = 30;
		int startnum = 0;
		while(if_file_exist(".\\weights\\"+filename+"policy"+std::to_string(startnum)+".txt")) startnum++;
		if(startnum == 0){
			policynetwork.writeNNtofile(".\\weights\\"+filename+"policy0.txt");
			valuenetwork.writeNNtofile(".\\weights\\"+filename+"value0.txt");
		} else {
			policynetwork.readNNfromfile(".\\weights\\"+filename+"policy"+std::to_string(startnum-1)+".txt");
			valuenetwork.readNNfromfile(".\\weights\\"+filename+"value"+std::to_string(startnum-1)+".txt");
		}
		for(int i = startnum; i < policyiters; i++){
			std::cout << "Starting policy iteration: " << i << std::endl;
			std::vector<std::vector<float>> training_games_states;
			std::vector<Matrix<float>> training_games_policyvectors, results;
			std::cout << "Starting generating games: " << std::endl;
			if(TEST && i == startnum) load_training_data(training_games_states, training_games_policyvectors, results, ".\\gamedata.txt");
			else generate_training_games(training_games_states, training_games_policyvectors, results, numberofgames);
			save_training_data(training_games_states, training_games_policyvectors, results, ".\\gamedata.txt");
			std::cout << "Starting training policy network: " << std::endl;
			policynetwork.trainNetwork(training_games_states, training_games_policyvectors, epochs);
			std::cout << "Starting training value  network: " << std::endl;
			valuenetwork.trainNetwork(training_games_states, results, epochs);	
			policynetwork.writeNNtofile(".\\weights\\"+filename+"policy"+std::to_string(i)+".txt");
			valuenetwork.writeNNtofile(".\\weights\\"+filename+"value"+std::to_string(i)+".txt");
		}

		 
	}
	void generate_training_games(std::vector<std::vector<float>> &states, std::vector<Matrix<float>> &policyvectors, std::vector<Matrix<float>> &results, int numberofgames){
		int mctsiters = 400;
		long long t = 0;
		for(int i = 0; i < numberofgames; i++){
			TIMERret timer(t);
			std::cout << "\rProgress: " << i << "/" << numberofgames << " time: " << t/1000.0 << " ms   ";
			t = 0;		
			CFBoard board;
			PVMCTSCF mctstree(&policynetwork, &valuenetwork);
			int gamelength = 0;
			while(!board.if_win()){
				gamelength++;
				int machinemove;
				mctstree.addnoisetopolicy();
				if(gamelength > 8)machinemove = mctstree.runMCTSforIters(mctsiters, 0);
				else machinemove = mctstree.runMCTSforIters(mctsiters, 1);
				states.push_back(mctstree.retnetform());
				policyvectors.push_back(mctstree.retnewpolicy());
				if(!board.move(machinemove)) std::cout << "mcts not working -> invalid move" << std::endl;
				mctstree.createSubTreeOfmove(machinemove);
			}
			float tmpres = board.if_win();
			if(tmpres == 2) tmpres = 0.5;
			else tmpres = (tmpres+1)/2;
			for(int j = 0; j < gamelength; j++){
				results.push_back(Matrix<float>({tmpres}));
			}
		}
		std::cout << std::endl;
	}
	
	void save_training_data(std::vector<std::vector<float>> &states, std::vector<Matrix<float>> &policyvectors, std::vector<Matrix<float>> &results, std::string path){
		std::ofstream file(path.c_str());
		int n = states.size();
		if(file.is_open()){
			file << n << " ";
			for(int i = 0; i < n; i++){
				for(int j = 0; j < 43; j++){
					 file << states[i][j] << " ";
				}
				for(int j = 0; j < 7; j++){
					 file << policyvectors[i](j) << " ";	
				}
				file << results[i](0) << " ";
			}
		} else {
			std::cout << "file: " << path << " couldn't open" << std::endl;
		}
		file.close();
	}
	void load_training_data(std::vector<std::vector<float>> &states, std::vector<Matrix<float>> &policyvectors, std::vector<Matrix<float>> &results, std::string path){
		std::ifstream file(path.c_str());
		int n; 
		file >> n;
		if(file.is_open()){
			for(int i = 0; i < n; i++){
				std::vector<float> tmpstate(43);
				std::vector<float> tmppolicyvector(7);
				float result;
				for(int j = 0; j < 43; j++){
					file >> tmpstate[j];
				}
				for(int j = 0; j < 7; j++){
					file >> tmppolicyvector[j];
				}
				file >> result;
				states.push_back(tmpstate);
				policyvectors.push_back(Matrix<float>(tmppolicyvector));
				results.push_back(Matrix<float>({result}));

			}
		} else {
			std::cout << "file: " << path << " couldn't open" << std::endl;
		}
		file.close();
	}

	std::vector<int> simulategame(std::string policy1, std::string value1, int order = 1){
		CFBoard board;
		AlphaZeroCF secondplayer(policy1, value1);
		int mctsiters = 200;
		PVMCTSCF player1(&policynetwork, &valuenetwork);
		PVMCTSCF player2(secondplayer.retpolicynet(), secondplayer.retvaluenet());
		player1.runMCTSforIters(mctsiters);
		player2.runMCTSforIters(mctsiters);
		std::vector<int> game;
		int machinemove;
		int movecounter = 0;
		while(!board.if_win()){
			movecounter++;
			if(order == 1){
				if(movecounter < 4) machinemove = player1.runMCTSforIters(mctsiters, 1);
				else machinemove = player1.runMCTSforIters(mctsiters);
				board.move(machinemove);
				game.push_back(machinemove);
				player1.createSubTreeOfmove(machinemove);
				player2.createSubTreeOfmove(machinemove);
				if(board.if_win()) break;
				if(movecounter < 4) machinemove = player2.runMCTSforIters(mctsiters, 1);
				else machinemove = player2.runMCTSforIters(mctsiters);
				board.move(machinemove);
				game.push_back(machinemove);
				player1.createSubTreeOfmove(machinemove);
				player2.createSubTreeOfmove(machinemove);
			} else {
				if(movecounter < 4) machinemove = player2.runMCTSforIters(mctsiters, 1);
				else machinemove = player2.runMCTSforIters(mctsiters);
				player1.createSubTreeOfmove(machinemove);
				player2.createSubTreeOfmove(machinemove);
				board.move(machinemove);
				game.push_back(machinemove);
				if(board.if_win()) break;
				if(movecounter < 4) machinemove = player1.runMCTSforIters(mctsiters, 1);
				else machinemove = player1.runMCTSforIters(mctsiters);
				board.move(machinemove);
				game.push_back(machinemove);
				player1.createSubTreeOfmove(machinemove);
				player2.createSubTreeOfmove(machinemove);
				
			}
		}
		return game;
	}
	
	void playgame(int selectplayer = 1){
		CFBoard board;
		int mctsiters = 400;
		PVMCTSCF mctstree(board, &policynetwork, &valuenetwork);
		while(!board.if_win()){
			if(selectplayer == 1){
				int x;
				std::cin >> x;
				board.move(x);
				mctstree.print_stats();
				std::cout << "root board" << std::endl;
				std::vector<float> moveprobs = mctstree.retroot()->getstate().getpolicy();
				for(int i = 0; i < (int) moveprobs.size(); i++){
					std::cout << moveprobs[i] << " ";
				}
				std::cout << std::endl;
				board.drawboard();
				mctstree.createSubTreeOfmove(x);
				if(board.if_win()) break;
				int machinemove = mctstree.runMCTSforIters(mctsiters);
				board.move(machinemove);
				mctstree.print_stats();
				board.drawboard();
				mctstree.createSubTreeOfmove(machinemove);
				mctstree.print_stats();
				
			} else {
				int machinemove = mctstree.runMCTSforIters(mctsiters);
				board.move(machinemove);
				board.drawboard();
				mctstree.createSubTreeOfmove(machinemove);
				if(board.if_win()) break;
				int x;
				std::cin >> x;
				board.move(x);
				board.drawboard();
			}

		}
	}

};

void stats(std::vector<std::vector<int>> &games){
	int results[] = {0,0,0};
	int n = (int)games.size();
	for(int i = 0; i < n; i++){
		CFBoard board;
		board.playgame(games[i]);
		results[std::abs(board.if_win()-1)]++;
	}
	std::cout << "winning: " << 100.0*results[0]/n << std::endl;
	std::cout << "drawing: " << 100.0*results[1]/n << std::endl;
	std::cout << "losing: "  << 100.0*results[2]/n << std::endl;
}


#endif