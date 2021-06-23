#ifndef CONNECTFOUR_H
#define CONNECTFOUR_H

#include <iostream>
#include <windows.h>
#include <vector>

class CFBoard{
    int board[6][7];
    int turn;
public:
    CFBoard(); 
    CFBoard(const CFBoard&);
    int getturn();
    bool move(int pos);
    int if_win();
    void drawboard();
    void playgame(std::vector<int>);
    CFBoard retmove(int);
    std::vector<float> getnetform();
    std::vector<int> listofmoves();
    std::vector<int> listofillmoves();
    void playrandommove();
};

CFBoard::CFBoard(){
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 7; j++)
            board[i][j] = 0;
    turn = 1;
}
CFBoard::CFBoard(const CFBoard& copyBoard){
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 7; j++)
            board[i][j] = copyBoard.board[i][j];
    turn = copyBoard.turn;
}

int CFBoard::getturn(){return turn;}
bool CFBoard::move(int pos){
    if(!(pos >= 0 && pos < 7)) return 0;
    if(board[0][pos] != 0) return 0;
    for(int i = 0; i < 6; i++){
        if(board[i][pos]!=0){
            board[i-1][pos] = turn;
            break;
        }
        else if(i == 5) board[i][pos] = turn;
    }
    turn=-turn;
    return 1;
}

int CFBoard::if_win(){
    bool ifwin = 1;
    for(int i = 0; i < 6; i++){
        for(int j = 0; j < 7; j++){
            int t = board[i][j];
            if(!t) continue;
            ifwin = 1;
            if(i <= 2){
                for(int k = 0; k < 4; k++){
                    if(board[i+k][j] != t){
                        ifwin = 0;
                    }
                }
                if(ifwin) return t;
                if(j <= 3){
                    ifwin = 1;
                    for(int k = 0; k < 4; k++){
                        if(board[i+k][j+k] != t){
                            ifwin = 0;
                        }
                    }
                    if(ifwin) return t;
                }
                if(j >= 3){
                    ifwin = 1;
                    for(int k = 0; k < 4; k++){
                        if(board[i+k][j-k] != t){
                            ifwin = 0;
                        }
                    }
                    if(ifwin) return t;
                }
            } else if(j <= 3){
                ifwin = 1;
                for(int k = 0; k < 4; k++){
                    if(board[i][j+k] != t){
                        ifwin = 0;
                    }
                }
                if(ifwin) return t;
            }
            }
    }
    for(int i = 0; i < 7; i++){
        if(board[0][i] == 0) return 0;
    }
    return 2;
}
std::vector<int> CFBoard::listofmoves(){
    std::vector<int> movelist;
    for(int i = 0; i < 7; i++){
        if(board[0][i] == 0) movelist.push_back(i);
    }
    return movelist;
}
std::vector<int> CFBoard::listofillmoves(){
    std::vector<int> movelist;
    for(int i = 0; i < 7; i++){
        if(board[0][i] != 0) movelist.push_back(i);
    }
    return movelist;
}

std::vector<float> CFBoard::getnetform(){
    std::vector<float>tmp(43);
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 7; j++)
            tmp[7*i+j] = board[i][j];
    tmp[42] = turn;
    return tmp;
}

void CFBoard::playrandommove(){
    std::vector<int> movelist = listofmoves();
    move(movelist[rand()%movelist.size()]);
}

void CFBoard::playgame(std::vector<int> game){
    for(int i = 0; i < (int) game.size(); i++){
        move(game[i]);
    }
}
CFBoard CFBoard::retmove(int m){
    CFBoard tmp(*this);
    tmp.move(m);
    return tmp;
}

void CFBoard::drawboard(){
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    for(int i = 0; i < 6; i++){
        std::cout << "+-------------+" << std::endl;
        for(int j = 0; j < 7; j++){
            std::cout << "|";
            if(board[i][j] == 1){SetConsoleTextAttribute(hConsole, 12); std::cout << "#";}
            else if(board[i][j] == -1) {SetConsoleTextAttribute(hConsole, 10); std::cout << "@";}
            else std::cout << " ";
            SetConsoleTextAttribute(hConsole, 7);
        }
        std::cout <<"|"<< std::endl;
    }
    std::cout << "+-------------+" << std::endl;
}

#endif