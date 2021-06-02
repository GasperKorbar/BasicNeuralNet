#ifndef timer_H
#define timer_H

#include <iostream>
#include <chrono>
#include <climits>
#include <ctime>

using namespace std::chrono;

class TIMER{
	high_resolution_clock::time_point start;
public:
	TIMER(){
		start = high_resolution_clock::now();
	}
	~TIMER(){
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop-start);	
		std::cout <<duration.count() << std::endl;
	}
};

class TIMERret{
	high_resolution_clock::time_point start;
	long long int &ret;
public:
	TIMERret(long long int &r) : ret(r){
		start = high_resolution_clock::now();
	}
	~TIMERret(){
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop-start);	
		ret += duration.count();		
	}
};

#endif