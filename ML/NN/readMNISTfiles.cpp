#ifndef readMNISTfiles_H
#define readMNISTfiles_H

#include <iostream>
#include <fstream>
#include <vector>

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
std::vector<std::vector<double>> read_mnist_images(std::string path)
{
    std::vector<std::vector<unsigned char>> ret;   
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            std::vector<unsigned char> tmp;
            for(int r=0;r<n_rows;++r)
            {
                if(i == 0)std::cout << std::endl;
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    if(i == 0)std::cout << ((int)temp ? '#' : ' ') << " ";
                    tmp.push_back(temp);
                }
            }
            ret.push_back(tmp);
        }
    } else std::cout << "file couldn't open" << std::endl;
    file.close();
    std::vector<std::vector<double>> input(ret.size());
    for(int i = 0; i < (int) ret.size(); i++) input[i] = std::vector<double>(ret[i].begin(), ret[i].end());
    for(int i = 0; i < (int) input.size(); i++)
        for(int a = 0; a < (int) input[0].size(); a++) input[i][a] /= 255;
    return input;
}

std::vector<Matrix<double>> read_mnist_labels(std::string path)
{
    std::vector<unsigned char> ret;   
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_labels=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= reverseInt(number_of_labels);
        for(int i=0;i<number_of_labels;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            if(i == 0)std::cout << (int)temp << std::endl;
            ret.push_back(temp);
        }
    } else std::cout << "file couldn't open" << std::endl;
    file.close();
    std::vector<Matrix<double>> tmp(ret.size(), Matrix<double>(10, 1));
    for(int i = 0; i <(int) ret.size(); i++){
        tmp[i](ret[i]) = 1;
    }
    return tmp;
}

#endif