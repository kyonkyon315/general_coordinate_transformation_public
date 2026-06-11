#ifndef BIN_SAVER_H
#define BIN_SAVER_H

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

class BinSaver{
private:
    std::ofstream ofs;

public:
    BinSaver(){
    }
    BinSaver(const std::string& filename){
        ofs.open(filename, std::ios::binary);
        if(!ofs){
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }

    void open(const std::string& filename){
        ofs.open(filename, std::ios::binary);
        if(!ofs){
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }

    // 基本型
    template<typename T>
    BinSaver& write(const T& val){
        static_assert(std::is_trivially_copyable<T>::value);
        ofs.write(reinterpret_cast<const char*>(&val), sizeof(T));
        return *this;
    }

    // vector専用
    template<typename T>
    BinSaver& write_vec(const std::vector<T>& val){
        static_assert(std::is_trivially_copyable<T>::value);
        if(!val.empty()){
            ofs.write(reinterpret_cast<const char*>(val.data()), sizeof(T)*val.size());
        }
        return *this;
    }

    void flush(){
        ofs.flush();
    }
};

#endif // BIN_SAVER_H