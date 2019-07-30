#ifndef DUPLICATION_COUNT_TRACKER_H
#define DUPLICATION_COUNT_TRACKER_H

#include <vector>
#include <string>
#include <map>

typedef std::vector<std::map<std::string, std::map<std::string, double> > > dup_count_t;

class DuplicationCountTracker{
public:
    DuplicationCountTracker(){};

    void set_data(const dup_count_t &new_data){data = new_data;};
private:
    dup_count_t data;
};

#endif