// #include <algorithm>
// #include <boost/iterator/iterator_facade.hpp>
#include <cassert>
#include <cstdio>
#include <cctype>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

namespace ex15 {
void test()
{
//ex15
    std::vector<const char *> input = {"hello", "world"};
    std::vector<std::string> output(2);

    std::copy(input.begin(), input.end(), output.begin());

    assert(output[0] == "hello");
    assert(output[1] == "world");
//dex15
}
} // namespace ex15

using namespace ex15;

int main(){
    test();
    return 0;
}