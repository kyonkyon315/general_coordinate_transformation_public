#ifndef PACK_H
#define PACK_H
#include <tuple>
template<typename... Objects>
class Pack
{
private:
    const std::tuple<Objects...> objects;
public:
    Pack(const Objects& ...objects):
        objects(objects...)
    {}
    template<int I>
    const auto& get_object()const{
        return std::get<I>(objects);
    }
    static constexpr int get_num_objects(){
        return sizeof...(Objects);
    }
    template<int I>
    using element = typename std::tuple_element<I, std::tuple<Objects...>>::type;
};

#endif //PACK_H