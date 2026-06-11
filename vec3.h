#ifndef VEC3_H
#define VEC3_H
template<typename T>
class Vec3{
public:
    T x,y,z;
    void operator+=(const Vec3& r){
        x+=r.x;
        y+=r.y;
        z+=r.z;
    }
};

#endif //VEC3_H
