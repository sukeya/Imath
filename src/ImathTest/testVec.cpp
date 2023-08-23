//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "testVec.h"
#include "testVecCUDA.h"
#include <ImathFun.h>
#include <ImathVec.h>
#include <ImathRandom.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <oneapi/tbb.h>

// Include ImathForward *after* other headers to validate forward declarations
#include <ImathForward.h>

using namespace std;
using namespace IMATH_INTERNAL_NAMESPACE;

namespace
{

template <class T>
void
testLength2T ()
{
    const T s = std::sqrt (std::numeric_limits<T>::min ());
    const T e = 4 * std::numeric_limits<T>::epsilon ();

    Vec2<T> v;

    v = Vec2<T> (0, 0);
    assert (v.length () == 0);
    assert (v.normalized ().length () == 0);

    v = Vec2<T> (3, 4);
    assert (v.length () == 5);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    v = Vec2<T> (3000, 4000);
    assert (v.length () == 5000);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    T t = s * (1 << 4);

    v = Vec2<T> (t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec2<T> (0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec2<T> (-t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (
        v.length (), t * std::sqrt (2), t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    t = s / (1 << 4);

    v = Vec2<T> (t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec2<T> (0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec2<T> (-t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (
        v.length (), t * std::sqrt (2), t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    t = s / (1 << 20);

    v = Vec2<T> (t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec2<T> (0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec2<T> (-t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (
        v.length (), t * std::sqrt (2), t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
}

template <class T>
void
testLength3T ()
{
    const T s = std::sqrt (std::numeric_limits<T>::min ());
    const T e = 4 * std::numeric_limits<T>::epsilon ();

    Vec3<T> v;

    v = Vec3<T> (0, 0, 0);
    assert (v.length () == 0);
    assert (v.normalized ().length () == 0);

    v = Vec3<T> (3, 4, 0);
    assert (v.length () == 5);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    v = Vec3<T> (3000, 4000, 0);
    assert (v.length () == 5000);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    v = Vec3<T> (1, -1, 1);
    assert (
        IMATH_INTERNAL_NAMESPACE::equal (v.length (), 1 * std::sqrt (3), e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    v = Vec3<T> (1000, -1000, 1000);
    assert (IMATH_INTERNAL_NAMESPACE::equal (
        v.length (), 1000 * std::sqrt (3), 1000 * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    T t = s * (1 << 4);

    v = Vec3<T> (t, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (0, t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (0, 0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (-t, -t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (
        v.length (), t * std::sqrt (3), t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    t = s / (1 << 4);

    v = Vec3<T> (t, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (0, t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (0, 0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (-t, -t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (
        v.length (), t * std::sqrt (3), t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    t = s / (1 << 20);

    v = Vec3<T> (t, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (0, t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (0, 0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec3<T> (-t, -t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (
        v.length (), t * std::sqrt (3), t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
}

template <class T>
void
testLength4T ()
{
    const T s = std::sqrt (std::numeric_limits<T>::min ());
    const T e = 4 * std::numeric_limits<T>::epsilon ();

    Vec4<T> v;

    v = Vec4<T> (0, 0, 0, 0);
    assert (v.length () == 0);
    assert (v.normalized ().length () == 0);

    v = Vec4<T> (3, 4, 0, 0);
    assert (v.length () == 5);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    v = Vec4<T> (3000, 4000, 0, 0);
    assert (v.length () == 5000);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    v = Vec4<T> (1, -1, 1, 1);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), 2, e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    v = Vec4<T> (1000, -1000, 1000, 1000);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), 2000, 1000 * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    T t = s * (1 << 4);

    v = Vec4<T> (t, 0, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, t, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, 0, t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, 0, 0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (-t, -t, -t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t * 2, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    t = s / (1 << 4);

    v = Vec4<T> (t, 0, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, t, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, 0, t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, 0, 0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (-t, -t, -t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t * 2, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));

    t = s / (1 << 20);

    v = Vec4<T> (t, 0, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, t, 0, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, 0, t, 0);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (0, 0, 0, t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
    v = Vec4<T> (-t, -t, -t, -t);
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.length (), t * 2, t * e));
    assert (IMATH_INTERNAL_NAMESPACE::equal (v.normalized ().length (), 1, e));
}

void testVecCUDAPerfomance()
{
    using std::chrono::system_clock;
    auto times = 10;

    std::size_t num = 1e6;
    auto ins = std::vector<HostPolyMesh>();
    ins.reserve(num);
    auto out = HostPolyMesh{
        typename HostPolyMesh::Points(num, typename HostPolyMesh::Point()),
        typename HostPolyMesh::Normals(num, typename HostPolyMesh::Normal())
    };
    auto mats = std::vector<Matrix44<float>>();
    mats.reserve(num);

    for (std::size_t time = 0; time < times; ++time)
    {
        auto in = HostPolyMesh();
        auto rand = Rand48(time);
        for (std::size_t i = 0; i < num; ++i)
        {
            in.points.push_back(hollowSphereRand<typename HostPolyMesh::Point>(rand));
        }
        in.normals = in.points;
        ins.push_back(in);

        auto mat = Matrix44<float>();
        for (std::size_t i = 0; i < 4; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                mat.x[i][j] = Imath::drand48();
            }
        }
        mats.push_back(mat);
    }


    system_clock::time_point start = system_clock::now();
    for (std::size_t time = 0; time < times; ++time)
    {
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<std::size_t>(0, num),
            [&mat = mats.at(time), &in = ins.at(time), &out](const oneapi::tbb::blocked_range<std::size_t>& range)
            {
                for(auto r = range.begin(); r != range.end(); ++r)
                {
                    out.points[r] = in.points[r] * mat;
                }
            }
        );
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<std::size_t>(0, num),
            [&mat = mats.at(time), &in = ins.at(time), &out](const oneapi::tbb::blocked_range<std::size_t>& range)
            {
                for(auto r = range.begin(); r != range.end(); ++r)
                {
                    out.normals[r] = in.normals[r] * mat;
                }
            }
        );
    }
    system_clock::time_point end = system_clock::now();

    std::cout << "times: " << times << '\n';
    std::cout << "size of vector: " << num << '\n';
    std::cout << "total time[milliseconds]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';
}

} // namespace

void
testVec ()
{
    cout << "Testing some basic vector operations" << endl;

    /*
    testLength2T<float> ();
    testLength2T<double> ();
    testLength3T<float> ();
    testLength3T<double> ();
    testLength4T<float> ();
    testLength4T<double> ();
    */
    testVecCUDAPerfomance();

    cout << "ok\n" << endl;
}
