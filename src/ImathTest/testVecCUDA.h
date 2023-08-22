//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//
#include <ImathVec.h>
#include <ImathMatrix.h>
#include <thrust/host_vector.h>

template <class PointType, class NormalType, template<class> class ContainerType>
struct PolyMesh
{
  template <class T>
  using Container = ContainerType<T>;
  using Point = PointType;
  using Normal = NormalType;

  using Points = Container<Point>;
  using Normals = Container<Normal>;

  Points points;
  Normals normals;
};

template <template<class> class C>
using ImathPolyMesh = PolyMesh<Imath::Vec3<float>, Imath::Vec3<float>, C>;

using HostPolyMesh = ImathPolyMesh<thrust::host_vector>;


void testVecCUDA(
  const HostPolyMesh& host_poly_mesh,
  const Imath::Matrix44<float>& xform_matrix,
  HostPolyMesh& xformed_poly_mesh
);