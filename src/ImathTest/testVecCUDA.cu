//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//


#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "testVecCUDA.h"
#include <ImathVec.h>
#include <ImathMatrix.h>

/*
#include <thrust/async/transform.h>
#include <thrust/async/copy.h>
*/
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// Include ImathForward *after* other headers to validate forward declarations
#include <ImathForward.h>

using namespace IMATH_INTERNAL_NAMESPACE;

using DevicePolyMesh = ImathPolyMesh<thrust::device_vector>;


template <class T>
struct V_multiplies_M
{
  using Matrix = Imath::Matrix44<float>;

  explicit V_multiplies_M(const Matrix& m) : m_(m) {}

  V_multiplies_M() = default;
  V_multiplies_M(const V_multiplies_M&) = default;
  V_multiplies_M& operator=(const V_multiplies_M&) = default;
  V_multiplies_M(V_multiplies_M&&) = default;
  V_multiplies_M& operator=(V_multiplies_M&&) = default;
  ~V_multiplies_M() = default;

  __host__ __device__ constexpr T operator()(const T& p) const
  {
    return p * m_;
  }

private:
  Matrix m_;
};


/*
thrust::device_event testVecCUDA(
  const HostPolyMesh& host_poly_mesh,
  const Imath::Matrix44<float>& xform_matrix,
  HostPolyMesh& xformed_poly_mesh
)
{
  using Point = typename HostPolyMesh::Point;
  using Normal = typename HostPolyMesh::Normal;

  // Device-side PolyMesh storing a transformed PolyMesh.
  auto device_poly_mesh = DevicePolyMesh();

  auto copying_points_from_host_to_device = thrust::async::copy(
    thrust::host,
    thrust::device,
    host_poly_mesh.points.begin(),
    host_poly_mesh.points.end(),
    device_poly_mesh.points.begin()
  );
  auto copying_normals_from_host_to_device = thrust::async::copy(
    thrust::host,
    thrust::device,
    host_poly_mesh.normals.begin(),
    host_poly_mesh.normals.end(),
    device_poly_mesh.normals.begin()
  );

  auto transforming_points = thrust::async::transform
  (
    thrust::device.after(copying_points_from_host_to_device),
    device_poly_mesh.points.begin(),
    device_poly_mesh.points.end(),
    device_poly_mesh.points.begin(),
    V_multiplies_M<Point>(xform_matrix)
  );
  auto transforming_normals = thrust::async::transform
  (
    thrust::device.after(copying_normals_from_host_to_device),
    device_poly_mesh.normals.begin(),
    device_poly_mesh.normals.end(),
    device_poly_mesh.normals.begin(),
    V_multiplies_M<Normal>(xform_matrix)
  );

  // Set result.
  auto copying_points_from_device_to_host = thrust::async::copy(
    thrust::device.after(transforming_points),
    thrust::host,
    device_poly_mesh.points.begin(),
    device_poly_mesh.points.end(),
    xformed_poly_mesh.points.begin()
  );
  auto copying_normals_from_device_to_host = thrust::async::copy(
    thrust::device.after(transforming_normals),
    thrust::host,
    device_poly_mesh.normals.begin(),
    device_poly_mesh.normals.end(),
    xformed_poly_mesh.normals.begin()
  );

  return thrust::when_all(copying_points_from_device_to_host, copying_normals_from_device_to_host);
}
*/


void testVecCUDA(
  const HostPolyMesh& host_poly_mesh,
  const Imath::Matrix44<float>& xform_matrix,
  HostPolyMesh& xformed_poly_mesh
)
{
  using Point = typename HostPolyMesh::Point;
  using Normal = typename HostPolyMesh::Normal;

  // Device-side PolyMesh storing a transformed PolyMesh.
  auto device_poly_mesh = DevicePolyMesh{
    typename DevicePolyMesh::Points(host_poly_mesh.points.size(), typename DevicePolyMesh::Point()),
    typename DevicePolyMesh::Normals(host_poly_mesh.normals.size(), typename DevicePolyMesh::Normal())
  };
  
  thrust::copy(
    host_poly_mesh.points.cbegin(),
    host_poly_mesh.points.cend(),
    device_poly_mesh.points.begin()
  );
  thrust::copy(
    host_poly_mesh.normals.cbegin(),
    host_poly_mesh.normals.cend(),
    device_poly_mesh.normals.begin()
  );

  thrust::transform
  (
    device_poly_mesh.points.begin(),
    device_poly_mesh.points.end(),
    device_poly_mesh.points.begin(),
    V_multiplies_M<Point>(xform_matrix)
  );
  thrust::transform
  (
    device_poly_mesh.normals.begin(),
    device_poly_mesh.normals.end(),
    device_poly_mesh.normals.begin(),
    V_multiplies_M<Normal>(xform_matrix)
  );

  // Set result.
  thrust::copy(
    device_poly_mesh.points.cbegin(),
    device_poly_mesh.points.cend(),
    xformed_poly_mesh.points.begin()
  );
  thrust::copy(
    device_poly_mesh.normals.cbegin(),
    device_poly_mesh.normals.cend(),
    xformed_poly_mesh.normals.begin()
  );
}
