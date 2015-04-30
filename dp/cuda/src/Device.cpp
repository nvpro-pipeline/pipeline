// Copyright NVIDIA Corporation 2015
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <dp/cuda/Device.h>

namespace dp
{
  namespace cuda
  {
    DeviceSharedPtr Device::create()
    {
      return( std::shared_ptr<Device>( new Device() ) );
    }

    Device::Device()
    {
      CUDA_VERIFY( cudaGetDevice( &m_device ) );
      CUDA_VERIFY( cudaGetDeviceProperties( &m_properties, m_device ) );
    }

    Device::~Device( )
    {
      DP_ASSERT( isCurrent() );
      CUDA_VERIFY( cudaDeviceReset() );
    }

    dp::math::Vec3i Device::getMaxThreadsDim() const
    {
      DP_ASSERT( isCurrent() );
      return( dp::math::Vec3i( m_properties.maxThreadsDim[0], m_properties.maxThreadsDim[1], m_properties.maxThreadsDim[2] ) );
    }

    int Device::getDevice() const
    {
      DP_ASSERT( isCurrent() );
      return( m_device );
    }

    bool Device::isCurrent() const
    {
      int dev;
      CUDA_VERIFY( cudaGetDevice( &dev ) );
      return( dev == m_device );
    }

    void Device::synchronize()
    {
      DP_ASSERT( isCurrent() );
      cudaDeviceSynchronize();
    }

  } // namespace cuda
} // namespace dp
