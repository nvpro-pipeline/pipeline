// Copyright NVIDIA Corporation 2002-2005
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


#include <dp/sg/ui/Trackball.h>
#include <dp/math/math.h>
#include <dp/math/Vecnt.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace ui
    {

      Trackball::Trackball(void)
      : m_tbsize(0.8f)
      {
      }

      Trackball::~Trackball(void)
      {
      }

      void Trackball::apply(const Vec2f &p0, const Vec2f &p1, Vec3f & axis, float & rad)
      {
        // determine the z coordinate on the sphere
        Vec3f pTB0(p0[0], p0[1], projectOntoTBSphere(p0));
        Vec3f pTB1(p1[0], p1[1], projectOntoTBSphere(p1));
  
        // calculate the rotation axis via cross product between p0 and p1 
        axis = pTB0^pTB1;
        axis.normalize();
    
        // calculate the angle 
        float t = distance( pTB0, pTB1 ) / (2.f * m_tbsize);
  
        // clamp between -1 and 1
        if (t > 1.0) 
          t = 1.0;
        else if (t < -1.0) 
          t = -1.0;
  
        rad = (float)(2.0 * asin(t));
      }

      float Trackball::projectOntoTBSphere(const Vec2f & p)
      {
        float z;
        float d = length( p );
        if (d < m_tbsize * 0.70710678118654752440)
        {
          // inside sphere
          z = (float)sqrt(m_tbsize * m_tbsize - d * d);
        }
        else
        {
          // on hyperbola 
          float t = m_tbsize / 1.41421356237309504880f;
          z = t*t / d;
        }
 
        return z;
      }

    } // namespace ui
  } // namespace sg
} // namespace dp
