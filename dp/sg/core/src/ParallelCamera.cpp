// Copyright NVIDIA Corporation 2002-2011
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


#include <dp/sg/core/ParallelCamera.h>

// enable memory leak detection

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      BEGIN_REFLECTION_INFO( ParallelCamera )
        DERIVE_STATIC_PROPERTIES( ParallelCamera, FrustumCamera );
      END_REFLECTION_INFO

      ParallelCameraSharedPtr ParallelCamera::create()
      {
        return( std::shared_ptr<ParallelCamera>( new ParallelCamera() ) );
      }

      HandledObjectSharedPtr ParallelCamera::clone() const
      {
        return( std::shared_ptr<ParallelCamera>( new ParallelCamera( *this ) ) );
      }

      ParallelCamera::ParallelCamera(void)
      {
        m_objectCode = OC_PARALLELCAMERA;
      }

      ParallelCamera::ParallelCamera( const ParallelCamera &rhs )
      : FrustumCamera(rhs)
      {
        m_objectCode = OC_PARALLELCAMERA;
      }

      ParallelCamera::~ParallelCamera(void)
      {
      }

      Mat44f  ParallelCamera::getProjection( void ) const
      {
        float l = getWindowOffset()[0] - getWindowSize()[0] / 2.0f;
        float b = getWindowOffset()[1] - getWindowSize()[1] / 2.0f;
        //  adjust the l/r/b/t values to the window region to view
        const Box2f & region = getWindowRegion();
        float r = l + region.getUpper()[0] * getWindowSize()[0];
        l += region.getLower()[0] * getWindowSize()[0];
        float t = b + region.getUpper()[1] * getWindowSize()[1];
        b += region.getLower()[1] * getWindowSize()[1];
        float n = getNearDistance();
        float f = getFarDistance();

        // pre-calculate frequently used terms
        float rml  = r-l;   // right minus left
        float rpl  = r+l;   // right plus left
        float tmb  = t-b;   // top minus bottom
        float tpb  = t+b;   // top plus bottom
        float fmn  = f-n;   // zFar minus zNear
        float fpn  = f+n;   // zFar plus zNear

        return( Mat44f( util::makeArray( 2.0f/rml,     0.0f,      0.0f, 0.0f
                                       ,     0.0f, 2.0f/tmb,      0.0f, 0.0f
                                       ,     0.0f,     0.0f, -2.0f/fmn, 0.0f
                                       , -rpl/rml, -tpb/tmb,  -fpn/fmn, 1.0f ) ) );
      }

      Mat44f  ParallelCamera::getInverseProjection( void ) const
      {
        float l = getWindowOffset()[0] - getWindowSize()[0] / 2.0f;
        float b = getWindowOffset()[1] - getWindowSize()[1] / 2.0f;
        //  adjust the l/r/b/t values to the window region to view
        const Box2f & region = getWindowRegion();
        float r = l + region.getUpper()[0] * getWindowSize()[0];
        l += region.getLower()[0] * getWindowSize()[0];
        float t = b + region.getUpper()[1] * getWindowSize()[1];
        b += region.getLower()[1] * getWindowSize()[1];
        float n = getNearDistance();
        float f = getFarDistance();

        // pre-calculate frequently used terms
        float rml  = r-l;   // right minus left
        float rpl  = r+l;   // right plus left
        float tmb  = t-b;   // top minus bottom
        float tpb  = t+b;   // top plus bottom
        float fmn  = f-n;   // zFar minus zNear
        float fpn  = f+n;   // zFar plus zNear

        return( Mat44f( util::makeArray( rml*0.5f,     0.0f,      0.0f, 0.0f
                                       ,     0.0f, tmb*0.5f,      0.0f, 0.0f
                                       ,     0.0f,     0.0f, -fmn*0.5f, 0.0f
                                       , rpl*0.5f, tpb*0.5f,  fpn*0.5f, 1.0f ) ) );
      }

      void  ParallelCamera::getPickRay(int x, int y, int w, int h, Vec3f& rayOrigin, Vec3f& rayDir) const
      {
        Vec3f delta( getWindowSize()[0] * (x - 0.5f * w) / w * (getDirection() ^ getUpVector())
                   + getWindowSize()[1] * (y - 0.5f * h) / h * getUpVector() );

        rayOrigin = getPosition() + delta;
        rayDir = getDirection();
        rayDir.normalize();
      }

      ParallelCamera & ParallelCamera::operator=(const ParallelCamera & rhs)
      {
        if (&rhs != this)
        {
          FrustumCamera::operator=(rhs);
        }
        return *this;
      }

      CullCode ParallelCamera::determineCullCode( const Sphere3f &sphere ) const
      {
        DP_ASSERT( isValid(sphere) );
        //  For each of the six frustum planes determine the signed distance of the center to the plane.
        //  Take the plane equation n*x + d = 0, with n the plane normal, x a point on the plane, and d
        //  the negative distance of the plane to the origin.
        //  With p an arbitrary point, we get n*p + d = c, with c the signed distance of p from the plane.
        //  c > 0   <=> p is 'above' the plane (the normal points from the plane to the point)
        //  c < 0   <=> p is 'below' the plane (the normal points from the point to the plane)
        //  Determine for each of the six frustum planes a plane equation with the normal pointing
        //  inward, meaning positive values of c are in, negative are out.
        //  Taking as the arbitrary point p the center of the bounding sphere, we get the following
        //  results (with r the radius of the bounding sphere):
        //  c + r < 0   <=> center is more than radius below the plane  <=> bounding sphere is completely outside
        //  c - r > 0   <=> center is more than radius above the plane  <=> bounding sphere is completely inside
        //  otherwise   <=> bounding sphere intersects the frustum at this plane.
        //  If the bounding sphere is completely outside relative to at least one plane, it is out.
        //  If the bounding sphere is completely inside relative to all planes, it is in.
        //  Otherwise it is partial.

        const Box2f & region = getWindowRegion();
        //  Determine the (signed) distances of the left and right window border from the target point.
        float hl = getWindowOffset()[0] + getWindowSize()[0] * ( region.getLower()[0] - 0.5f );
        float hr = getWindowOffset()[0] + getWindowSize()[0] * ( region.getUpper()[0] - 0.5f );
        //  Determine the (signed) distances of the bottom and top window border from the target point.
        float hb = getWindowOffset()[1] + getWindowSize()[1] * ( region.getLower()[1] - 0.5f );
        float ht = getWindowOffset()[1] + getWindowSize()[1] * ( region.getUpper()[1] - 0.5f );

        CullCode  cc;
        //  front and back plane are the same as with perspective camera
        //  left plane: n = 1,0,0, p = hl,0,0 => n * p + d = hl + d = 0 => d = -hl
        //    => cl = n * c + d =  center.x - hl
        //  right plane: n = -1,0,0, p = hr,0,0 => n*p + d = -hr + d = 0 => d = hr
        //    => cr = n * c + d = -center.x + hr
        //  bottom plane: n = 0,1,0, p = 0,hb,0 => n * p + d = hb + d = 0 => d = -hb
        //    => cb = n * c + d =  center.y - hb
        //  top plane: n = 0,-1,0, p = 0,ht,0 => n * p + d = -ht + d = 0 => d = ht
        //    => ct = n * c + d = -center.y + ht
        if (    ( - sphere.getCenter()[2] - getNearDistance() + sphere.getRadius() < 0.0f )
            ||  (   sphere.getCenter()[2] + getFarDistance()  + sphere.getRadius() < 0.0f )
            ||  (   sphere.getCenter()[0] - hl + sphere.getRadius() < 0.0f )
            ||  ( - sphere.getCenter()[0] + hr + sphere.getRadius() < 0.0f )
            ||  (   sphere.getCenter()[1] - hb + sphere.getRadius() < 0.0f )
            ||  ( - sphere.getCenter()[1] + ht + sphere.getRadius() < 0.0f ) )
        {
          cc = CC_OUT;
        }
        else if (     ( 0.0f < - sphere.getCenter()[2] - getNearDistance() - sphere.getRadius() )

                  &&  ( 0.0f <   sphere.getCenter()[2] + getFarDistance()  - sphere.getRadius() )

                  &&  ( 0.0f <   sphere.getCenter()[0] - hl - sphere.getRadius() )
                  &&  ( 0.0f < - sphere.getCenter()[0] + hr - sphere.getRadius() )
                  &&  ( 0.0f <   sphere.getCenter()[1] - hb - sphere.getRadius() )
                  &&  ( 0.0f < - sphere.getCenter()[1] + ht - sphere.getRadius() ) )
        {
          cc = CC_IN;
        }
        else
        {
          cc = CC_PART;
        }

        return( cc );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
