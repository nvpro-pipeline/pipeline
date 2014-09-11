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


#include <dp/sg/core/PerspectiveCamera.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( PerspectiveCamera, FieldOfView );

      BEGIN_REFLECTION_INFO( PerspectiveCamera )
        DERIVE_STATIC_PROPERTIES( PerspectiveCamera, Camera );
        INIT_STATIC_PROPERTY_RW( PerspectiveCamera, FieldOfView,  float,    SEMANTIC_VALUE,   value, value );
      END_REFLECTION_INFO

      PerspectiveCameraSharedPtr PerspectiveCamera::create()
      {
        return( std::shared_ptr<PerspectiveCamera>( new PerspectiveCamera() ) );
      }

      HandledObjectSharedPtr PerspectiveCamera::clone() const
      {
        return( std::shared_ptr<PerspectiveCamera>( new PerspectiveCamera( *this ) ) );
      }

      PerspectiveCamera::PerspectiveCamera(void)
      {
        m_objectCode = OC_PERSPECTIVECAMERA;
      }

      PerspectiveCamera::PerspectiveCamera( const PerspectiveCamera &rhs )
      : FrustumCamera(rhs)
      {
        m_objectCode = OC_PERSPECTIVECAMERA;
      }

      PerspectiveCamera::~PerspectiveCamera(void)
      {
      }

      void  PerspectiveCamera::setFieldOfView( float fov )
      {
        float currentAspectRatio = getWindowSize()[0] / getWindowSize()[1];
        float y = 2.0f * getFocusDistance() * tanf( 0.5f * fov );
        float x = currentAspectRatio * y;
        setWindowSize( Vec2f( x, y ) ); // this updates the incarnation counter
        notify( PropertyEvent( this, PID_FieldOfView ) );
      }

      Mat44f  PerspectiveCamera::getProjection( void ) const
      {
        float scale = getNearDistance() / getFocusDistance();
        float l = scale * ( getWindowOffset()[0] - getWindowSize()[0] / 2.0f );
        float b = scale * ( getWindowOffset()[1] - getWindowSize()[1] / 2.0f );
        //  adjust the l/r/b/t values to the window region to view
        const Box2f region = getWindowRegion();
        float r = l + scale * region.getUpper()[0] * getWindowSize()[0];
        l += scale * region.getLower()[0] * getWindowSize()[0];
        float t = b + scale * region.getUpper()[1] * getWindowSize()[1];
        b += scale * region.getLower()[1] * getWindowSize()[1];
        float n = getNearDistance();
        float f = getFarDistance();

        // pre-calculate frequently used terms
        float _2n  = 2.0f*n; // 2 times zNear
        float _2nf = _2n*f; // 2 times zNear times zFar
        float rml  = r-l;   // right minus left
        float rpl  = r+l;   // right plus left
        float tmb  = t-b;   // top minus bottom
        float tpb  = t+b;   // top plus bottom
        float fmn  = f-n;   // zFar minus zNear
        float fpn  = f+n;   // zFar plus zNear

        // projection matrix
        return( Mat44f( util::makeArray( _2n/rml,    0.0f,      0.0f,  0.0f
                                       ,    0.0f, _2n/tmb,      0.0f,  0.0f
                                       , rpl/rml, tpb/tmb, - fpn/fmn, -1.0f
                                       ,    0.0f,    0.0f, -_2nf/fmn,  0.0f ) ) );
      }

      Mat44f  PerspectiveCamera::getInverseProjection( void ) const
      {
        float scale = getNearDistance() / getFocusDistance();
        float l = scale * ( getWindowOffset()[0] - getWindowSize()[0] / 2.0f );
        float b = scale * ( getWindowOffset()[1] - getWindowSize()[1] / 2.0f );
        //  adjust the l/r/b/t values to the window region to view
        const Box2f & region = getWindowRegion();
        float r = l + scale * region.getUpper()[0] * getWindowSize()[0];
        l += scale * region.getLower()[0] * getWindowSize()[0];
        float t = b + scale * region.getUpper()[1] * getWindowSize()[1];
        b += scale * region.getLower()[1] * getWindowSize()[1];
        float n = getNearDistance();
        float f = getFarDistance();

        // pre-calculate frequently used terms
        float _2n  = 2.0f*n; // 2 times zNear
        float _2nf = _2n*f; // 2 times zNear times zFar
        float rml  = r-l;   // right minus left
        float rpl  = r+l;   // right plus left
        float tmb  = t-b;   // top minus bottom
        float tpb  = t+b;   // top plus bottom
        float fmn  = f-n;   // zFar minus zNear
        float fpn  = f+n;   // zFar plus zNear

        return( Mat44f( util::makeArray( rml/_2n,    0.0f,  0.0f,      0.0f
                                       ,    0.0f, tmb/_2n,  0.0f,      0.0f
                                       ,    0.0f,    0.0f,  0.0f, -fmn/_2nf
                                       , rpl/_2n, tpb/_2n, -1.0f,  fpn/_2nf ) ) );
      }

      void  PerspectiveCamera::setFocusDistance( float td )
      {
        // changed focus distance should keep fieldOfView constant, so we have to adjust the window size here
        DP_ASSERT( ( td > FLT_EPSILON ) && ( getFocusDistance() > FLT_EPSILON ) );

        setWindowSize( getWindowSize() * td / getFocusDistance() );
        setWindowOffset( getWindowOffset() * td / getFocusDistance() );
        FrustumCamera::setFocusDistance( td );
      }

      void  PerspectiveCamera::getPickRay(int x, int y, int w, int h, Vec3f& rayOrigin, Vec3f& rayDir) const
      {
        // note that getDirection and getUpVector do not return a reference but a copy of a Vec3f
        Vec3f camDir(getDirection());
        Vec3f upVec(getUpVector());

        rayOrigin = getPosition();
        rayDir = getFocusDistance() * camDir
               + getWindowSize()[0] * (x - 0.5f * w) / w * (camDir ^ upVec)
               + getWindowSize()[1] * (y - 0.5f * h) / h * upVec;
        rayDir.normalize();
      }

      PerspectiveCamera & PerspectiveCamera::operator=(const PerspectiveCamera & rhs)
      {
        if (&rhs != this)
        {
          FrustumCamera::operator=(rhs);
        }
        return *this;
      }

      CullCode PerspectiveCamera::determineCullCode( const Sphere3f &sphere ) const
      {
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

        //  Each of the four side planes are determined by two of the window vertices and the origin
        //  (apex of the frustum. The normals of the right and left side are in the x-z-plane; the
        //  normals of the top and bottom side are in the y-z-plane.
        //  With t the target distance, sx the window size in x, we get d = sqrt( t*t + 0.25*sx*sx ),
        //  the distance from the camera to the right border of the window. With alpha horizontal
        //  field of view, we get sinAlpha = 0.5 * sx / d, cosAlpha = t / d.
        //  We also have to take the window offset and window region into account (e.g. in a cluster). 

        //  front plane: n = 0,0,-1, p = 0,0,-near => n*p + d = near + d = 0 => d = -near
        //    => cf = n * c + d = -center.z - near
        //  back plane : n = 0,0,+1, p = 0,0,-far  => n*p + d = -far + d = 0 => d = far
        //    => cb = n * c + d =  center.z + far

        //  Determine the distances from the camera to those points.
        float tSquare = square( getFocusDistance() );
        float dl = sqrt( tSquare + square( hl ) );
        float dr = sqrt( tSquare + square( hr ) );

        //  Determine sine and cosine of the vector from the camera along the border of the frustum.
        float sal = hl / dl;
        float cal = getFocusDistance() / dl;
        float sar = hr / dr;
        float car = getFocusDistance() / dr;

        //  Now the vectors from the camera to the left and right borders are
        //  vl = [sal,0,-cal]    => nl = [cal,0,sal]
        //  vr = [sar,0,-car]    => nr = [-car,0,-sar]
        //  left plane: nl = [-cal,0,sal], p = 0,0,0 => n*p + d = 0 + d = 0 => d = 0
        //    => cl = nl * c + d = cal * center.x + sal * center.z
        //  right plane: nr = [-car,0,-sar], p = 0,0,0 => n*p + d = 0 + d = 0 => d = 0
        //    => cr = nr * c + d = -car * center.x - sar * center.z
        float cl =   cal * sphere.getCenter()[0] + sal * sphere.getCenter()[2];
        float cr = - car * sphere.getCenter()[0] - sar * sphere.getCenter()[2];

        //  Determine the distances from the camera to those points.
        float db = sqrt( tSquare + square( hb ) );
        float dt = sqrt( tSquare + square( ht ) );

        //  Determine sine and cosine of the vector from the camera along the border of the frustum.
        float sab = hb / db;
        float cab = getFocusDistance() / db;
        float sat = ht / dt;
        float cat = getFocusDistance() / dt;

        //  Now the vectors from the camera to the bottom and top borders are
        //  vb = [0,sab,-cab]    => nb = [0,cab,sab]
        //  vt = [0,sat,-cat]    => nt = [0,-cat,-sat]
        //  bottom plane: nb = [0,cab,sab], ... => d = 0
        //    => cb = nb * c + d = cab * center.y + sab * center.z
        //  top plane: nt = [0,-cat,-sat], ... => d = 0
        //    => ct = nt * c + d = -cat * center.y - sat * center.z
        float cb =   cab * sphere.getCenter()[1] + sab * sphere.getCenter()[2];
        float ct = - cat * sphere.getCenter()[1] - sat * sphere.getCenter()[2];

        CullCode  cc;
        if (    ( - sphere.getCenter()[2] - getNearDistance() + sphere.getRadius() < 0.0f )
            ||  (   sphere.getCenter()[2] + getFarDistance()  + sphere.getRadius() < 0.0f )
            ||  ( cl + sphere.getRadius() < 0.0f ) || ( cr + sphere.getRadius() < 0.0f )
            ||  ( cb + sphere.getRadius() < 0.0f ) || ( ct + sphere.getRadius() < 0.0f ) )
        {
          cc = CC_OUT;
        }
        else if (     ( 0.0f < - sphere.getCenter()[2] - getNearDistance() - sphere.getRadius() )
                  &&  ( 0.0f <   sphere.getCenter()[2] + getFarDistance()  - sphere.getRadius() )
                  &&  ( 0.0f < cl - sphere.getRadius() ) && ( 0.0f < cr - sphere.getRadius() )
                  &&  ( 0.0f < cb - sphere.getRadius() ) && ( 0.0f < ct - sphere.getRadius() ) )
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
