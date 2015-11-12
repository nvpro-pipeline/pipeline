// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/RayIntersectTraverser.h>

#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/ClipPlane.h>
#include <dp/sg/core/FrustumCamera.h>
#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/LOD.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/core/Switch.h>
#include <dp/sg/core/Transform.h>

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using std::string;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {
      const Vec3f RayIntersectTraverser::_RAY_ORIGIN_DEFAULT    = Vec3f(0.f, 0.f,  0.f);
      const Vec3f RayIntersectTraverser::_RAY_DIRECTION_DEFAULT = Vec3f(0.f, 0.f, -1.f);

      inline float doubleToFloat(double d)
      {
        // clamp to [-FLT_MAX, FLT_MAX] and cast to float
        return (float)std::min<double>(std::max<double>(d, -FLT_MAX), FLT_MAX);
      }

      // returns true if point lies inside the half-space defined by clipPlane
      inline bool isInsideHalfSpace(const ClipPlaneSharedPtr& clipPlane, const Vec3f& point)
      {
        return clipPlane->isInside(point);
      }

      RayIntersectTraverser::RayIntersectTraverser(void)
      : m_nearestIntIdx(0)
      , m_rayOrigin(_RAY_ORIGIN_DEFAULT)
      , m_rayDir(_RAY_DIRECTION_DEFAULT)
      , m_viewportHeight(0)
      , m_viewportWidth(0)
      , m_camClipping(true)
      {
        m_clipPlanes.push( vector<ClipPlaneSharedPtr>() );
        m_msRayOrigin.push( m_rayOrigin );
        m_msRayDir.push( m_rayDir );
        m_scaleFactors.push( 1.0f );
      }

      RayIntersectTraverser::~RayIntersectTraverser(void)
      {
        DP_ASSERT(m_curPath == NULL);
      }

      void RayIntersectTraverser::setRay( const Vec3f &origin, const Vec3f &dir)
      {
        DP_ASSERT(isNormalized(dir));

        m_rayOrigin = origin;
        m_rayDir    = dir;

        DP_ASSERT( ( m_msRayOrigin.size() == 1 ) && ( m_msRayDir.size() == 1 ) );
        m_msRayOrigin.pop();
        m_msRayDir.pop();
        m_msRayOrigin.push( m_rayOrigin );
        m_msRayDir.push( m_rayDir);
      }

      void RayIntersectTraverser::setViewportSize( unsigned int width, unsigned int height )
      {
        m_viewportWidth = width;
        m_viewportHeight = height;
      }

      void RayIntersectTraverser::doApply( const NodeSharedPtr & root )
      {
        DP_ASSERT( m_curPath == NULL );
        DP_ASSERT( m_viewState && "This traverser needs a valid ViewState. Use setViewState() prior calling apply()");
        DP_ASSERT( m_camera && "This traverser needs a valid camera as part of the ViewState" );
        DP_ASSERT( m_clipPlanes.size() == 1 );

        // requires camera available and locked by the framework already
        m_msCamDir.push( m_camera->getDirection() );

        // do the traversing
        if ( root )
        {
          m_currentHints.push_back(root->getHints());

          m_curPath = Path::create();
          SharedModelViewTraverser::doApply( root );
          m_curPath.reset();

          m_currentHints.clear();
        }

        m_msCamDir.pop();
        DP_ASSERT( m_msCamDir.size() == 0 );

        DP_ASSERT( m_clipPlanes.size() == 1 );
      }

      void RayIntersectTraverser::release()
      {
        // clean up the intersection map
        if (!m_intersectionList.empty())
        {
          m_intersectionList.erase(m_intersectionList.begin(), m_intersectionList.end());
        }

        // clean up the ray settings
        m_rayOrigin = _RAY_ORIGIN_DEFAULT;
        m_rayDir    = _RAY_DIRECTION_DEFAULT;

        // make sure that ....
        DP_ASSERT(m_curPath == NULL);
      }

      void  RayIntersectTraverser::handleBillboard( const Billboard * p )
      {
        m_curPath->push( p->getSharedPtr<Object>() );

        Sphere3f bs( p->getBoundingSphere() );

        int hints = m_currentHints.back() | p->getHints();
        m_currentHints.push_back(hints);

        if ( continueTraversal(hints, bs) )
        {
          SharedModelViewTraverser::handleBillboard(p);
        }

        m_currentHints.pop_back();

        m_curPath->pop();
      }

      static float getLineWidth( const dp::sg::core::PipelineDataSharedPtr & pipelineData )
      {
        if ( pipelineData )
        {
          const ParameterGroupDataSharedPtr & parameterGroupData = pipelineData->findParameterGroupData( string( "standardGeometryParameters" ) );
          if ( parameterGroupData )
          {
            return( parameterGroupData->getParameter<float>( parameterGroupData->getParameterGroupSpec()->findParameterSpec( "lineWidth" ) ) );
          }
        }
        return( 1.0f );
      }

      static float getPointSize( const dp::sg::core::PipelineDataSharedPtr & pipelineData )
      {
        if ( pipelineData )
        {
          const ParameterGroupDataSharedPtr & parameterGroupData = pipelineData->findParameterGroupData( string( "standardGeometryParameters" ) );
          if ( parameterGroupData )
          {
            return( parameterGroupData->getParameter<float>( parameterGroupData->getParameterGroupSpec()->findParameterSpec( "pointSize" ) ) );
          }
        }
        return( 1.0f );
      }

      void RayIntersectTraverser::handleGeoNode( const GeoNode *p )
      {
        m_curPath->push( p->getSharedPtr<Object>() );

        Sphere3f bs( p->getBoundingSphere() );

        int hints = m_currentHints.back() | p->getHints();
        m_currentHints.push_back(hints);

        if ( continueTraversal(hints, bs) )
        {
          m_currentLineWidth = getLineWidth( p->getMaterialPipeline() );
          m_currentPointSize = getPointSize( p->getMaterialPipeline() );
          SharedModelViewTraverser::handleGeoNode( p );
        }

        m_currentHints.pop_back();

        // clean up the stack (Don't forget!!!)
        m_curPath->pop();
      }

      void  RayIntersectTraverser::handleGroup( const Group * p )
      {
        m_curPath->push( p->getSharedPtr<Object>() );

        Sphere3f bs( p->getBoundingSphere() );

        int hints = m_currentHints.back() | p->getHints();
        m_currentHints.push_back(hints);

        if ( continueTraversal(hints, bs) )
        {
          SharedModelViewTraverser::handleGroup(p);
        }

        m_currentHints.pop_back();

        m_curPath->pop();
      }

      void  RayIntersectTraverser::handleLOD( const LOD * p )
      {
        m_curPath->push( p->getSharedPtr<Object>() );

        Sphere3f bs( p->getBoundingSphere() );

        int hints = m_currentHints.back() | p->getHints();
        m_currentHints.push_back(hints);

        if ( continueTraversal(hints, bs) )
        {
          SharedModelViewTraverser::handleLOD(p);
        }

        m_currentHints.pop_back();

        m_curPath->pop();
      }

      void  RayIntersectTraverser::handleSwitch( const Switch * p )
      {
        m_curPath->push( p->getSharedPtr<Object>() );

        Sphere3f bs( p->getBoundingSphere() );

        int hints = m_currentHints.back() | p->getHints();
        m_currentHints.push_back(hints);

        if ( continueTraversal(hints, bs) )
        {
          SharedModelViewTraverser::handleSwitch(p);
        }

        m_currentHints.pop_back();

        m_curPath->pop();
      }

      void  RayIntersectTraverser::handleTransform( const Transform * p )
      {
        m_curPath->push( p->getSharedPtr<Object>() );

        Sphere3f bs( p->getBoundingSphere() );

        int hints = m_currentHints.back() | p->getHints();
        m_currentHints.push_back(hints);

        if ( continueTraversal(hints, bs) )
        {
          SharedModelViewTraverser::handleTransform(p);
        }

        m_currentHints.pop_back();

        m_curPath->pop();
      }

      inline bool RayIntersectTraverser::continueTraversal(unsigned int hints, const Sphere3f& bs)
      {
        return !(hints & Object::DP_SG_HINT_ALWAYS_INVISIBLE) &&
                isValid( bs ) && checkIntersection( bs ) &&
                ( (hints & GeoNode::DP_SG_HINT_DONT_CLIP) || checkClipPlanes( bs ) );
      }

      bool RayIntersectTraverser::preTraverseGroup( const Group * p )
      {
        if ( 0 < p->getNumberOfActiveClipPlanes() )
        {
          m_clipPlanes.push( m_clipPlanes.top() );
          for ( Group::ClipPlaneConstIterator gcpci = p->beginClipPlanes() ; gcpci != p->endClipPlanes() ; ++gcpci )
          {
            if ( (*gcpci)->isEnabled() )
            {
              m_clipPlanes.top().push_back( *gcpci );
            }
          }
        }
        return( SharedModelViewTraverser::preTraverseGroup( p ) );
      }

      void RayIntersectTraverser::postTraverseGroup( const Group * p )
      {
        SharedModelViewTraverser::postTraverseGroup( p );
        if ( 0 < p->getNumberOfActiveClipPlanes() )
        {
          m_clipPlanes.pop();
        }
      }

      bool RayIntersectTraverser::preTraverseTransform( const Trafo *p )
      {
        bool ok = SharedModelViewTraverser::preTraverseTransform( p ) && !isSingular( m_transformStack.getWorldToModel() );
        if ( ok )
        {
          m_scaleFactors.push( m_scaleFactors.top() * maxElement( p->getScaling() ) );
          m_msRayOrigin.push( Vec3f( Vec4f( m_rayOrigin, 1.0f ) * m_transformStack.getWorldToModel() ) );
          Vec3f dir = Vec3f( Vec4f( m_rayDir, 0.0f ) * m_transformStack.getWorldToModel() );
          dir.normalize();
          m_msRayDir.push( dir );
          dir = Vec3f( Vec4f( m_camera->getDirection(), 0.0f ) * m_transformStack.getWorldToModel() );
          dir.normalize();
          m_msCamDir.push( dir );
        }
        return( ok );
      }

      void RayIntersectTraverser::postTraverseTransform( const Trafo *p )
      {
        SharedModelViewTraverser::postTraverseTransform( p );
        m_msRayOrigin.pop();
        m_msRayDir.pop();
        m_msCamDir.pop();
        m_scaleFactors.pop();
      }

      bool RayIntersectTraverser::checkClipPlanes( const Vec3f & p )
      {
        bool ok = true;
        for ( size_t i=0 ; ok && i<m_clipPlanes.top().size() ; i++ )
        {
          ok = isInsideHalfSpace(m_clipPlanes.top()[i], p);
        }
        return( ok );
      }

      bool RayIntersectTraverser::checkClipPlanes( const Sphere3f &p )
      {
        bool ok = true;
        for ( size_t i=0 ; ok && i<m_clipPlanes.top().size() ; i++ )
        {
          float r(p.getRadius());
          ok = isInsideHalfSpace(m_clipPlanes.top()[i], p.getCenter() + Vec3f(r, r, r));
        }
        return( ok );
      }

      bool RayIntersectTraverser::checkIntersection( const Sphere3f &sphere )
      {
        // see ray / sphere intersection (optimized solution)
        // p.299, Tomas Möller, Eric Haines "Real-Time Rendering"

        Vec3f l = Vec3f( Vec4f( sphere.getCenter(), 1.0f ) * m_transformStack.getModelToWorld() ) - m_rayOrigin;
        float d = l * m_rayDir;
        float l2 = l * l;
        float r = sphere.getRadius() * m_scaleFactors.top();
        float r2 = r * r;

        bool intersects = ( ( l2 <= r2 ) || ( ( 0.0f <= d ) && ( ( l2 - d*d ) <= r2 ) ) );
        if ( intersects && m_camClipping )
        {
          Vec3f cl = Vec3f( Vec4f( sphere.getCenter(), 1.0f ) * m_transformStack.getModelToWorld() ) - m_camera->getPosition();
          float cd = cl * m_camera->getDirection();
          if ( m_camera.isPtrTo<FrustumCamera>() )
          {
            //  if near/far clipping, intersection is valid only if the sphere is not behind the far plane
            //  and not in front of the near plane
            FrustumCameraSharedPtr const& fc = m_camera.staticCast<FrustumCamera>();
            intersects = ( ( cd - r ) <= fc->getFarDistance() ) && ( fc->getNearDistance() <= ( cd + r ) );
          }
          else
          {
            // for non-FrustumCameras, we just can check if distance is positive
            intersects = ( 0.0f < cd + r );
          }
        }
        return( intersects );
      }

      Vec3f RayIntersectTraverser::getModelIntersection( float dist )  // Input distance is in modelspace!
      {
        return m_msRayOrigin.top() + dist * m_msRayDir.top();
      }

      Vec3f RayIntersectTraverser::getWorldIntersection( const Vec3f& misp )
      {
        return Vec3f( Vec4f( misp, 1.0f ) * m_transformStack.getModelToWorld() );  // intersection point in world space
      }

      bool RayIntersectTraverser::isClipped( const Vec3f& isp, const Vec3f& misp )
      {
        Vec3f ctop  = isp  - m_camera->getPosition();   //  vector from camera position to intersection point
        float cdist = ctop * m_camera->getDirection();  //  distance along camera direction
        bool outOfView = m_camClipping;
        if ( m_camClipping )
        {
          if ( m_camera.isPtrTo<FrustumCamera>() )
          {
            FrustumCameraSharedPtr const& fc = m_camera.staticCast<FrustumCamera>();
            outOfView = ( cdist < fc->getNearDistance() ) || ( fc->getFarDistance() < cdist );
          }
          else
          {
            // for non-FrustumCameras, with camera clipping enabled, we classify a point behind the camera as out of view
            outOfView = ( cdist < 0.0f );
          }
        }
        if ( outOfView || !checkClipPlanes( misp ) )
        {
          return( true );
        }

        return( false );
      }

      bool RayIntersectTraverser::intersectBox( const Vec3f &p0, const Vec3f &p1, Vec3f &isp, float &dist )
      {
      #define LEFT    0
      #define MIDDLE  1
      #define RIGHT   2

        DP_ASSERT( ( p0[0] <= p1[0] ) && ( p0[1] <= p1[1] ) && ( p0[2] <= p1[2] ) );
        DP_ASSERT( isNormalized( m_msRayDir.top() ) );
        Vec3f rayDir = m_msRayDir.top();
        Vec3f rayOrg = m_msRayOrigin.top();

        //  first assume ray starts in box
        bool ok = true;
        isp = rayOrg;
        dist = 0.0f;

        //  find candidate planes
        char  octant[3];
        float candidate[3];
        for ( unsigned int i=0 ; i<3 ; i++ )
        {
          if ( rayOrg[i] < p0[i] )
          {
            octant[i] = LEFT;
            candidate[i] = p0[i];
            ok = false;
          }
          else if ( p1[i] < rayOrg[i] )
          {
            octant[i] = RIGHT;
            candidate[i] = p1[i];
            ok = false;
          }
          else
          {
            octant[i] = MIDDLE;
          }
        }

        if ( ! ok )
        {
          //  ray origin outside of the box
          //  calculate distances to candidate planes
          float maxDist[3];
          for ( unsigned int i=0 ; i<3 ; i++ )
          {
            maxDist[i] = ( ( octant[i] != MIDDLE ) && ( FLT_EPSILON < abs(rayDir[i]) ) )
                        ? ( candidate[i] - rayOrg[i] ) / rayDir[i]
                        : -1.0f;
          }

          //  get largest of the maxDist's for final choice of intersection
          unsigned int index = ( maxDist[0] < maxDist[1] ) ? 1 : 0;
          if ( maxDist[index] < maxDist[2] )
          {
            index = 2;
          }

          //  check final candidate actually inside box
          ok = ( 0.0f < maxDist[index] );
          if ( ok )
          {
            dist = maxDist[index];
            for ( unsigned int i=0 ; ok && i<3 ; i++ )
            {
              if ( index != i )
              {
                isp[i] = rayOrg[i] + maxDist[index] * rayDir[i];
                ok = ! (    ( ( octant[i] == RIGHT ) && ( isp[i] <= p0[i] ) )
                        ||  ( ( octant[i] == LEFT  ) && ( p1[i] <= isp[i] ) ) );
              }
              else
              {
                isp[i] = candidate[i];
              }
            }
          }
        }

        if ( ok )
        {
          Vec3f misp = getModelIntersection( dist );

          isp  = getWorldIntersection( misp ); // Return the intersection point in world space.
          dist = distance(m_rayOrigin, isp);   // Return the intersection distance in world space.

          ok = !isClipped( isp, misp );
        }

        return ok;

      #undef LEFT
      #undef MIDDLE
      #undef RIGHT
      }

      bool RayIntersectTraverser::intersectLine( const Vec3f & v0, const Vec3f & v1, float width, Vec3f & isp, float &dist )
      {
        DP_ASSERT( isNormalized( m_msRayDir.top() ) );

        Vec3f lineDir = v1 - v0;
        float lineLength = length( lineDir );
        if ( lineLength < FLT_EPSILON )
        {
          return( intersectPoint( v0, width, isp, dist ) );
        }
        lineDir /= lineLength;    // normalized lineDir

        bool ok = false;
        Vec3f rayDir = m_msRayDir.top();
        Vec3f rayOrg = m_msRayOrigin.top();

        Vec3f crossDir = rayDir ^ lineDir;
        if ( isNull( crossDir ) )
        {
          float t0 = rayDir * ( v0 - rayOrg );  // signed distance of orthogonal projection of v0 on ray from ray origin
          float t1 = rayDir * ( v1 - rayOrg );  // signed distance of orthogonal projection of v1 on ray from ray origin
          if ( ( 0.0f < t0 ) || ( 0.0f < t1 ) )
          {
            // if at least one of those distances is positive, an intersection might be in front
            if ( ( 0.0f < t0 ) && ( 0.0f < t1 ) )
            {
              // if both distances are positive, take the closer endpoint of the line
              if ( t0 < t1 )
              {
                ok = equal( v0, rayOrg + t0 * rayDir, width );
                if ( ok )
                {
                  dist = t0;
                }
              }
              else
              {
                ok = equal( v1, rayOrg + t1 * rayDir, width );
                if ( ok )
                {
                  dist = t1;
                }
              }
            }
            else
            {
              // if only one distance is positive, the ray origin might be on the line -> get the corresponding point on the line
              float s = lineDir * ( rayOrg - v0 );  // signed distance of orthogonal projecton of rayOrg on line (started at v0)
              DP_ASSERT( ( 0.0f <= s ) && ( s <= lineLength ) );  // if we get here, s has to be in that range !
              ok = equal( v0 + s * lineDir, rayOrg, width );
              if ( ok )
              {
                dist = 0.0f;
              }
            }
          }
        }
        else
        {
          //  ray and line are not parallel => determine nearest points
          crossDir /= lengthSquared( crossDir );
          Vec3f diff = v0 - rayOrg;
          float t = ( diff ^ lineDir ) * crossDir;  // distance of nearest point on ray
          if ( 0.0f < t )                           // happens on positive side of the ray ?
          {
            float s = ( diff ^ rayDir ) * crossDir; // distance of neareast point on line (measured from v0)
            Vec3f rayP = rayOrg + t * rayDir;

            if ( s < 0.0f )
            {
              // due to screen resolution, rayP and v0 might be equal, even though v0 is not the nearest point
              ok = equal( rayP, v0, width );
            }
            else
            {
              float lineLength = distance( v0, v1 );
              if ( lineLength < s )
              {
                // again, due to screen resolution, rayP and v1 might be equal
                ok = equal( rayP, v1, width );
              }
              else
              {
                // just check, if the nearest points on ray and line are equal (respecting screen resolution!)
                ok = equal( rayP, v0 + s * lineDir, width );
              }
            }
            if ( ok )
            {
              dist = t;
            }
          }
        }

        if ( ok )
        {
          Vec3f misp = getModelIntersection( dist );

          isp  = getWorldIntersection( misp ); // Return the intersection point in world space.
          dist = distance(m_rayOrigin, isp);   // Return the intersection distance in world space.

          ok = !isClipped( isp, misp );
        }

        return ok;
      }

      bool RayIntersectTraverser::intersectPoint( const Vec3f & v0, float size, Vec3f & isp, float &dist )
      {
        DP_ASSERT( isNormalized( m_msRayDir.top() ) );
        dist = ( v0 - m_msRayOrigin.top() ) * m_msRayDir.top();
        if ( FLT_EPSILON < dist &&
             equal( v0, m_msRayOrigin.top() + dist * m_msRayDir.top(), size ) )
        {
          Vec3f misp = getModelIntersection( dist );

          isp  = getWorldIntersection( misp );  // Return the intersection point in world space.
          dist = distance(m_rayOrigin, isp);    // Return the intersection distance in world space.

          return !isClipped( isp, misp );
        }
        return false;
      }

      bool RayIntersectTraverser::intersectTriangle( const Vec3f & v0
                                                   , const Vec3f & v1
                                                   , const Vec3f & v2
                                                   , Vec3f & isp
                                                   , float &dist )
      {
        // see ray / triangle intersection
        // p.305, Tomas Möller, Eric Haines "Real-Time Rendering"
        // or: http://www.acm.org/jgt/papers/MollerTrumbore97/

        // use double precision in calculations
        Vec3d vd0(v0);
        Vec3d vd1(v1);
        Vec3d vd2(v2);
        Vec3d rayDir(m_msRayDir.top());
        Vec3d rayOrg(m_msRayOrigin.top());

        // find vectors for two edges sharing v0
        Vec3d e1 = vd1 - vd0;
        Vec3d e2 = vd2 - vd0;

        // begin calculating determinant - also used to calculate U parameter
        Vec3d p  = rayDir ^ e2;

        // if determinant is near zero, ray lies in plane of triangle
        double det = e1 * p;

        if ( fabs(det) < DBL_EPSILON )
        {
          return false;
        }
        double invDet = 1.0 / det;

        // calculate distance from vert0 to ray origin
        Vec3d s = rayOrg - vd0;

        // calculate U parameter and test bounds
        double u = invDet * (s * p);
        if ( u < 0.0 || 1.0 < u )
        {
          return false;
        }

        // prepare to test V parameter
        Vec3d q = s ^ e1;

        // calculate V parameter and test bounds
        double v = invDet * (rayDir * q);
        if ( v < 0.0 || 1.0 < u + v ) // Sum of barycentric coordinates cannot be greater than 1.0
        {
          return false;
        }

        // intersection:
        // dist = distance to the rays origin
        // (u,v) barycentric coordinates
        double ddist = invDet * (e2 * q);
        if ( ddist < 0.0 )
        {
          return false;
        }
        // back to single precision
        // NOTE: the result is incorrect if ddist > FLT_MAX or < -FLT_MAX!
        dist = doubleToFloat(ddist);

        Vec3f misp = getModelIntersection( dist );

        isp  = getWorldIntersection( misp ); // Return the intersection point in world space.
        dist = distance(m_rayOrigin, isp);   // Return the intersection distance in world space.

        return !isClipped( isp, misp );
      }

      void RayIntersectTraverser::storeIntersection( const Primitive * p
                                                   , const Vec3f & isp
                                                   , float dist
                                                   , unsigned int primitiveIndex
                                                   , const vector<unsigned int> & vertexIndices )
      {
        if (!m_intersectionList.empty() && dist < m_intersectionList[m_nearestIntIdx].getDist())
        {
          // we can use the size as the index, because we have not added the intersection yet
          m_nearestIntIdx = dp::checked_cast<unsigned int>(m_intersectionList.size());
        }

        // OK, here we have an intersection, so let's store it
        m_intersectionList.push_back(Intersection( dp::sg::core::Path::create( m_curPath )
                                                 , p->getSharedPtr<Primitive>()
                                                 , isp
                                                 , dist
                                                 , primitiveIndex
                                                 , vertexIndices ) );
      }

      void RayIntersectTraverser::checkLine( const Primitive * p, const Buffer::ConstIterator<Vec3f>::Type &vertices
                                           , unsigned int i0, unsigned int i1, unsigned int pi )
      {
        const Vec3f & v0 = vertices[i0];
        const Vec3f & v1 = vertices[i1];

        Vec3f isp;
        float dist;
        if ( intersectLine( v0, v1, m_currentLineWidth, isp, dist ) )
        {
          vector<unsigned int> vertIndices;
          vertIndices.push_back(i0);
          vertIndices.push_back(i1);

          storeIntersection( p, isp, dist, pi, vertIndices );
        }
      }

      void RayIntersectTraverser::checkQuad( const Primitive * p, const Buffer::ConstIterator<Vec3f>::Type &vertices
                                           , unsigned int i0, unsigned int i1, unsigned int i2, unsigned int i3, unsigned int pi )
      {
        const Vec3f & v0 = vertices[i0];
        const Vec3f & v1 = vertices[i1];
        const Vec3f & v2 = vertices[i2];
        const Vec3f & v3 = vertices[i3];

        Vec3f isp;
        float dist;
        if (  intersectTriangle( v0, v1, v2, isp, dist )
           || intersectTriangle( v2, v3, v0, isp, dist ) )
        {
          vector<unsigned int> vertIndices;
          vertIndices.push_back(i0);
          vertIndices.push_back(i1);
          vertIndices.push_back(i2);
          vertIndices.push_back(i3);

          // Assuming planar quads, otherwise the returned distance might not actually be the nearest one.
          storeIntersection( p, isp, dist, pi, vertIndices );
        }
      }

      void RayIntersectTraverser::checkTriangle( const Primitive * p, const Buffer::ConstIterator<Vec3f>::Type &vertices
                                               , unsigned int i0, unsigned int i1, unsigned int i2, unsigned int pi )
      {
        const Vec3f & v0 = vertices[i0];
        const Vec3f & v1 = vertices[i1];
        const Vec3f & v2 = vertices[i2];

        Vec3f isp;
        float dist;
        if ( intersectTriangle( v0, v1, v2, isp, dist ) )
        {
          vector<unsigned int> vertIndices;
          vertIndices.push_back(i0);
          vertIndices.push_back(i1);
          vertIndices.push_back(i2);

          storeIntersection( p, isp, dist, pi, vertIndices );
        }
      }

      bool RayIntersectTraverser::equal( const Vec3f & v0, const Vec3f & v1, float width ) const
      {
        bool ok;
        if ( m_viewportWidth && m_viewportHeight )
        {
          //  project the two point into normalized screen space
          Mat44f m2s = m_transformStack.getModelToClip();
          Vec4f p0 = Vec4f( v0, 1.0f ) * m2s;
          p0 /= p0[3];
          Vec4f p1 = Vec4f( v1, 1.0f ) * m2s;
          p1 /= p1[3];

          //  determine distance in screen space
          ok = (  distance( Vec2f( m_viewportWidth * ( 1 + p0[0] ) / 2, m_viewportHeight * ( 1 - p0[1] ) / 2 )
                          , Vec2f( m_viewportWidth * ( 1 + p1[0] ) / 2, m_viewportHeight * ( 1 - p1[1] ) / 2 ) )
                < width );
        }
        else
        {
          ok = ( distance( v0, v1 ) < width );
        }
        return( ok );
      }

      void RayIntersectTraverser::handleTriangles( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );

          // assume no primitive restarts in data stream
          for( unsigned int i = 0, j=0; i < count; i+=3, j++ )
          {
            checkTriangle( p, vertices, iter[i], iter[i+1], iter[i+2], j );
          }
        }
        else
        {
          for ( unsigned int i=offset, j=0 ; i<offset+count ; i+=3, j++ )
          {
            checkTriangle( p, vertices, i, i+1, i+2, j );
          }
        }
      }

      void RayIntersectTraverser::handleTriangleStrip( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );
          unsigned int prIdx = p->getIndexSet()->getPrimitiveRestartIndex();

          for( unsigned int i = 2, j=0; i < count; i++ )
          {
            if( iter[i] == prIdx )
            {
              // increment twice so that i will start at base + 2, after loop increment
              i+=2;
              ++j;  // index of the next tri strip
              continue;
            }

            checkTriangle( p, vertices, iter[i-2], iter[i-1], iter[i], j );
          }
        }
        else
        {
          for ( unsigned int i=offset+2 ; i<offset+count ; ++i )
          {
            checkTriangle( p, vertices, i-2, i-1, i, 0 /* only one possible */ );
          }
        }
      }

      void RayIntersectTraverser::handleTriangleFan( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );
          unsigned int prIdx = p->getIndexSet()->getPrimitiveRestartIndex();
          unsigned int startIdx = 0;

          for( unsigned int i = 2, j=0; i < count; i++ )
          {
            if( iter[i] == prIdx )
            {
              i++;
              startIdx = i; // set startidx at next index in list

              // increment one more so that i will start at startIdx + 2, after
              // loop increment
              i++;
              ++j;  // index of the next tri fan
              continue;
            }

            checkTriangle( p, vertices, iter[startIdx], iter[i-1], iter[i], j );
          }
        }
        else
        {
          for ( unsigned int i=offset+2 ; i<offset+count ; ++i )
          {
            checkTriangle( p, vertices, offset, i-1, i, 0 /* can be only one */ );
          }
        }
      }

      void RayIntersectTraverser::handleLines( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );

          // assume no primitive restarts
          for( unsigned int i = 0, j=0; i < count; i+=2,j++ )
          {
            checkLine( p, vertices, iter[i], iter[i+1], j );
          }
        }
        else
        {
          for ( unsigned int i=offset, j=0 ; i<offset+count ; i+=2, j++ )
          {
            checkLine( p, vertices, i, i+1, j );
          }
        }
      }

      void RayIntersectTraverser::handleLineStrip( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );
          unsigned int prIdx = p->getIndexSet()->getPrimitiveRestartIndex();

          for( unsigned int i = 1, j=0; i < count; i++ )
          {
            if( iter[i] == prIdx )
            {
              // skip pridx
              ++i;  // To start on the 2nd vertex after the for-loop increment.
              ++j;  // index of the next line strip
              continue;
            }

            checkLine( p, vertices, iter[i-1], iter[i], j );
          }
        }
        else
        {
          for ( unsigned int i=offset+1 ; i<offset+count ; ++i )
          {
            checkLine( p, vertices, i-1, i, 0 /*only one strip possible in array*/ );
          }
        }
      }

      void RayIntersectTraverser::handleLineLoop( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );
          unsigned int prIdx = p->getIndexSet()->getPrimitiveRestartIndex();
          unsigned int startIdx = 0;

          for( unsigned int i = 0, j=0; i < count; i+=2 )
          {
            if( iter[i] == prIdx )
            {
              // check last line in loop
              checkLine( p, vertices, iter[i-1], iter[startIdx], j );
              startIdx = i+1;
              // subtract 1 so that when i is incremented by two we are back on track.
              i-=1;
              ++j;  // index of the next line loop

              continue;
            }

            checkLine( p, vertices, iter[i], iter[i+1], j );
          }
        }
        else
        {
          unsigned int i;

          for ( i=offset ; i<offset+count ; i+=2 )
          {
            checkLine( p, vertices, i, i+1, 0 /*only one strip possible in array*/ );
          }

          // check last line in loop
          checkLine( p, vertices, i-1, 0, 0 /*only one loop possible in array*/ );
        }
      }


      void RayIntersectTraverser::checkAPoint( const Primitive *p, const Vec3f & v0, unsigned int idx )
      {
        Vec3f isp;
        float dist;
        if ( intersectPoint( v0, m_currentPointSize, isp, dist ) )
        {
          vector<unsigned int> vertIndices;
          vertIndices.push_back(idx);

          storeIntersection( p
                           , isp
                           , dist
                           , idx  // index of the point
                           , vertIndices );
        }
      }

      void RayIntersectTraverser::handlePoints( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );

          // assume no primitive restarts in indices
          for( unsigned int i = 0; i < count; i++ )
          {
            checkAPoint( p, vertices[ iter[i] ], i );
          }
        }
        else
        {
          for ( unsigned int i=offset ; i<offset+count ; i++ )
          {
            checkAPoint( p, vertices[i], i );
          }
        }
      }

      void RayIntersectTraverser::handleQuads( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );

          // assume no primitive restarts in indices
          for( unsigned int i = 0, j=0; i < count; i += 4, j++ )
          {
            checkQuad( p, vertices, iter[i], iter[i+1], iter[i+2], iter[i+3], j );
          }
        }
        else
        {
          for ( unsigned int i=offset, j=0 ; i<offset+count ; i+=4, j++ )
          {
            checkQuad( p, vertices, i, i+1, i+2, i+3, j );
          }
        }
      }

      void RayIntersectTraverser::handleQuadStrip( const Primitive *p )
      {
        Buffer::ConstIterator<Vec3f>::Type vertices = p->getVertexAttributeSet()->getVertices();
        unsigned int offset = p->getElementOffset();
        unsigned int count  = p->getElementCount();

        if( p->isIndexed() )
        {
          IndexSet::ConstIterator<unsigned int> iter( p->getIndexSet(), offset );
          unsigned int prIdx = p->getIndexSet()->getPrimitiveRestartIndex();

          for( unsigned int i = 3, j=0; i < count; i+=2 )
          {
            if( iter[i] == prIdx )
            {
              // increment thrice so that i will start at base + 3, after loop increment
              i+=3;
              ++j;  // index of the next quad strip
              continue;
            }

            checkQuad( p, vertices, iter[i-3], iter[i-2], iter[i], iter[i-1], j );
          }
        }
        else
        {
          for ( unsigned int i=offset+3 ; i<offset+count ; ++i )
          {
            checkQuad( p, vertices, i-3, i-2, i, i-1, 0 /* only one possible */ );
          }
        }
      }

      void RayIntersectTraverser::handlePatches( const Primitive * p )
      {
        // for patches with 3 vertices, we assume it's close to a triangle and handle it like that
        // for all others, we can't do anything
        DP_ASSERT( p->getPrimitiveType() == PRIMITIVE_PATCHES );
        if ( verticesPerPatch( p->getPatchesType() ) == 3 )
        {
          handleTriangles( p );
        }
      }

      void RayIntersectTraverser::handlePrimitive( const Primitive * p )
      {
        Sphere3f bs( p->getBoundingSphere() );

        unsigned int hints = m_currentHints.back() | p->getHints();
        if ( continueTraversal(hints, bs) )
        {
          // dispatch to proper handler
          switch( p->getPrimitiveType() )
          {
            case PRIMITIVE_POINTS:
              handlePoints( p );
              break;

            case PRIMITIVE_LINES:
              handleLines( p );
              break;

            case PRIMITIVE_LINE_STRIP:
              handleLineStrip( p );
              break;

            case PRIMITIVE_LINE_LOOP:
              handleLineLoop( p );
              break;

            case PRIMITIVE_TRIANGLES:
              handleTriangles( p );
              break;

            case PRIMITIVE_TRIANGLE_STRIP:
              handleTriangleStrip( p );
              break;

            // handle polygon like a fan for now, assuming it is convex and verts are coplanar
            case PRIMITIVE_POLYGON:
            case PRIMITIVE_TRIANGLE_FAN:
              handleTriangleFan( p );
              break;

            case PRIMITIVE_QUADS:
              handleQuads( p );
              break;

            case PRIMITIVE_QUAD_STRIP:
              handleQuadStrip( p );
              break;

            case PRIMITIVE_TRIANGLE_STRIP_ADJACENCY:
            case PRIMITIVE_LINE_STRIP_ADJACENCY:
            case PRIMITIVE_LINES_ADJACENCY:
            case PRIMITIVE_TRIANGLES_ADJACENCY:
              // TODO: ADD support for ME
              break;

            case PRIMITIVE_PATCHES:
              handlePatches( p );
              break;

            default:
            case PRIMITIVE_UNINITIALIZED:
              break;
          }
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
