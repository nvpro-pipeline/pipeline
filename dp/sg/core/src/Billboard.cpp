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


#include <dp/sg/core/Billboard.h>
#include <dp/sg/core/Camera.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( Billboard, Alignment);
      DEFINE_STATIC_PROPERTY( Billboard, RotationAxis);

      BEGIN_REFLECTION_INFO ( Billboard )
        DERIVE_STATIC_PROPERTIES( Billboard, Group );
        INIT_STATIC_PROPERTY_RW_ENUM( Billboard, Alignment,     Alignment,  SEMANTIC_VALUE,  value,           value );
        INIT_STATIC_PROPERTY_RW     ( Billboard, RotationAxis,  Vec3f,      SEMANTIC_VALUE,  const_reference, const_reference );
      END_REFLECTION_INFO

      BillboardSharedPtr Billboard::create()
      {
        return( std::shared_ptr<Billboard>( new Billboard() ) );
      }

      HandledObjectSharedPtr Billboard::clone() const
      {
        return( std::shared_ptr<Billboard>( new Billboard( *this ) ) );
      }

      Billboard::Billboard( void )
      : m_alignment(Alignment::AXIS)
      , m_rotationAxis(0.0f,1.0f,0.0f)
      {
        m_objectCode = ObjectCode::BILLBOARD;
      }

      Billboard::Billboard( const Billboard &rhs )
      : Group(rhs)
      , m_alignment(rhs.m_alignment)
      , m_rotationAxis(rhs.m_rotationAxis)
      {
        m_objectCode = ObjectCode::BILLBOARD;
      }

      Billboard::~Billboard( void )
      {
      }

      Trafo Billboard::getTrafo( CameraSharedPtr const& cam, Mat44f const& worldToModel ) const
      {
        Trafo trafo;

        switch( m_alignment )
        {
          case Alignment::AXIS :
            getTrafoAxisAligned( cam, worldToModel, trafo );
            break;
          case Alignment::SCREEN :
            getTrafoScreenAligned( cam, worldToModel, trafo );
            break;
          case Alignment::VIEWER :
            getTrafoViewerAligned( cam, worldToModel, trafo );
            break;
          default :
            DP_ASSERT( false );
            break;
        }
        return( trafo );
      }

      void Billboard::getTrafoAxisAligned( CameraSharedPtr const& cam, Mat44f const& worldToModel, Trafo & trafo ) const
      {
        DP_ASSERT( m_alignment == Alignment::AXIS );

        //  get the camera position relative to Billboard
        Vec4f viewerPosition = Vec4f( cam->getPosition(), 1.0f ) * worldToModel;
        viewerPosition.normalize();

        //  determine the new z-axis by first normalizing the viewerPosition
        Vec3f newZAxis(viewerPosition);
        newZAxis.normalize();

        //  if the rotation axis and the new z-axis are collinear, the result is undefined
        if ( ! areCollinear( m_rotationAxis, newZAxis ) )
        {
          //  determine quat that rotates the default y-axis into the rotation axis
          Quatf orientation = Quatf( Vec3f( 0.0f, 1.0f, 0.0f ), m_rotationAxis );

          //  get the old z-axis by rotating the standard z-axis accordingly
          Vec3f oldZAxis = Vec3f( 0.0f, 0.0f, 1.0f ) * orientation ;

          //  determine the corrected z-axis
          Vec3f xAxis = m_rotationAxis ^ newZAxis;
          xAxis.normalize();
          newZAxis = xAxis ^ m_rotationAxis;

          //  set the trafo to be the rotation that rotates the old z-axis into the new one
          trafo.setOrientation( Quatf( oldZAxis, newZAxis ) );
        }
      }

      void Billboard::getTrafoScreenAligned( CameraSharedPtr const& cam, Mat44f const& worldToModel, Trafo & trafo ) const
      {
        DP_ASSERT( m_alignment == Alignment::SCREEN );

        //  get the (negative) camera direction as the new z axis
        Vec3f newZAxis( Vec4f( -cam->getDirection(), 0.0f ) * worldToModel );
        newZAxis.normalize();

        //  get the up vector of the camera as the rotation axis
        Vec3f upVector( Vec4f( cam->getUpVector(), 0.0f ) * worldToModel );
        upVector.normalize();
        DP_ASSERT( areOrthonormal( newZAxis, upVector ) );

        //  determine the newXAxis as the cross product of newYAxis and newZAxis
        Vec3f newXAxis = upVector ^ newZAxis;

        //  determine the rotation matrix that does just that
        Mat33f rot( { newXAxis, upVector, newZAxis } );
        DP_ASSERT( isRotation( rot ) );

        //  and set the Orientation
        trafo.setOrientation( Quatf( rot ) );
      }

      void Billboard::getTrafoViewerAligned( CameraSharedPtr const& cam, Mat44f const& worldToModel, Trafo & trafo ) const
      {
        DP_ASSERT( m_alignment == Alignment::VIEWER );

        //  get the up vector of the camera as the rotation axis
        Vec3f upVector( Vec4f( cam->getUpVector(), 0.0f ) * worldToModel );
        upVector.normalize();

        //  get the camera position relative to Billboard
        Vec4f viewerPosition = Vec4f( cam->getPosition(), 1.0f ) * worldToModel;
        DP_ASSERT( FLT_EPSILON < length( viewerPosition ) );

        //  determine the new z-axis by first normalizing the viewerPosition
        Vec3f newZAxis(viewerPosition);
        newZAxis.normalize();

        if ( areCollinear( newZAxis, upVector ) )
        {
          //  the new z-axis and the old y-axis are collinear, so we just rotate to get the new z-axis
          trafo.setOrientation( Quatf( Vec3f( 0.0f, 0.0f, 1.0f), newZAxis ) );
        }
        else
        {
          //  determine the new y-axis by projecting the upVector on the plane determined by newZAxis
          Vec3f newYAxis = orthonormalize( newZAxis, upVector );

          //  determine the newXAxis as the cross product of newYAxis and newZAxis
          Vec3f newXAxis = newYAxis ^ newZAxis;

          //  determine the rotation matrix that does just that
          Mat33f rot( { newXAxis, newYAxis, newZAxis } );
          DP_ASSERT( isRotation( rot ) );

          //  and set the Orientation
          trafo.setOrientation( Quatf( rot ) );
        }
      }

      Billboard & Billboard::operator=(const Billboard & rhs)
      {
        if (&rhs != this)
        {
          Group::operator=(rhs);

          m_rotationAxis = rhs.m_rotationAxis;
          m_alignment    = rhs.m_alignment;
        }
        return *this;
      }

      bool Billboard::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<Billboard>() && Group::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          BillboardSharedPtr const& b = object.staticCast<Billboard>();
          equi = ( m_alignment == b->m_alignment )
              && ( ( m_alignment != Alignment::AXIS ) || ( m_rotationAxis == b->m_rotationAxis ) );
        }
        return( equi );
      }

      void Billboard::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Group::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_alignment), sizeof(m_alignment) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_rotationAxis), sizeof(m_rotationAxis) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

namespace dp
{
  namespace util
  {

    template <> const std::string EnumReflection<dp::sg::core::Billboard::Alignment>::name = "BillboardAlignment";

    template <> const std::map<dp::sg::core::Billboard::Alignment,std::string> EnumReflection<dp::sg::core::Billboard::Alignment>::values =
    {
      { dp::sg::core::Billboard::Alignment::AXIS,   "axis"    },
      { dp::sg::core::Billboard::Alignment::VIEWER, "viewer"  },
      { dp::sg::core::Billboard::Alignment::SCREEN, "screen"  }
    };

  } //namespace util
} // namespace dp
