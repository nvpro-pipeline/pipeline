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


#include <dp/sg/core/ClipPlane.h>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      BEGIN_REFLECTION_INFO( ClipPlane )
        DERIVE_STATIC_PROPERTIES( ClipPlane, Object )
      END_REFLECTION_INFO

      ClipPlaneSharedPtr ClipPlane::create()
      {
        return( std::shared_ptr<ClipPlane>( new ClipPlane() ) );
      }

      HandledObjectSharedPtr ClipPlane::clone() const
      {
        return( std::shared_ptr<ClipPlane>( new ClipPlane( *this ) ) );
      }

      ClipPlane::ClipPlane()
      : m_plane(Vec3f(0.f, 0.f, 0.f), 0.f) // the default constructor of Plane3f does not perform any initialization!
      , m_enabled(true) // enabled by default
      {
        m_objectCode = ObjectCode::CLIP_PLANE;
      }

      ClipPlane::ClipPlane(const ClipPlane& rhs)
      : Object(rhs)
      , m_plane(rhs.m_plane)
      , m_enabled(rhs.m_enabled)
      {
        m_objectCode = rhs.m_objectCode;
      }

      ClipPlane::~ClipPlane()
      {
      }

      void ClipPlane::setNormal(const Vec3f& normal)
      {
        if ( normal != m_plane.getNormal() )
        {
          m_plane.setNormal(normal);
          notify( Event(this ) );
        }
      }

      void ClipPlane::setOffset(float offset)
      {
        if ( fabs( offset - m_plane.getOffset() ) >= std::numeric_limits<float>::epsilon() )
        {
          m_plane.setOffset(offset);
          notify( Event(this ) );
        }
      }

      void ClipPlane::setEnabled(bool onOff)
      {
        if ( m_enabled != onOff )
        {
          m_enabled = onOff;
          // altering the enable status of a clipping plane requires
          // a render list rebuild to become active
          notify( Object::Event( this ) ); // TREE_INCARNATION
        }
      }

      bool ClipPlane::isEquivalent( ObjectSharedPtr const& object , bool ignoreNames, bool deepCompare) const
      {
        if ( object == this )
        {
          return( true );
        }

        bool equi = object.isPtrTo<ClipPlane>() && Object::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          ClipPlaneSharedPtr const& plane = object.staticCast<ClipPlane>();
          equi = ( m_plane == plane->m_plane ) && ( m_enabled == plane->m_enabled );
        }
        return( equi );
      }

      void ClipPlane::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Object::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_plane), sizeof(m_plane) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_enabled), sizeof(m_enabled) );
      }

    } // namespace core
  } // namespace sg
} // namespace dp