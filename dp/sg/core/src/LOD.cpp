// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/sg/core/LOD.h>
#include <cstring>

using namespace dp::math;

namespace dp
{
  namespace sg
  {
    namespace core
    {

      DEFINE_STATIC_PROPERTY( LOD, Center );

      BEGIN_REFLECTION_INFO( LOD )
        DERIVE_STATIC_PROPERTIES( LOD, Group );
        INIT_STATIC_PROPERTY_RW ( LOD, Center,    Vec3f,  Semantic::POSITION, const_reference, const_reference );
      END_REFLECTION_INFO

      LODSharedPtr LOD::create()
      {
        return( std::shared_ptr<LOD>( new LOD() ) );
      }

      HandledObjectSharedPtr LOD::clone() const
      {
        return( std::shared_ptr<LOD>( new LOD( *this ) ) );
      }

      LOD::LOD( void )
      : m_center(0.f, 0.f, 0.f)
      , m_isRangeLocked(false)
      , m_rangeLock(~0)
      {
        m_objectCode = ObjectCode::LOD;
      }

      LOD::LOD( const LOD &rhs )
      : Group(rhs)
      , m_center(rhs.m_center)
      , m_ranges(rhs.m_ranges)
      , m_isRangeLocked(rhs.m_isRangeLocked)
      , m_rangeLock(rhs.m_rangeLock)
      {
        m_objectCode = ObjectCode::LOD;
      }

      LOD::~LOD(void)
      {
      }

      bool LOD::setRanges( const float * ranges, unsigned int count )
      {
        if (    ( m_ranges.size() != count )
            ||  ( count && memcmp( &m_ranges[0], ranges, count * sizeof(float ) ) != 0 ) )
        {
          m_ranges.clear();
          m_ranges.reserve(count);
          if (ranges)
          {
            m_ranges.assign( &ranges[0], &ranges[count] );
          }
          notify( Object::Event(this ) );
          return( true );
        }
        return( false );
      }

      unsigned int LOD::getNumberOfRanges() const
      {
        return( dp::checked_cast<unsigned int>(m_ranges.size()) );
      }

      const float * LOD::getRanges() const
      {
        if (!m_ranges.empty())
        {
          return (const float *) &(m_ranges[0]);
        }
        return NULL;
      }

      void LOD::setCenter(const Vec3f & center)
      {
        if ( m_center != center )
        {
          m_center = center;
          notify( PropertyEvent( this, PID_Center ) );
        }
      }

      const Vec3f & LOD::getCenter() const
      {
        return m_center;
      }

      unsigned int LOD::getLODToUse(const Mat44f & modelViewMatrix, float scaleFactor) const
      {
        unsigned int childLOD = ~0;
        unsigned int numKids  = getNumberOfChildren();
        if ( numKids != 0)
        {
          if (!m_isRangeLocked)
          {
            // transform center of LOD into eye space:
            Vec3f centerES(Vec4f( m_center, 1.0f ) * modelViewMatrix);

            // calculate the squared distance
            // diff   = centerES - eyeES, eyeES = position of the camera
            // on eye space is always (0,0,0) => diff = centerES =>
            float distSQ = lengthSquared( centerES );

            // figure out which child to use:
            for (childLOD = 0; childLOD < m_ranges.size() && childLOD < numKids-1; childLOD++)
            {
              float rangeSQ = m_ranges[childLOD] * scaleFactor;
              rangeSQ *= rangeSQ;

              if (distSQ < rangeSQ)
              {
                break;
              }
            }
          }
          else
          {
            childLOD = std::min(m_rangeLock, numKids-1);
          }
        }

        return childLOD;
      }

      LOD & LOD::operator=(const LOD & rhs)
      {
        if (&rhs != this)
        {
          Group::operator=(rhs);

          m_center        = rhs.m_center;
          m_ranges        = rhs.m_ranges;
          m_isRangeLocked = rhs.m_isRangeLocked;
          m_rangeLock     = rhs.m_rangeLock;
          notify( PropertyEvent( this, PID_Center ) );
          notify( Object::Event( this ) );
        }
        return *this;
      }

      void LOD::setRangeLock(bool on, unsigned int rangeNumber)
      {
        if (  m_isRangeLocked != on || m_rangeLock != rangeNumber )
        {
          m_isRangeLocked = on;
          m_rangeLock = rangeNumber;
          notify( Object::Event(this ) );
        }
      }

      unsigned int LOD::getRangeLock() const
      {
        return m_rangeLock;
      }

      bool LOD::isRangeLockEnabled() const
      {
        return m_isRangeLocked;
      }

      bool LOD::isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const
      {
        if ( object.get() == this )
        {
          return( true );
        }

        bool equi = std::dynamic_pointer_cast<LOD>(object) && Group::isEquivalent( object, ignoreNames, deepCompare );
        if ( equi )
        {
          LODSharedPtr const& l = std::static_pointer_cast<LOD>(object);
          equi =    ( m_center        == l->m_center        )
                &&  ( m_ranges        == l->m_ranges        )
                &&  ( m_isRangeLocked == l->m_isRangeLocked )
                &&  ( !m_isRangeLocked || ( m_rangeLock == l->m_rangeLock ) );
        }
        return( equi );
      }

      void LOD::feedHashGenerator( util::HashGenerator & hg ) const
      {
        Group::feedHashGenerator( hg );
        hg.update( reinterpret_cast<const unsigned char *>(&m_center), sizeof(m_center) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_ranges), dp::checked_cast<unsigned int>(m_ranges.size() * sizeof(float) ) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_rangeLock), sizeof(m_rangeLock) );
        hg.update( reinterpret_cast<const unsigned char *>(&m_isRangeLocked), sizeof(m_isRangeLocked) );
      }

      inline bool rangesLessThan( LODSharedPtr const& lhs, LODSharedPtr const& rhs)
      {
        return lhs->getNumberOfRanges() < rhs->getNumberOfRanges() ||
               (lhs->getNumberOfRanges() == rhs->getNumberOfRanges() && memcmp(lhs->getRanges(), rhs->getRanges(), lhs->getNumberOfRanges() * sizeof(float)) < 0);
      }

      bool LODLessThan::operator()( const LODSharedPtr& lhs, const LODSharedPtr& rhs) const
      {
        return (   lhs->getNumberOfChildren() < rhs->getNumberOfChildren()  ||
                 !(lhs->getNumberOfChildren() > rhs->getNumberOfChildren()) &&   lhs->getCenter() < rhs->getCenter()  ||
                 !(lhs->getNumberOfChildren() > rhs->getNumberOfChildren()) && !(lhs->getCenter() > rhs->getCenter()) &&  rangesLessThan(lhs, rhs) ||
                 !(lhs->getNumberOfChildren() > rhs->getNumberOfChildren()) && !(lhs->getCenter() > rhs->getCenter()) && !rangesLessThan(lhs, rhs) &&   lhs->isRangeLockEnabled() < rhs->isRangeLockEnabled()  ||
                 !(lhs->getNumberOfChildren() > rhs->getNumberOfChildren()) && !(lhs->getCenter() > rhs->getCenter()) && !rangesLessThan(lhs, rhs) && !(lhs->isRangeLockEnabled() > rhs->isRangeLockEnabled()) && lhs->getRangeLock() < rhs->getRangeLock() );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
