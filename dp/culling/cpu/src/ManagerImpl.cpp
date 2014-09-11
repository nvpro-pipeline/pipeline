// Copyright NVIDIA Corporation 2012-2013
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


#include <dp/culling/cpu/Manager.h>
#include <dp/culling/cpu/inc/ManagerImpl.h>
#include <dp/culling/GroupBitSet.h>
#include <dp/culling/ObjectBitSet.h>
#include <dp/culling/ResultBitSet.h>
#include <dp/util/FrameProfiler.h>

#if defined(DP_ARCH_X86_64)
  #define SSE
#endif

#if defined(SSE)
#include <dp/math/sse/Vecnt.h>
#include <dp/math/sse/Matmnt.h>
static bool useSSE = true;
#else
static bool useSSE = false;
#endif

#if defined(DP_ARCH_ARM_32)
#define NEON
#endif

#if defined(NEON)
#include <dp/math/neon/Vecnt.h>
#include <dp/math/neon/Matmnt.h>
static bool useNEON = true;
#else
static bool useNEON = false;
#endif

namespace dp
{
  namespace culling
  {
    namespace cpu
    {

      namespace {

        struct OBB
        {
          dp::math::Vec4f point;
          dp::math::Vec4f ex;
          dp::math::Vec4f ey;
          dp::math::Vec4f ez;
        };

        /************************************************************************/
        /* GroupCPU                                                             */
        /* This group stores the cached OBB for each object                     */
        /************************************************************************/
        class GroupCPU : public GroupBitSet
        {
        public:
          GroupCPU();

          void updateOBBs();

          std::vector<OBB> const & getOBBs() const;
      
        private:
          std::vector<OBB> m_obbs;
          size_t m_objectIncarnationOBB;
        };

        GroupCPU::GroupCPU()
          : GroupBitSet()
          , m_objectIncarnationOBB( m_objectIncarnation - 1)
        {
        }

        std::vector<OBB> const & GroupCPU::getOBBs() const
        {
          return m_obbs;
        }

        void GroupCPU::updateOBBs()
        {
          m_obbDirty |= (m_objectIncarnationOBB != m_objectIncarnation);

          if ( m_obbDirty )
          {
            m_obbs.resize( m_objects.size() );

            char const* basePtr = reinterpret_cast<char const*>( getMatrices() );
            size_t matricesStride = getMatricesStride();

            for ( size_t index = 0; index < m_objects.size();++index )
            {
              OBB &obb = m_obbs[index];
              ObjectBitSetHandle const & objectImpl = getObject( index );
              dp::math::Mat44f const & modelView = reinterpret_cast<dp::math::Mat44f const &>(*(basePtr + objectImpl->getTransformIndex() * matricesStride) );

              obb.point = m_objects[index]->getLowerLeft() * modelView;

              dp::math::Vec4f const & extent = m_objects[index]->getExtent();

#if defined(SSE)
              if ( useSSE )
              {
                reinterpret_cast<dp::math::sse::Vec4f&>(obb.ex) = extent[0] * reinterpret_cast<dp::math::sse::Vec4f const&>(modelView[0]);
                reinterpret_cast<dp::math::sse::Vec4f&>(obb.ey) = extent[1] * reinterpret_cast<dp::math::sse::Vec4f const&>(modelView[1]);
                reinterpret_cast<dp::math::sse::Vec4f&>(obb.ez) = extent[2] * reinterpret_cast<dp::math::sse::Vec4f const&>(modelView[2]);
              }
              else
#elif defined(NEON)
                if ( useNEON )
                {
                  reinterpret_cast<dp::math::neon::Vec4f&>(obb.ex) = extent[0] * reinterpret_cast<dp::math::neon::Vec4f const&>(modelView[0]);
                  reinterpret_cast<dp::math::neon::Vec4f&>(obb.ey) = extent[1] * reinterpret_cast<dp::math::neon::Vec4f const&>(modelView[1]);
                  reinterpret_cast<dp::math::neon::Vec4f&>(obb.ez) = extent[2] * reinterpret_cast<dp::math::neon::Vec4f const&>(modelView[2]);
                }
                else
#endif
              {
                obb.ex = extent[0] * modelView[0];
                obb.ey = extent[1] * modelView[1];
                obb.ez = extent[2] * modelView[2];
              }
            }

            m_objectIncarnationOBB = m_objectIncarnation;
            m_obbDirty = false;
          }
        }

      } // namespace anonymous

      typedef dp::util::SmartPtr<GroupCPU> GroupCPUHandle;

      /************************************************************************/
      /* ManagerImpl                                                          */
      /************************************************************************/

      Manager* Manager::create()
      {
        return new ManagerImpl;
      }

      ManagerImpl::ManagerImpl()
      {
      }

      ManagerImpl::~ManagerImpl()
      {

      }

      ObjectHandle ManagerImpl::objectCreate( const dp::util::SmartRCObject& userData )
      {
        return new ObjectBitSet( userData );
      }

      GroupHandle ManagerImpl::groupCreate()
      {
        return new GroupCPU();
      }

      ResultHandle ManagerImpl::groupCreateResult( GroupHandle const& group )
      {
        const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>(group);

        return new ResultBitSet( groupImpl );
      }

      inline void determineCullFlags( const dp::math::Vec4f &p, unsigned int & cfo, unsigned int & cfa )
      {
        unsigned int cf = 0;

        if ( p[0] <= -p[3] )
        {
          cf |= 0x01;
        }
        else if ( p[3] <= p[0] )
        {
          cf |= 0x02;
        }
        if ( p[1] <= -p[3] )
        {
          cf |= 0x04;
        }
        else if ( p[3] <= p[1] )
        {
          cf |= 0x08;
        }
        if ( p[2] <= -p[3] )
        {
          cf |= 0x10;
        }
        else if ( p[3] <= p[2] )
        {
          cf |= 0x20;
        }
        cfo |= cf;
        cfa &= cf;
      }

      inline bool isVisible( const dp::math::Mat44f& projection, const dp::math::Mat44f &modelView, const dp::math::Vec4f &lower, const dp::math::Vec4f &extent)
      {
        unsigned int cfo = 0;
        unsigned int cfa = ~0;

        dp::math::Vec4f vectors[8];
        vectors[0] = (lower * modelView) * projection;

        dp::math::Vec4f x( extent[0] * modelView[0] );
        dp::math::Vec4f y( extent[1] * modelView[1] );
        dp::math::Vec4f z( extent[2] * modelView[2] );

        x = x * projection;
        y = y * projection;
        z = z * projection;

        vectors[1] = vectors[0] + x;
        vectors[2] = vectors[0] + y;
        vectors[3] = vectors[1] + y;
        vectors[4] = vectors[0] + z;
        vectors[5] = vectors[1] + z;
        vectors[6] = vectors[2] + z;
        vectors[7] = vectors[3] + z;

        for ( unsigned int i = 0;i < 8; ++i )
        {
          determineCullFlags( vectors[i], cfo, cfa );
        }

        return !cfo || !cfa;
      }

      inline bool isVisible( const dp::math::Mat44f& projection, OBB const & obb)
      {
        unsigned int cfo = 0;
        unsigned int cfa = ~0;

        dp::math::Vec4f vectors[8];
        vectors[0] = obb.point * projection;

        dp::math::Vec4f x = obb.ex * projection;
        dp::math::Vec4f y = obb.ey * projection;
        dp::math::Vec4f z = obb.ez * projection;

        vectors[1] = vectors[0] + x;
        vectors[2] = vectors[0] + y;
        vectors[3] = vectors[1] + y;
        vectors[4] = vectors[0] + z;
        vectors[5] = vectors[1] + z;
        vectors[6] = vectors[2] + z;
        vectors[7] = vectors[3] + z;

        for ( unsigned int i = 0;i < 8; ++i )
        {
          determineCullFlags( vectors[i], cfo, cfa );
        }

        return !cfo || !cfa;
      }

#if defined(SSE)
      inline void determineCullFlagsSSE( const dp::math::sse::Vec4f& point, unsigned int &cfa )
      {
        __m128 homogen = _mm_shuffle_ps( point.sse(), point.sse(), _MM_SHUFFLE( 3,3,3,3 ) );

        // compare each component point with the w coordinate and generate the corresponding bitmask
        unsigned int mask = _mm_movemask_ps( _mm_cmpgt_ps( point.sse(), homogen ) ) << 4;

        // IEEE 754 specifies that the highest bit of a float is the sign bit. Toggle it to get the negative value.
        __m128 signBits =  _mm_castsi128_ps(_mm_set1_epi32( 0x80000000 ));
        __m128 negHomogen = _mm_xor_ps( homogen, signBits );

        mask |= _mm_movemask_ps( _mm_cmpgt_ps( negHomogen, point.sse() ) );

        cfa &= mask;
      }

      inline bool isVisibleSSE( const dp::math::sse::Mat44f& projection, const dp::math::sse::Mat44f &modelView, const dp::math::sse::Vec4f &lower, const dp::math::sse::Vec4f &extent)
      {
        unsigned int cfa = ~0;

        dp::math::sse::Vec4f v = (lower * modelView) * projection;

        determineCullFlagsSSE( v, cfa); // v

        // cfa != 0 -> trivial out.
        // check for trivial out until each plane has one vertex which is not trivial out.
        if ( cfa )
        {
          // Compute x,z,y only if the first vertex is trivial out and the other vertices have to be checked.
          // Compute all of them at once to avoid redundant loads later on.
          dp::math::sse::Vec4f x( (modelView[0] * extent[0] ) * projection );
          dp::math::sse::Vec4f y( (modelView[1] * extent[1] ) * projection );
          dp::math::sse::Vec4f z( (modelView[2] * extent[2] ) * projection );
          v += x; determineCullFlagsSSE( v, cfa); // v + x
          if ( cfa ) 
          {
            v += y; determineCullFlagsSSE( v, cfa); // v + x + y
            if ( cfa )
            {
              v -= x; determineCullFlagsSSE( v, cfa); // v + y
              if ( cfa )
              {
                v += z; determineCullFlagsSSE( v, cfa); // v + y + z
                if ( cfa )
                {
                  v += x; determineCullFlagsSSE( v, cfa); // v + x + y + z
                  if ( cfa )
                  {
                    v -= y; determineCullFlagsSSE( v, cfa); // v + x + z
                    if ( cfa )
                    {
                      v -= x; determineCullFlagsSSE( v, cfa); // v + z
                    }
                  }
                }
              }
            }
          }
        }

        return !cfa;
      }

      inline bool isVisibleSSE( dp::math::sse::Mat44f const & projection, OBB const & obb )
      {
        unsigned int cfa = ~0;

        dp::math::sse::Vec4f v = reinterpret_cast<dp::math::sse::Vec4f const &>(obb.point) * projection;

        determineCullFlagsSSE( v, cfa); // v

        // cfa != 0 -> trivial out.
        // check for trivial out until each plane has one vertex which is not trivial out.
        if ( cfa )
        {
          // Compute x,z,y only if the first vertex is trivial out and the other vertices have to be checked.
          // Compute all of them at once to avoid redundant loads later on.
          dp::math::sse::Vec4f x( reinterpret_cast<dp::math::sse::Vec4f const &>(obb.ex) * projection );
          dp::math::sse::Vec4f y( reinterpret_cast<dp::math::sse::Vec4f const &>(obb.ey) * projection );
          dp::math::sse::Vec4f z( reinterpret_cast<dp::math::sse::Vec4f const &>(obb.ez) * projection );

          v += x; determineCullFlagsSSE( v, cfa); // v + x
          v += y; determineCullFlagsSSE( v, cfa); // v + x + y
          v -= x; determineCullFlagsSSE( v, cfa); // v + y
          v += z; determineCullFlagsSSE( v, cfa); // v + y + z
          v += x; determineCullFlagsSSE( v, cfa); // v + x + y + z
          v -= y; determineCullFlagsSSE( v, cfa); // v + x + z
          v -= x; determineCullFlagsSSE( v, cfa); // v + z
        }

        return !cfa;
      }
#endif

#if defined(NEON)

      inline void determineCullFlagsNEON( const dp::math::neon::Vec4f &p, unsigned int & cfa )
      {
        unsigned int cf = 0;

        if ( p[0] <= -p[3] )
        {
          cf |= 0x01;
        }
        else if ( p[3] <= p[0] )
        {
          cf |= 0x02;
        }
        if ( p[1] <= -p[3] )
        {
          cf |= 0x04;
        }
        else if ( p[3] <= p[1] )
        {
          cf |= 0x08;
        }
        if ( p[2] <= -p[3] )
        {
          cf |= 0x10;
        }
        else if ( p[3] <= p[2] )
        {
          cf |= 0x20;
        }
        cfa &= cf;
      }

      inline bool isVisibleNEON( const dp::math::neon::Mat44f& projection, const dp::math::neon::Mat44f &modelView, const dp::math::neon::Vec4f &lower, const dp::math::neon::Vec4f &extent)
      {
        unsigned int cfa = ~0;

        dp::math::neon::Vec4f v = (lower * modelView) * projection;

        determineCullFlagsNEON( v, cfa); // v

        // cfa != 0 -> trivial out.
        // check for trivial out until each plane has one vertex which is not trivial out.
        if ( cfa )
        {
          // Compute x,z,y only if the first vertex is trivial out and the other vertices have to be checked.
          // Compute all of them at once to avoid redundant loads later on.
          dp::math::neon::Vec4f x( (modelView[0] * extent[0] ) * projection );
          dp::math::neon::Vec4f y( (modelView[1] * extent[1] ) * projection );
          dp::math::neon::Vec4f z( (modelView[2] * extent[2] ) * projection );
          v += x; determineCullFlagsNEON( v, cfa); // v + x
          if ( cfa )
          {
            v += y; determineCullFlagsNEON( v, cfa); // v + x + y
            if ( cfa )
            {
              v -= x; determineCullFlagsNEON( v, cfa); // v + y
              if ( cfa )
              {
                v += z; determineCullFlagsNEON( v, cfa); // v + y + z
                if ( cfa )
                {
                  v += x; determineCullFlagsNEON( v, cfa); // v + x + y + z
                  if ( cfa )
                  {
                    v -= y; determineCullFlagsNEON( v, cfa); // v + x + z
                    if ( cfa )
                    {
                      v -= x; determineCullFlagsNEON( v, cfa); // v + z
                    }
                  }
                }
              }
            }
          }
        }
        return !cfa;

      }
#endif

      void ManagerImpl::cull( GroupHandle const& group, ResultHandle const& result, const dp::math::Mat44f& viewProjection )
      {
        dp::util::ProfileEntry p("cull");
        GroupCPUHandle const & groupImpl = dp::util::smart_cast<GroupCPU>(group);
        ResultBitSetHandle const& resultImpl = dp::util::smart_cast<ResultBitSet>(result);

        groupImpl->updateOBBs();
        std::vector<OBB> const &obbs = groupImpl->getOBBs();

        // TODO this is an allocation which is potential slow. Keep memory allocated per group?
        DP_STATIC_ASSERT( sizeof( dp::util::BitArray::BitStorageType) % sizeof(dp::util::Uint32) == 0 );
        dp::util::BitArray visible( groupImpl->getObjectCount() );

        char const* basePtr = reinterpret_cast<char const*>(groupImpl->getMatrices());
        size_t matricesStride = groupImpl->getMatricesStride();
        size_t const count = groupImpl->getObjectCount();

#if defined(SSE)
        if ( useSSE )
        {
          dp::math::sse::Mat44f vp = *reinterpret_cast<dp::math::sse::Mat44f const*>(&viewProjection);

          for ( int index = 0;index < count; ++index )
          {
            visible.setBit( index, isVisibleSSE( vp, obbs[index] ) );
          }
        }
        else
#elif defined(NEON)
        if ( useNEON )
        {
          dp::math::neon::Mat44f vp = *reinterpret_cast<dp::math::neon::Mat44f const*>(&viewProjection);

          for ( int index = 0;index < count; ++index )
          {
            const ObjectBitSetHandle& objectImpl = groupImpl->getObject( index );
            const dp::math::neon::Mat44f &modelView = reinterpret_cast<const dp::math::neon::Mat44f&>(*(basePtr + objectImpl->getTransformIndex() * matricesStride) );
            visible.setBit( index, isVisibleNEON( vp, modelView, *reinterpret_cast<dp::math::neon::Vec4f const*>(&objectImpl->getLowerLeft())
              , *reinterpret_cast<dp::math::neon::Vec4f const*>(&objectImpl->getExtent()) ) );
          }
        }
        else
#endif
        {
//#pragma omp parallel for
          for ( size_t index = 0;index < count; ++index )
          {
            visible.setBit( index, isVisible( viewProjection, obbs[index] ) );
          }
        }

        resultImpl->updateChanged( reinterpret_cast<dp::util::Uint32 const*>( visible.getBits() ) );
      }

    } // namespace cpu
  } // namespace culling
} // namespace dp
