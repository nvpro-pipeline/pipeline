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


#include <dp/culling/ObjectBitSet.h>
#include <dp/culling/ResultBitSet.h>
#include <dp/culling/ManagerBitSet.h>
#include <dp/culling/GroupBitSet.h>
#include <dp/util/BitArray.h>
#include <boost/scoped_array.hpp>

#include <dp/util/FrameProfiler.h>

#include <limits>

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

    ManagerBitSet::ManagerBitSet()
    {

    }
    ManagerBitSet::~ManagerBitSet()
    {

    }

    void ManagerBitSet::objectSetUserData( const ObjectHandle& object, const dp::util::SmartRCObject& userData )
    {
      const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>(object);
      objectImpl->setUserData( userData );
    }

    const dp::util::SmartRCObject& ManagerBitSet::objectGetUserData( const ObjectHandle& object )
    {
      const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>(object);

      return objectImpl->getUserData();
    }

    void ManagerBitSet::objectSetTransformIndex( const ObjectHandle& object, size_t index )
    {
      const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>(object);

      objectImpl->setTransformIndex( index );
      if ( objectImpl->getGroup() )
      {
        objectImpl->getGroup()->setOBBDirty( true );
      }
    }

    void ManagerBitSet::objectSetBoundingBox( const ObjectHandle& object, const dp::math::Box3f& boundingBox )
    {
      const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>(object);
      objectImpl->setLowerLeft( dp::math::Vec4f(boundingBox.getLower(), 1.0f ) );
      objectImpl->setExtent( dp::math::Vec4f( boundingBox.getSize(), 0.0f ) );

      if ( objectImpl->getGroup() )
      {
        objectImpl->getGroup()->setOBBDirty( true );
      }
    }

    void ManagerBitSet::groupAddObject( const GroupHandle& group, const ObjectHandle& object )
    {
      const GroupBitSetHandle groupImpl = dp::util::smart_cast<GroupBitSet>(group);
      const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>(object);

      groupImpl->addObject( objectImpl );
    }

    ObjectHandle ManagerBitSet::groupGetObject( const GroupHandle& group, size_t index )
    {
      const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>(group);

      return groupImpl->getObject( index );
    }

    void ManagerBitSet::groupRemoveObject( const GroupHandle& group, const ObjectHandle& objectHandle )
    {
      const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>(group);
      const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>(objectHandle);

      groupImpl->removeObject( objectImpl );
    }

    size_t ManagerBitSet::groupGetCount( const GroupHandle& group )
    {
      const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>(group);

      return groupImpl->getObjectCount();
    }

    void ManagerBitSet::groupSetMatrices( const GroupHandle& group, void const* matrices, size_t numberOfMatrices, size_t stride )
    {
      const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>(group);

      groupImpl->setMatrices( matrices, numberOfMatrices, stride );
    }

    void ManagerBitSet::groupMatrixChanged( GroupHandle const& group, size_t index )
    {
      const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>(group);

      groupImpl->markMatrixDirty( index );
    }

    std::vector<ObjectHandle> const & ManagerBitSet::resultGetChanged( const ResultHandle& result )
    {
      const ResultBitSetHandle& resultImpl = dp::util::smart_cast<ResultBitSet>(result);

      return resultImpl->getChangedObjects();
    }

    bool ManagerBitSet::resultObjectIsVisible( ResultHandle const& result, ObjectHandle const& object )
    {
      ResultBitSetHandle const & resultImpl = dp::util::smart_cast<ResultBitSet>(result);
      ObjectBitSetHandle const & objectImpl = dp::util::smart_cast<ObjectBitSet>(object);
      
      return resultImpl->isVisible( objectImpl );
    }

    dp::math::Box3f ManagerBitSet::getBoundingBox( const GroupHandle& group ) const
    {
      dp::util::ProfileEntry p("cull::getBoundingBox");

      const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>( group );
      if ( groupImpl->isBoundingBoxDirty() )
      {
        groupImpl->setBoundingBox( calculateBoundingBox( group ) );
        groupImpl->setBoundingBoxDirty( false );
      }
      return groupImpl->getBoundingBox();
    }

    dp::math::Box3f ManagerBitSet::calculateBoundingBox( const GroupHandle& group ) const
    {
#if defined(SSE)
      if ( useSSE )
      {
        const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>( group );

        __m128 minValue = _mm_set1_ps( std::numeric_limits<float>::signaling_NaN() );
        __m128 maxValue = _mm_set1_ps( std::numeric_limits<float>::signaling_NaN() );

        char const* basePtr = reinterpret_cast<char const*>(groupImpl->getMatrices());
        for ( size_t index = 0;index < groupImpl->getObjectCount(); ++index )
        {
          const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>( groupImpl->getObject( index ) );
          dp::math::sse::Mat44f const& modelView = *reinterpret_cast<dp::math::sse::Mat44f const*>(basePtr + objectImpl->getTransformIndex() * groupImpl->getMatricesStride());
          dp::math::Vec4f const& extent = objectImpl->getExtent();

          dp::math::sse::Vec4f vectors[8];
          vectors[0] = *reinterpret_cast<dp::math::sse::Vec4f const*>(&objectImpl->getLowerLeft()) * modelView;

          dp::math::sse::Vec4f x( extent[0] * modelView[0] );
          dp::math::sse::Vec4f y( extent[1] * modelView[1] );
          dp::math::sse::Vec4f z( extent[2] * modelView[2] );

          vectors[1] = vectors[0] + x;
          vectors[2] = vectors[0] + y;
          vectors[3] = vectors[1] + y;
          vectors[4] = vectors[0] + z;
          vectors[5] = vectors[1] + z;
          vectors[6] = vectors[2] + z;
          vectors[7] = vectors[3] + z;

          for ( unsigned int i = 0;i < 8; ++i )
          {
            minValue = _mm_min_ps( minValue, vectors[i].sse() );
            maxValue = _mm_max_ps( maxValue, vectors[i].sse() );
          }
        }

        dp::math::Vec3f minVec, maxVec;
        _MM_EXTRACT_FLOAT( minVec[0], minValue, 0); 
        _MM_EXTRACT_FLOAT( minVec[1], minValue, 1); 
        _MM_EXTRACT_FLOAT( minVec[2], minValue, 2); 

        _MM_EXTRACT_FLOAT( maxVec[0], maxValue, 0); 
        _MM_EXTRACT_FLOAT( maxVec[1], maxValue, 1); 
        _MM_EXTRACT_FLOAT( maxVec[2], maxValue, 2); 

        return dp::math::Box3f( minVec, maxVec );
      }
      else
#elif defined(NEON)
        if ( useNEON )
        {
          const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>( group );

          float32x4_t minValue = vdupq_n_f32( std::numeric_limits<float>::max() );
          float32x4_t maxValue = vdupq_n_f32( -std::numeric_limits<float>::max() );

          char const* basePtr = reinterpret_cast<char const*>(groupImpl->getMatrices());
          for ( size_t index = 0;index < groupImpl->getObjectCount(); ++index )
          {
            const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>( groupImpl->getObject( index ) );
            dp::math::neon::Mat44f const& modelView = *reinterpret_cast<dp::math::neon::Mat44f const*>(basePtr + objectImpl->getTransformIndex() * groupImpl->getMatricesStride());
            dp::math::Vec4f const& extent = objectImpl->getExtent();

            dp::math::neon::Vec4f vectors[8];
            vectors[0] = *reinterpret_cast<dp::math::neon::Vec4f const*>(&objectImpl->getLowerLeft()) * modelView;

            dp::math::neon::Vec4f x( extent[0] * modelView[0] );
            dp::math::neon::Vec4f y( extent[1] * modelView[1] );
            dp::math::neon::Vec4f z( extent[2] * modelView[2] );

            vectors[1] = vectors[0] + x;
            vectors[2] = vectors[0] + y;
            vectors[3] = vectors[1] + y;
            vectors[4] = vectors[0] + z;
            vectors[5] = vectors[1] + z;
            vectors[6] = vectors[2] + z;
            vectors[7] = vectors[3] + z;

            for ( unsigned int i = 0;i < 8; ++i )
            {
              minValue = vminq_f32( minValue, vectors[i].neon() );
              maxValue = vmaxq_f32( maxValue, vectors[i].neon() );
            }

          }

          dp::math::Vec3f minVec, maxVec;

          vst1q_lane_f32( &minVec[0], minValue, 0);
          vst1q_lane_f32( &minVec[1], minValue, 1);
          vst1q_lane_f32( &minVec[2], minValue, 2);

          vst1q_lane_f32( &maxVec[0], maxValue, 0);
          vst1q_lane_f32( &maxVec[1], maxValue, 1);
          vst1q_lane_f32( &maxVec[2], maxValue, 2);

          return dp::math::Box3f( minVec, maxVec );
        }
        else

#endif
      // CPU fallback
      {
        const GroupBitSetHandle& groupImpl = dp::util::smart_cast<GroupBitSet>( group );

        dp::math::Box4f boundingBox;

        char const* basePtr = reinterpret_cast<char const*>(groupImpl->getMatrices());
        for ( size_t index = 0;index < groupImpl->getObjectCount(); ++index )
        {
          const ObjectBitSetHandle objectImpl = dp::util::smart_cast<ObjectBitSet>( groupImpl->getObject( index ) );
          dp::math::Mat44f const& modelView = reinterpret_cast<dp::math::Mat44f const&>(*(basePtr + objectImpl->getTransformIndex() * groupImpl->getMatricesStride()));
          dp::math::Vec4f const& extent = objectImpl->getExtent();

          dp::math::Vec4f vectors[8];
          vectors[0] = (objectImpl->getLowerLeft() * modelView);

          dp::math::Vec4f x( extent[0] * modelView.getPtr()[0], extent[0] * modelView.getPtr()[1], extent[0] * modelView.getPtr()[2], extent[0] * modelView.getPtr()[3] );
          dp::math::Vec4f y( extent[1] * modelView.getPtr()[4], extent[1] * modelView.getPtr()[5], extent[1] * modelView.getPtr()[6], extent[1] * modelView.getPtr()[7] );
          dp::math::Vec4f z( extent[2] * modelView.getPtr()[8], extent[2] * modelView.getPtr()[9], extent[2] * modelView.getPtr()[10], extent[2] * modelView.getPtr()[11] );

          vectors[1] = vectors[0] + x;
          vectors[2] = vectors[0] + y;
          vectors[3] = vectors[1] + y;
          vectors[4] = vectors[0] + z;
          vectors[5] = vectors[1] + z;
          vectors[6] = vectors[2] + z;
          vectors[7] = vectors[3] + z;

          for ( unsigned int i = 0;i < 8; ++i )
          {
            boundingBox.update( vectors[i] );
          }
        }

        dp::math::Vec4f lower = boundingBox.getLower();
        dp::math::Vec4f upper = boundingBox.getUpper();

        return dp::math::Box3f( dp::math::Vec3f( lower[0], lower[1], lower[2]), dp::math::Vec3f( upper[0], upper[1], upper[2]));
      }
    }

  } // namespace culling
} // namespace dp
