// Copyright NVIDIA Corporation 2012
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


#include <dp/culling/Culling.h>
#include <dp/culling/inc/GroupImpl.h>
#include <dp/culling/inc/ObjectImpl.h>

namespace dp
{
  namespace culling
  {

    /************************************************************************/
    /* ResultImpl                                                           */
    /************************************************************************/
    class ResultImpl : public Result
    {
    public:
      ResultImpl();
      GroupImplHandle m_changed;
    };

    ResultImpl::ResultImpl()
    {
      m_changed = new GroupImpl;
    }
    
    typedef dp::util::SmartPtr<ResultImpl> ResultImplHandle;

    /************************************************************************/
    /* ManagerImpl                                                          */
    /************************************************************************/
    class ManagerImpl : public Manager
    {
    public:
      virtual ObjectHandle objectCreate( const dp::util::SmartRCObject& userData );
      virtual void objectSetBoundingBox( const ObjectHandle& object, const dp::math::Box3f& boundingBox );
      virtual void objectSetTransformIndex( const ObjectHandle& object, size_t index );
      virtual void objectSetUserData( const ObjectHandle& object, const dp::util::SmartRCObject& userData );
      virtual bool objectIsVisible( const ObjectHandle& object );

      virtual const dp::util::SmartRCObject& objectGetUserData( const ObjectHandle& object );

      virtual GroupHandle groupCreate();
      virtual void groupAddObject( const GroupHandle& group, const ObjectHandle& object );
      virtual ObjectHandle groupGetObject( const GroupHandle& group, size_t index );
      virtual void groupRemoveObject( const GroupHandle& group, const ObjectHandle& object );
      virtual size_t groupGetCount( const GroupHandle& group );
      virtual void groupSetMatrices( const GroupHandle& group, void const* matrices, size_t stride );

      virtual ResultHandle resultCreate() ;
      virtual GroupHandle resultGetChanged( const ResultHandle& result );

      virtual void cull( const GroupHandle& group, const ResultHandle& result, const dp::math::Mat44f& viewProjection );
    };

    Manager* Manager::create()
    {
      return new ManagerImpl;
    }

    GroupHandle ManagerImpl::groupCreate()
    {
      return new GroupImpl();
    }

    ObjectHandle ManagerImpl::objectCreate( const dp::util::SmartRCObject& userData )
    {
      return new ObjectImpl( userData );
    }

    void ManagerImpl::objectSetUserData( const ObjectHandle& object, const dp::util::SmartRCObject& userData )
    {
      const ObjectImplHandle objectImpl = dp::util::smart_cast<ObjectImpl>(object);
      objectImpl->m_userData = userData;
    }

    const dp::util::SmartRCObject& ManagerImpl::objectGetUserData( const ObjectHandle& object )
    {
      const ObjectImplHandle objectImpl = dp::util::smart_cast<ObjectImpl>(object);

      return objectImpl->m_userData;
    }

    void ManagerImpl::objectSetTransformIndex( const ObjectHandle& object, size_t index )
    {
      const ObjectImplHandle objectImpl = dp::util::smart_cast<ObjectImpl>(object);

      objectImpl->m_transformIndex = index;
    }

    bool ManagerImpl::objectIsVisible( const ObjectHandle& object )
    {
      const ObjectImplHandle objectImpl = dp::util::smart_cast<ObjectImpl>(object);

      return objectImpl->m_visible;
    }


    void ManagerImpl::objectSetBoundingBox( const ObjectHandle& object, const dp::math::Box3f& boundingBox )
    {
      const ObjectImplHandle objectImpl = dp::util::smart_cast<ObjectImpl>(object);
      objectImpl->m_lowerLeft = dp::math::Vec4f(boundingBox.getLower(), 1.0f );
      objectImpl->m_extends = dp::math::Vec4f( boundingBox.getSize(), 0.0f );
    }


    void ManagerImpl::groupAddObject( const GroupHandle& group, const ObjectHandle& object )
    {
      const GroupImplHandle groupImpl = dp::util::smart_cast<GroupImpl>(group);
      const ObjectImplHandle objectImpl = dp::util::smart_cast<ObjectImpl>(object);

      groupImpl->addObject( objectImpl );
    }

    ObjectHandle ManagerImpl::groupGetObject( const GroupHandle& group, size_t index )
    {
      const GroupImplHandle& groupImpl = dp::util::smart_cast<GroupImpl>(group);

      return groupImpl->getObject( index );
    }

    void ManagerImpl::groupRemoveObject( const GroupHandle& group, const ObjectHandle& objectHandle )
    {

    }

    size_t ManagerImpl::groupGetCount( const GroupHandle& group )
    {
      const GroupImplHandle& groupImpl = dp::util::smart_cast<GroupImpl>(group);

      return groupImpl->getCount();
    }

    void ManagerImpl::groupSetMatrices( const GroupHandle& group, void const* matrices, size_t stride )
    {
      const GroupImplHandle& groupImpl = dp::util::smart_cast<GroupImpl>(group);

      groupImpl->setMatrices( matrices, stride );
    }


    ResultHandle ManagerImpl::resultCreate()
    {
      return new ResultImpl;
    }

    GroupHandle ManagerImpl::resultGetChanged( const ResultHandle& result )
    {
      const ResultImplHandle& resultImpl = dp::util::smart_cast<ResultImpl>(result);

      return resultImpl->m_changed;
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

    inline bool isVisible( const dp::math::Mat44f& projection, const dp::math::Mat44f &modelView, const dp::math::Vec4f &lower, const dp::math::Vec4f &extends)
    {
      unsigned int cfo = 0;
      unsigned int cfa = ~0;

      dp::math::Vec4f vectors[8];
      vectors[0] = (lower * modelView) * projection;

      dp::math::Vec4f x( extends[0] * modelView.getPtr()[0], extends[0] * modelView.getPtr()[1], extends[0] * modelView.getPtr()[2], extends[0] * modelView.getPtr()[3] );
      dp::math::Vec4f y( extends[1] * modelView.getPtr()[4], extends[1] * modelView.getPtr()[5], extends[1] * modelView.getPtr()[6], extends[1] * modelView.getPtr()[7] );
      dp::math::Vec4f z( extends[2] * modelView.getPtr()[8], extends[2] * modelView.getPtr()[9], extends[2] * modelView.getPtr()[10], extends[2] * modelView.getPtr()[11] );

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

      return( !cfo ? true : cfa ? false : true );
    }

    void ManagerImpl::cull( const GroupHandle& group, const ResultHandle& result, const dp::math::Mat44f& viewProjection )
    {
      const GroupImplHandle& groupImpl = dp::util::smart_cast<GroupImpl>(group);
      const ResultImplHandle& resultImpl = dp::util::smart_cast<ResultImpl>(result);

      dp::math::Mat44f vp = viewProjection;

      resultImpl->m_changed->clear();

      char const* basePtr = reinterpret_cast<char const*>(groupImpl->m_matrices);
      size_t const count = groupImpl->getCount();
      for ( size_t index = 0;index < count; ++index )
      {
        const ObjectImplHandle& objectImpl = groupImpl->getObject( index );
        const dp::math::Mat44f &modelView = reinterpret_cast<const dp::math::Mat44f&>(*(basePtr + objectImpl->m_transformIndex * groupImpl->m_matricesStride));
        bool visible = isVisible( viewProjection, modelView, objectImpl->m_lowerLeft, objectImpl->m_extends );
        if( visible != objectImpl->m_visible )
        {
          resultImpl->m_changed->addObject( groupImpl->getObject( index ) );
          objectImpl->m_visible = visible;
        }
      }

    }

  } // namespace culling
} // namespace dp
