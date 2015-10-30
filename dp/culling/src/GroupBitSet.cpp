// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/culling/GroupBitSet.h>
#include <stdexcept>

namespace
{
  struct ShaderObject
  {
    // TODO pack matrix into 4th component of extent!
    //dp::math::Mat44f matrix;
    uint32_t        matrix;
    uint32_t        pad0;
    uint32_t        pad1;
    uint32_t        pad2;
    dp::math::Vec4f lowerLeft;
    dp::math::Vec4f extent;
  };
}

namespace dp
{
  namespace culling
  {
      GroupBitSetSharedPtr GroupBitSet::create()
      {
        return( std::shared_ptr<GroupBitSet>( new GroupBitSet() ) );
      }

      GroupBitSet::GroupBitSet()
        : m_matrices( nullptr )
        , m_matricesCount( 0 )
        , m_matricesStride( 0 )
        , m_inputChanged( true )
        , m_matricesChanged( true )
        , m_objectIncarnation( 0 )
        , m_dirtyMatrices( 0 )
        , m_boundingBoxDirty( true )
        , m_obbDirty( true )
      {
      }

      GroupBitSet::~GroupBitSet()
      {
        // reset the groupIndex for all objects within this group so that they can be reused in another group
        for ( size_t index = 0;index < m_objects.size(); ++index )
        {
          m_objects[index]->setGroupIndex(~0);
        }
      }

      void GroupBitSet::addObject( const ObjectBitSetSharedPtr& object )
      {
        if ( object->getGroupIndex() == ~0 )
        {
          object->setGroupIndex( m_objects.size() );
          m_objects.push_back(object);
          m_inputChanged = true;
          ++m_objectIncarnation;
          m_boundingBoxDirty = true;
          m_obbDirty = true;
        }
        else
        {
          throw std::runtime_error( "object already belongs to a different group" );
        }
      }

      void GroupBitSet::removeObject( ObjectBitSetSharedPtr const & object )
      {
        size_t oldGroupIndex = object->getGroupIndex();
        if ( oldGroupIndex < m_objects.size() && m_objects[oldGroupIndex] == object )
        {
          // move object from the end to the freed location
          m_objects[ oldGroupIndex ] = m_objects.back();
          m_objects[ oldGroupIndex ]->setGroupIndex(oldGroupIndex);
          object->setGroupIndex(~0);
          object->setGroup( GroupBitSetSharedPtr() );

          // remove the last element
          m_objects.pop_back();

          // notify the move from the last element to the index of the removed element
          notify( Event( m_objects.size(), oldGroupIndex) );

          m_inputChanged = true;
          m_boundingBoxDirty = true;
          m_obbDirty = true;
          ++m_objectIncarnation;
        }
        else
        {
          throw std::runtime_error( "object does not belong to this group" );
        }
      }

      void GroupBitSet::setMatrices( const void* matrices, size_t numberOfMatrices, size_t stride )
      {
        if (  matrices != m_matrices || numberOfMatrices != m_matricesCount || stride != m_matricesStride )
        {
          m_matrices = matrices;
          m_matricesCount = numberOfMatrices;
          m_matricesStride = stride;
          m_matricesChanged = true;

          m_dirtyMatrices.resize( numberOfMatrices );
          m_dirtyMatrices.fill();
          m_boundingBoxDirty = true;
          m_obbDirty = true;
        }
      }

      void GroupBitSet::clearObjects()
      {
        m_objects.clear();
        m_boundingBoxDirty = true;
        m_obbDirty = true;
      }

  } // namespace culling
} // namespace dp
