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


#pragma once

#include <dp/culling/GroupBitSet.h>
#include <boost/scoped_array.hpp>

namespace dp
{
  namespace culling
  {

    class ResultBitSet : public Result, public dp::util::Observer
    {
    public:
      DP_CULLING_API ResultBitSet( GroupBitSetHandle const& parentGroup );
      DP_CULLING_API ~ResultBitSet();

      DP_CULLING_API std::vector<ObjectHandle> const & getChangedObjects() const;

      /** \brief Update the group of changed objects.
          \param visibility is a bitmask where the visibility for object i is specified in bit i
      **/
      DP_CULLING_API void updateChanged( dp::util::Uint32 const* visibility );

      DP_CULLING_API virtual void onNotify( dp::util::Event const& event, dp::util::Payload* payload );
      DP_CULLING_API virtual void onDestroyed( dp::util::Subject const& subject, dp::util::Payload* payload );

      bool isVisible( ObjectBitSetHandle const & object );

    private:
      GroupBitSetHandle m_groupParent;
      std::vector<ObjectHandle> m_changedObjects;

      size_t m_objectIncarnation;
      bool   m_groupChanged;

      dp::util::BitArray m_results;
   };

    inline bool ResultBitSet::isVisible( ObjectBitSetHandle const & object )
    {
      size_t groupIndex = object->getGroupIndex();
      DP_ASSERT( groupIndex != ~0 );
      // DP_ASSERT( m_groupParent->m_objects[groupIndex] == objectImpl ); befriend GroupBitSet with ResultBitSet?

      return (groupIndex < m_results.getSize()) ? m_results.getBit( groupIndex ) : true;
    }

   typedef dp::util::SmartPtr<ResultBitSet> ResultBitSetHandle;

  } // namespace culling
} // namespace dp

