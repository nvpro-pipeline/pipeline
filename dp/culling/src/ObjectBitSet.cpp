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
#include <dp/culling/GroupBitSet.h>

namespace dp
{
  namespace culling
  {
      ObjectBitSetSharedPtr ObjectBitSet::create( PayloadSharedPtr const& userData )
      {
        return( std::shared_ptr<ObjectBitSet>( new ObjectBitSet( userData ) ) );
      }

      ObjectBitSet::ObjectBitSet( PayloadSharedPtr const& userData )
      : m_userData( userData )
      , m_transformIndex( ~0 )
      , m_groupIndex( ~0 )
      , m_group( nullptr )
      {
      };

      void ObjectBitSet::setGroup( GroupBitSetSharedPtr const & group )
      {
        m_group = group.getWeakPtr();
      }

      GroupBitSetSharedPtr ObjectBitSet::getGroup() const
      {
        return( m_group ? GroupSharedPtr( m_group->shared_from_this() ).staticCast<GroupBitSet>() : GroupBitSetSharedPtr::null );
      }

  } // namespace culling
} // namespace dp