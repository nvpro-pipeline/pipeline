// Copyright NVIDIA Corporation 2002-2005
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
/** \file */

#include <map>
#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/OptimizeTraverser.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      //! Traverser that replaces identity transforms by groups.
      class IdentityToGroupTraverser : public OptimizeTraverser
      {
        public:
          //! Constructor
          DP_SG_ALGORITHM_API IdentityToGroupTraverser( void );

          //! Destructor
          DP_SG_ALGORITHM_API virtual ~IdentityToGroupTraverser( void );

        protected:
          //! If the root node is an identity Transform, it is replaced by a Group.
          DP_SG_ALGORITHM_API virtual void postApply( const dp::sg::core::NodeSharedPtr & root );

          //! Replace any identity Transform of it's children to a Group.
          DP_SG_ALGORITHM_API virtual void handleBillboard( dp::sg::core::Billboard *p );

          //! Replace any identity Transform of it's children to a Group.
          DP_SG_ALGORITHM_API virtual void handleGroup( dp::sg::core::Group *p );

          //! Replace any identity Transform of it's children to a Group.
          DP_SG_ALGORITHM_API virtual void handleLOD( dp::sg::core::LOD *p );

          //! Replace any identity Transform of it's children to a Group.
          DP_SG_ALGORITHM_API virtual void handleSwitch( dp::sg::core::Switch *p );

          //! Replace any identity Transform of it's children to a Group.
          DP_SG_ALGORITHM_API virtual void handleTransform( dp::sg::core::Transform *p );

        private:
          dp::sg::core::GroupSharedPtr  createGroupFromTransform( const dp::sg::core::TransformSharedPtr & th );
          bool                  isTransformToReplace( const dp::sg::core::NodeSharedPtr & nh );
          void                  replaceTransforms( dp::sg::core::Group *p );

        private:
          std::set<const void *>  m_objects;      //!< A set of pointers to hold all objects already encountered.
      };

    } // namespace algorithm
  } // namespace sp
} // namespace dp
