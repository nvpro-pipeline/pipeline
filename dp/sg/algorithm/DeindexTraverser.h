// Copyright NVIDIA Corporation 2002-2011
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

#include <set>
#include <vector>

#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/Traverser.h>
#include <dp/sg/core/nvsg.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Primitive.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {
      //! Traverser to convert indexed Primitive nodes to non-indexed Primitive nodes.
      // \note Use with care! This traverser expands VertexAttributeSets shared by multiple Primitives unconditionally.
      // The resulting Primitives will have their own non-indexed VertexAttributeSet after the DeindexTraverser has been applied.
      // Stripped primitives with primitive restart indices cannot be expanded to single Primitives, that is why
      // the DeindexTraverser::doApply() function will run a DestrippingTraverser over the scene first.
      // Skinned primitives will retain their IndexSet because skinning is only implemented for indexed primitives.
      class DeindexTraverser : public ExclusiveTraverser
      {
        public:
          //! Constructor
          DP_SG_ALGORITHM_API DeindexTraverser(void);

        protected:
          //! Protected destructor to prevent instantiation of a DeindexTraverser.
          DP_SG_ALGORITHM_API virtual ~DeindexTraverser(void);

          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );
          DP_SG_ALGORITHM_API virtual void handlePrimitive( dp::sg::core::Primitive *p );
      };

    } // namespace algorithm
  } // namespace sp
} // namespace dp
