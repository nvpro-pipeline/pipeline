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
/** @file */

#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/core/BoundingVolumeObject.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Serves as base class for all tree nodes
        * \par Namespace: dp::sg::core
        */
      class Node : public BoundingVolumeObject
      {
        public:
          /*! \brief Destructs a Node. */
          DP_SG_CORE_API virtual ~Node();

          REFLECTION_INFO_API( DP_SG_CORE_API, Node );

        protected:
          /*! \brief Default-constructs a Node. */
          DP_SG_CORE_API Node();

          /*! \brief Constructs a Node as a copy of another Node. */
          DP_SG_CORE_API Node(const Node& rhs);

          /*! \brief Assigns new content from another Node. 
            * \param rhs Reference to a Node from which to assign the new content.
            * \return A reference to this object.
            * \remarks The assignment operator unreferences the old content before assigning the new content.
            * The new content will be a deep-copy of the content of right-hand-sided object. */
          DP_SG_CORE_API Node & operator=(const Node & rhs);
      };
  
    } // namespace core
  } // namespace sg
} // namespace dp
