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

#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/Traverser.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {
      /*! \brief Traverser to destripify stripped primitives.
       *  \par Namespace: dp::sg::algorithm
       *  \remarks This Traverser destripifies Primitives of type PrimitiveType::TRIANGLE_FAN and PrimitiveType::TRIANGLE_STRIP
       *  to Primitives of type PrimitiveType::TRIANGLES, Primitives of type PrimitiveType::QUAD_STRIP to Primitives of type
       *  PrimitiveType::QUADS, and Primitives of type PrimitiveType::STRIPS and PrimitiveType::LINE_LOOP to Primitives of type
       *  PrimitiveType::LINES.\n
       *  Using good TriStrips usually is more efficient than using Triangles, but often it is more efficient to use
       *  Triangles instead of badly stripped TriStrips. That is, it might be worth to try to destrip all the TriStrips
       *  to Triangles. The same holds for the other primitive types.
       *  \sa OptimizeTraverser */
      class DestrippingTraverser : public ExclusiveTraverser
      {
        public:
          /*! \brief Default Constructor
           *  \remarks A DestrippingTraverser potentially modifies the tree of the scene. */
          DP_SG_ALGORITHM_API DestrippingTraverser(void);
          DP_SG_ALGORITHM_API virtual ~DestrippingTraverser(void);

        protected:
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

          DP_SG_ALGORITHM_API virtual void handleGeoNode( dp::sg::core::GeoNode * p );

          /*! \brief Change stripped Primitives to non-stripped Primitives
           *  \param p A pointer to the dp::sg::core::Primitive to handle. */
          DP_SG_ALGORITHM_API virtual void handlePrimitive( dp::sg::core::Primitive * p );

          /*! \brief Test whether this Object should be optimized
           *  \param p A pointer to the constant dp::sg::core::Object to test. */
          DP_SG_ALGORITHM_API virtual bool optimizationAllowed( dp::sg::core::ObjectSharedPtr const& p );

        private:
          dp::sg::core::PrimitiveSharedPtr                                    m_primitive;
          std::map<dp::sg::core::Primitive*,dp::sg::core::PrimitiveSharedPtr> m_primitiveMap;
      };

    } // namespace algorithm
  } // namespace sp
} // namespace dp
