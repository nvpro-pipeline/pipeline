// Copyright NVIDIA Corporation 2002-2015
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
#include <list>
#include <map>
#include <set>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      //! Traverser to convert primitives of type PRIMITIVE_TRIANGLES or PRIMITIVE_QUADS to Primitives of type PRIMITIVE_TRIANGLE_STRIPS or PRIMITIVE_QUAD_STRIPS, respectively.
      /** It is strongly recommended to use a VertexUnifyTraverser before using this StrippingTraverser, because it 
        * works only on indices. */
      class StrippingTraverser : public ExclusiveTraverser
      {
        public:
          //! Constructor
          DP_SG_ALGORITHM_API StrippingTraverser(void);

          //! Destructor
          DP_SG_ALGORITHM_API virtual ~StrippingTraverser(void);

        protected:
          DP_SG_ALGORITHM_API virtual void handleGeoNode( dp::sg::core::GeoNode * p );

          //! Convert each Primitive of type PRIMITIVE_QUADS or PRIMITIVE_TRIANGLES to PRIMITIVE_QUAD_STRIP and PRIMITIVE_TRIANGLE_STRIP, respectively.
          DP_SG_ALGORITHM_API virtual void handlePrimitive( dp::sg::core::Primitive * p );

          /*! \brief Test whether this Object should be optimized
           *  \param p A pointer to the constant dp::sg::core::Object to test. */
          DP_SG_ALGORITHM_API virtual bool optimizationAllowed( dp::sg::core::ObjectSharedPtr const& p );

        private:
          void changeToStrips( dp::sg::core::Primitive * p );

        private:
          dp::sg::core::PrimitiveSharedPtr  m_strip;
      };

    } // namespace algorithm
  } // namespace sg
} // namespace dp
