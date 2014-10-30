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

#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/Traverser.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {
      class DestrippingTraverser;

      //! Traverser that can smooth some drawables.
      /** Currently, this SmoothTraverser supports Triangles only. If these Triangles are non-trivially indexed, they are
        * changed to be trivially indexed.  */
      class SmoothTraverser : public ExclusiveTraverser
      {
        public:
          //! Constructor
          DP_SG_ALGORITHM_API SmoothTraverser(void);

          //! Destructor
          DP_SG_ALGORITHM_API virtual ~SmoothTraverser(void);

          //! Set the crease angle for smoothing.
          DP_SG_ALGORITHM_API void  setCreaseAngle( float creaseAngle );

          //! Get the crease angle for smoothing.
          DP_SG_ALGORITHM_API float getCreaseAngle( ) const;

          REFLECTION_INFO_API( DP_SG_ALGORITHM_API, SmoothTraverser );
          BEGIN_DECLARE_STATIC_PROPERTIES
            DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( CreaseAngle );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          //! doApply override
          DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

          //! Handle a GeoNode object.
          /** The GeoNode is the primary object to smooth.  */
          DP_SG_ALGORITHM_API virtual void  handleGeoNode( dp::sg::core::GeoNode *p              //!<  GeoNode to handle
                                              );

          //! Handle a Primitive object.
          DP_SG_ALGORITHM_API virtual void  handlePrimitive( dp::sg::core::Primitive *p        //!<  Primitive to handle
                                            );

          /*! \brief Test whether this Object should be optimized
           *  \param p A pointer to the constant dp::sg::core::Object to test. */
          DP_SG_ALGORITHM_API bool optimizationAllowed( dp::sg::core::PrimitiveSharedPtr const& p );

        private:
          void flattenPrimitive( dp::sg::core::Primitive *p );

        private:
          float                                         m_creaseAngle;
          std::shared_ptr<DestrippingTraverser>         m_destrippingTraverser;
          std::vector<dp::sg::core::PrimitiveSharedPtr> m_primitives;
      };

    } // namespace algorithm
  } // namespace sp
} // namespace dp
