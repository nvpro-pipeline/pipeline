// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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
#pragma warning(push)
#pragma warning( disable : 4503 )   // decorated name length exceeded, name was truncated

#include <dp/sg/algorithm/Config.h>
#include <dp/sg/algorithm/OptimizeTraverser.h>
#include <dp/sg/core/Primitive.h>
#include <dp/util/Flags.h>
#include <list>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      /*! \brief Traverser that combines a number of Objects, if appropriate.
       *  \remarks There are a number of different objects types combinable:\n
       *  - All geometries of all GeoNodes under one Group are combined into a single GeoNode.
       *  - All LODs with the same ranges under one Group into a single LOD.
       *  - Consecutive ranges in an LOD that reference the same Node into one range.
       *  - All PrimitveSets of the same type bound to the same StateSet are combined into one. */
      class CombineTraverser : public OptimizeTraverser
      {
        public:
          enum class Target
          {
            GEONODE     = BIT0,   //!< CombineTarget GeoNode: combine compatible GeoNode objects into one.
            LOD         = BIT1,   //!< CombineTarget LOD: combine compatible LOD objects into one.
            LOD_RANGES  = BIT2,   //!< CombineTarget LOD Range: combine identical adjacent LOD levels into one.
            TRANSFORM   = BIT3,   //!< CombineTarget Transform: combine compatible Transform objects into one.
            ALL         = ( GEONODE | LOD | LOD_RANGES | TRANSFORM )
          };

          typedef dp::util::Flags<Target> TargetMask;

        public:
          DP_SG_ALGORITHM_API static bool areCombinable( dp::sg::core::GeoNodeSharedPtr const& p0, dp::sg::core::GeoNodeSharedPtr const& p1, bool ignoreNames );
          DP_SG_ALGORITHM_API static bool areCombinable( dp::sg::core::GroupSharedPtr const& p0, dp::sg::core::GroupSharedPtr const& p1, bool ignoreNames );
          DP_SG_ALGORITHM_API static bool areCombinable( dp::sg::core::LODSharedPtr const& p0, dp::sg::core::LODSharedPtr const& p1, bool ignoreNames );
          DP_SG_ALGORITHM_API static bool areCombinable( dp::sg::core::ObjectSharedPtr const& p0, dp::sg::core::ObjectSharedPtr const& p1, bool ignoreNames );
          DP_SG_ALGORITHM_API static bool areCombinable( dp::sg::core::PrimitiveSharedPtr const& p0, dp::sg::core::PrimitiveSharedPtr const& p1, bool ignoreNames );
          DP_SG_ALGORITHM_API static bool areCombinable( dp::sg::core::TransformSharedPtr const& p0, dp::sg::core::TransformSharedPtr const& p1, bool ignoreNames );
          DP_SG_ALGORITHM_API static bool areCombinable( dp::sg::core::VertexAttributeSetSharedPtr const& p0, dp::sg::core::VertexAttributeSetSharedPtr const& p1, bool ignoreNames );

        public:
          //! Constructor
          DP_SG_ALGORITHM_API CombineTraverser( void );

          //! Destructor
          DP_SG_ALGORITHM_API virtual ~CombineTraverser( void );

          /*! \brief Get the bitmask describing the targets to combine.
           *  \return A bitmask describing the targets to combine. */
          DP_SG_ALGORITHM_API TargetMask getCombineTargets() const;

          /*! \brief Set the bitmask describing the targets to combine.
           *  \param mask The bitmask describing the targets to combine. */
          DP_SG_ALGORITHM_API void setCombineTargets( TargetMask mask );

          //! Get the 'ignore acceleration builder' hint.
          /** If the 'ignore acceleration builder' hint is set, combinable objects with different hints are
            * still considered for combining.
            * \return true if the names will be ignored, otherwise false */
          DP_SG_ALGORITHM_API bool getIgnoreAccelerationBuilderHints() const;

          //! Set the 'ignore acceleration builder' hint.
          /** If the 'ignore acceleration builder' hint is set, combinable objects with different hints are
            * still considered for combining. */
          DP_SG_ALGORITHM_API void setIgnoreAccelerationBuilderHints( bool ignore );

          REFLECTION_INFO_API( DP_SG_ALGORITHM_API, CombineTraverser );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( CombineTargets );
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( IgnoreAccelerationBuilderHints );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          //! Cleanup temporary memory.
          DP_SG_ALGORITHM_API virtual void postApply( const dp::sg::core::NodeSharedPtr & root );

          //! Combine all GeoNodes and LODs directly underneath this Billboard.
          DP_SG_ALGORITHM_API virtual void handleBillboard( dp::sg::core::Billboard * p );

          //! Combine all GeoNodes and LODs directly underneath this Group.
          DP_SG_ALGORITHM_API virtual void handleGroup( dp::sg::core::Group * p );

          //! Combine all consecutive LOD ranges referencing the same Node.
          DP_SG_ALGORITHM_API virtual void handleLOD( dp::sg::core::LOD * p );

          //! Combine all GeoNodes and LODs directly underneath this Transform.
          DP_SG_ALGORITHM_API virtual void handleTransform( dp::sg::core::Transform * p );

        private:
          void combineGeoNodes( dp::sg::core::Group *p );
          void combineLODs( dp::sg::core::Group *p );
          void combineLODRanges( dp::sg::core::LOD *p );
          void combineTransforms( dp::sg::core::Group *p );
          void filterMultiple( dp::sg::core::Group *p );

          // for Primitive combining:
          unsigned int combine( const dp::sg::core::PrimitiveSharedPtr & p0, std::vector<unsigned int> & combinedIndices
                              , unsigned int indexOffset, const dp::sg::core::PrimitiveSharedPtr & p1 );
          void combine( const dp::sg::core::VertexAttributeSetSharedPtr & vash0, const dp::sg::core::VertexAttributeSetSharedPtr & vash1 );
          unsigned int gatherIndices( const dp::sg::core::PrimitiveSharedPtr & primitive, unsigned int * newIndices, unsigned int offset );
          unsigned int reduceVertexAttributeSet( const dp::sg::core::PrimitiveSharedPtr & p );
          dp::sg::core::VertexAttributeSetSharedPtr reduceVertexAttributeSet( const dp::sg::core::VertexAttributeSetSharedPtr & p, const std::vector<unsigned int> & indexMap, unsigned int foundIndices );
          dp::sg::core::VertexAttributeSetSharedPtr reduceVertexAttributeSet( const dp::sg::core::VertexAttributeSetSharedPtr & p, unsigned int offset, unsigned int count );

        private:
          TargetMask              m_combineTargets;
          bool                    m_ignoreAccelerationBuilderHints;
          std::set<const void *>  m_objects;      //!< A set of pointers to hold all objects already encountered.
      };

      inline CombineTraverser::TargetMask operator|( CombineTraverser::Target bit0, CombineTraverser::Target bit1 )
      {
        return CombineTraverser::TargetMask( bit0 ) | bit1;
      }

      inline CombineTraverser::TargetMask CombineTraverser::getCombineTargets() const
      {
        return( m_combineTargets );
      }

      inline void CombineTraverser::setCombineTargets( TargetMask mask )
      {
        if ( mask != m_combineTargets )
        {
          m_combineTargets = mask;
          notify( PropertyEvent( this, PID_CombineTargets ) );
        }
      }

      inline bool CombineTraverser::getIgnoreAccelerationBuilderHints() const
      {
        return( m_ignoreAccelerationBuilderHints );
      }

      inline void CombineTraverser::setIgnoreAccelerationBuilderHints( bool ignore )
      {
        if ( ignore != m_ignoreAccelerationBuilderHints )
        {
          m_ignoreAccelerationBuilderHints = ignore;
          notify( PropertyEvent( this, PID_IgnoreAccelerationBuilderHints ) );
        }
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp

namespace dp
{
  namespace util
  {
    /*! \brief Specialization of the TypedPropertyEnum template for type CombineTraverser::TargetMask. */
    template <> struct TypedPropertyEnum<dp::sg::algorithm::CombineTraverser::TargetMask>
    {
      enum { type = Property::Type::UINT };
    };
  }
}

#pragma warning(pop)
