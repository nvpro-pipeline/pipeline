// Copyright NVIDIA Corporation 2002-2006
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

#include <dp/sg/algorithm/BaseRenderElement.h>
#include <dp/sg/algorithm/RenderElementPool.h>
#include <dp/sg/core/Group.h>
#include <vector>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      /*! \brief Helper class to hold the clip planes valid for a number of PrimitiveSets.
       *  \remarks This class holds the complete clip plane information valid for the objects being
       *  beneath a Group in the scene tree.
       *  \sa BaseRenderElement */
      class ClipPlanesElement : public BaseRenderElement
      {
        public:
          /*! \brief Default constructor.
           *  \remarks The default constructor initializes the corresponding Group to be NULL. It is
           *  used to initialize the stack of ClipPlanesElements maintained by the RenderTraverser. */
          DP_SG_ALGORITHM_API ClipPlanesElement();

          /*! \brief Constructor of a ClipPlanesElement.
           *  \param group A pointer to the Group that adds at least one clip plane.
           *  \param modelToWorld A reference to the corresponding constant model to world
           *  transformation matrix.
           *  \param rhs A reference to the constant ClipPlanesElement to get the first clip planes from.
           *  \remarks The RenderTraverser maintains a stack of ClipPlaneElmenents. Whenever a Group
           *  with at least one clip plane is encountered on traversal, a new ClipPlaneElement is pushed
           *  on that stack, holding all the clip planes being valid at that point, plus the clip planes
           *  defined in that Group. */
          DP_SG_ALGORITHM_API ClipPlanesElement( const dp::sg::core::GroupSharedPtr & group, const dp::math::Mat44f & modelToWorld
                                    , const ClipPlanesElement &rhs );

          /*! \brief Get the clip plane at index \a index.
           *  \param index The index of the clip plane to get.
           *  \return The clip plane at index \a index.
           *  \note The behavior is undefined, if \a index is larger or equal to the number of clip
           *  planes in this ClipPlanesElement.
           *  \sa getNumberOfClipPlanes, getModelToWorld */
          DP_SG_ALGORITHM_API const nvmath::Plane3f & getClipPlane( unsigned int index ) const;

          /*! \brief Get the Group corresponding to this ClipPlanesElement.
           *  \return The Group corresponding to this ClipPlanesElement.
           *  \remarks This is used in RenderTraverser::postTraverseGroup to ensure that this is a
           *  Group that added some clip planes. */
          DP_SG_ALGORITHM_API const dp::sg::core::GroupWeakPtr & getGroup() const;

          /*! \brief Get the model-to-world transformation matrix at index \a index.
           *  \param index The index of the model-to-world transformation matrix to get.
           *  \return The model-to-world transformation matrix corresponding to the clip plane at index
           *  \a index.
           *  \note The behavior is undefined, if \a index is larger or equal to the number of clip
           *  planes in this ClipPlanesElement.
           *  \sa getNumberOfClipPlanes, getClipPlane */
          DP_SG_ALGORITHM_API const dp::math::Mat44f & getModelToWorld( unsigned int index ) const;

          /*! \brief Get the number of clip planes in this ClipPlanesElement.
           *  \return The number of clip planes in this ClipPlanesElement.
           *  \remarks For each of the clip planes, there is a corresponding model-to-world
           *  transformation matrix.
           *  \sa getClipPlane, getModelToWorld */
          DP_SG_ALGORITHM_API unsigned int getNumberOfClipPlanes() const;

        private:
          dp::sg::core::GroupWeakPtr            m_group;
          std::vector<nvmath::Plane3f>  m_clipPlanes;
          std::vector<dp::math::Mat44f> m_modelToWorld;
      };

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // non-member functions
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      /*! \brief Helper function to create an empty ClipPlanesElement using a given RenderElementPool.
       *  \param pool The memory pool to create the the RenderElement in.
       *  \return A pointer to the newly created ClipPlanesElement.
       *  \sa constructRenderElement, RenderElementPool */
      inline ClipPlanesElement * createClipPlanesElement( RenderElementPool * pool )
      {
        return( constructRenderElement<ClipPlanesElement>( pool ) );
      }

      /*! \brief Helper function to create a ClipPlanesElement using a given RenderElementPool.
       *  \param pool The memory pool to create the the RenderElement in.
       *  \param group A pointer to the Group that adds at least one clip plane.
       *  \param modelToWorld A reference to the corresponding constant model to world
       *  transformation matrix.
       *  \param rhs A reference to the constant ClipPlanesElement to get the first clip planes from.
       *  \return A pointer to the newly created ClipPlanesElement.
       *  \sa constructRenderElement, RenderElementPool */
      inline ClipPlanesElement * createClipPlanesElement( RenderElementPool * pool
                                                        , const dp::sg::core::GroupSharedPtr & group
                                                        , const dp::math::Mat44f & modelToWorld
                                                        , const ClipPlanesElement & rhs )
      {
        return( constructRenderElement<ClipPlanesElement,const dp::sg::core::GroupSharedPtr &, const dp::math::Mat44f &
                                      , const ClipPlanesElement &>( pool, group, modelToWorld, rhs ) );
      }

      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      // inlines
      // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      inline ClipPlanesElement::ClipPlanesElement()
        : m_group(NULL)
      {
      }

      inline ClipPlanesElement::ClipPlanesElement( const dp::sg::core::GroupSharedPtr & group
                                                 , const dp::math::Mat44f & modelToWorld
                                                 , const ClipPlanesElement &rhs )
        : BaseRenderElement(rhs)
        , m_clipPlanes(rhs.m_clipPlanes)
        , m_modelToWorld(rhs.m_modelToWorld)
        , m_group(group.get())
      {
        DP_ASSERT(m_group);
        GroupLock g(group);
        for ( unsigned int i=0 ; i<g->getNumberOfClipPlanes() ; i++ )
        {
          if ( g->isClipPlaneActive(i) )
          {
            m_clipPlanes.push_back( g->getClipPlane(i) );
            m_modelToWorld.push_back( modelToWorld );
          }
        }
      }

      inline const nvmath::Plane3f & ClipPlanesElement::getClipPlane( unsigned int index ) const
      {
        DP_ASSERT( index < m_clipPlanes.size() );
        return( m_clipPlanes[index] );
      }

      inline const dp::sg::core::GroupWeakPtr & ClipPlanesElement::getGroup() const
      {
        return( m_group );
      }

      inline const dp::math::Mat44f & ClipPlanesElement::getModelToWorld( unsigned int index ) const
      {
        DP_ASSERT( index < m_modelToWorld.size() );
        return( m_modelToWorld[index] );
      }

      inline unsigned int ClipPlanesElement::getNumberOfClipPlanes() const
      {
        return( checked_cast<unsigned int>(m_clipPlanes.size()) );
      }

    } // namespace algorithm
  } // namspace sg
} // namespace dp
