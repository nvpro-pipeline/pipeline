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
#include <dp/sg/algorithm/TransformStack.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      //! Traverser that can handle the model view transform stack.
      /** Use this class as a base for traversers that need to maintain the
        * current model view transformation. 
        * \note All ModelViewTraverser based classes need a valid \link dp::sg::ui::ViewState dp::sg::ui::ViewState \endlink.
        * If the ViewState is not valid the runtime behaviour is not defined. Call setViewState() 
        * with a valid ViewState prior calling apply().*/
      class SharedModelViewTraverser : public SharedTraverser
      {
        protected:
          //! Protected constructor to prevent instantiation of a ModelViewTraverser.
          /** \note A SharedModelViewTraverser doesn't change anything in the scene graph, but might be used by a modifying traverser.
            * therefore, it gets the readOnly flag as a parameter to pass to Traverser. */
          DP_SG_ALGORITHM_API SharedModelViewTraverser();

          //! Protected destructor to prevent instantiation of a SharedModelViewTraverser.
          DP_SG_ALGORITHM_API virtual ~SharedModelViewTraverser();

          //! Provide special treatment of a Billboard node.
          /** On a Billboard the modelview matrix is modified, the children are traversed, and the modelview matrix is
            * restored. */
          DP_SG_ALGORITHM_API virtual void handleBillboard( const dp::sg::core::Billboard *p   //!<  Billboard to handle
                                               );

          //! Provide special treatment of a Transform node.
          /** On a Transform the modelview matrix is modified, the children are traversed, and the modelview matrix is
            * restored. */
          DP_SG_ALGORITHM_API virtual void handleTransform( const dp::sg::core::Transform *p          //!<  Transform to handle
                                              );

          //! Handles actions to take between transform stack adjustment and traversal.
          /** In this base class, this is a NOP.  */
          DP_SG_ALGORITHM_API virtual bool preTraverseTransform( const dp::math::Trafo *p        //!< Trafo of node to traverse next 
                                                    );

          //! Handles actions to take between traversal and transform stack adjustment.
          /** When this function returns true, the subtree beneath is traversed. Otherwise it isn't.
            * \return true */
          DP_SG_ALGORITHM_API virtual void postTraverseTransform( const dp::math::Trafo *p       //!< Trafo of node that was traversed immediately before this call. 
                                                    );

          /*! \brief Handle any type of Camera on traversal.
           *  \param camera The read-locked camera to traverse.
           *  \remarks This function is called by the framework while traversing any type of Camera. On
           *  traversal, no or one camera will be traversed. The stack of transforms is initialized with
           *  the world-to-view and the view-to-clip matrices. Moreover, the camera is stored, referenced,
           *  and locked for further usage. It will be unlocked and unreferenced in doApply.
           *  \sa lockCamera, unlockCamera */
          DP_SG_ALGORITHM_API virtual void traverseCamera(const dp::sg::core::Camera * camera);

        protected:
          dp::sg::algorithm::TransformStack m_transformStack; //!< stack of transformations that holds the actual transformations while traversing a tree
      };

      //! Traverser that can handle the model view transform stack.
      /** Use this class as a base for traversers that need to maintain the
      * current model view transformation. 
      * \note All ModelViewTraverser based classes need a valid \link dp::sg::ui::ViewState dp::sg::ui::ViewState \endlink.
      * If the ViewState is not valid the runtime behaviour is not defined. Call setViewState() 
      * with a valid ViewState prior calling apply().*/
      class ExclusiveModelViewTraverser : public ExclusiveTraverser
      {
      protected:
        //! Protected constructor to prevent instantiation of a ModelViewTraverser.
        /** \note A ExclusiveModelViewTraverser doesn't change anything in the scene graph, but might be used by a modifying traverser.
        * therefore, it gets the readOnly flag as a parameter to pass to Traverser. */
        DP_SG_ALGORITHM_API ExclusiveModelViewTraverser();

        //! Protected destructor to prevent instantiation of a ExclusiveModelViewTraverser.
        DP_SG_ALGORITHM_API virtual ~ExclusiveModelViewTraverser();

        //! Provide special treatment of a Billboard node.
        /** On a Billboard the modelview matrix is modified, the children are traversed, and the modelview matrix is
        * restored. */
        DP_SG_ALGORITHM_API virtual void handleBillboard( dp::sg::core::Billboard *p   //!<  Billboard to handle
          );

        //! Provide special treatment of a Transform node.
        /** On a Transform the modelview matrix is modified, the children are traversed, and the modelview matrix is
        * restored. */
        DP_SG_ALGORITHM_API virtual void handleTransform( dp::sg::core::Transform *p          //!<  Transform to handle
          );

        //! Handles actions to take between transform stack adjustment and traversal.
        /** In this base class, this is a NOP.  */
        DP_SG_ALGORITHM_API virtual bool preTraverseTransform( const dp::math::Trafo *p        //!< Trafo of node to traverse next 
          );

        //! Handles actions to take between traversal and transform stack adjustment.
        /** When this function returns true, the subtree beneath is traversed. Otherwise it isn't.
        * \return true */
        DP_SG_ALGORITHM_API virtual void postTraverseTransform( const dp::math::Trafo *p       //!< Trafo of node that was traversed immediately before this call. 
          );

        /*! \brief Handle any type of Camera on traversal.
         *  \param camera The write-locked camera to traverse.
         *  \remarks This function is called by the framework while traversing any type of Camera. On
         *  traversal, no or one camera will be traversed. The stack of transforms is initialized with
         *  the world-to-view and the view-to-clip matrices. Moreover, the camera is stored, referenced,
         *  and locked for further usage. It will be unlocked and unreferenced in doApply.
         *  \sa lockCamera, unlockCamera */
        DP_SG_ALGORITHM_API virtual void traverseCamera(dp::sg::core::Camera * camera);

      protected:
        dp::sg::algorithm::TransformStack m_transformStack; //!< stack of transformations that holds the actual transformations while traversing a tree
      };

    } // namespace algorithm
  } // namespace sg
} // namespace dp
