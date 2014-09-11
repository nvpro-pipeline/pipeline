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
#include <dp/sg/algorithm/ModelViewTraverser.h>
#include <dp/math/math.h>
#include <set>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      //! AppTraverser Class
      /** This class provides the mechanism to apply changes and update the scene graph. 
        * (The 'apply' step of the application.) Use an AppTraverser derived class as a unique 
        * entry point for manipulating nodes in the scene, such as data from gadgets, 
        * animations, and interactions.
        * \note The current implementation takes care of the animation frames, camera clip planes
        * highlighting, etc...
        * \note Needs a valid ViewState, which in turns holds a valid camera. Call setViewState prior to apply().*/

      class AppTraverser : public ExclusiveModelViewTraverser
      {
      public:
        //! Default constructor.
        DP_SG_ALGORITHM_API AppTraverser();

      protected:
        //!Default destructor.
        DP_SG_ALGORITHM_API virtual ~AppTraverser();

        //! Determines if the AppTraverser really should traverse the tree.
        /** Depending on some internal data, the traversal is recognized to be necessary.
          * The AppTraverser needs to traverse if the ViewState, the Scene, or the animation
          * frame changed. */
        DP_SG_ALGORITHM_API virtual bool needsTraversal( const dp::sg::core::NodeSharedPtr & root ) const;

        //! doApply override
        DP_SG_ALGORITHM_API virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

        // data accessible for derived classes
        unsigned int  m_animationFrame; //!< Current animation frame.

      public:

        /*! \brief Force traversal on the next frame.
         *  \remarks The AppTraverser tries to traverse the scene as seldom as possible. In some
         *  circumstances it might be necessary to force a traversal. That can be done with this function.
         *  \sa needsTraversal */
        DP_SG_ALGORITHM_API void forceTraversal();

        //////////////////////////////////////////////////////

      private:

        // Some data to cache so we can decide if we really have to apply the animation data. 
        // This does not help to prevent from unneeded traversal steps since we can't 
        // decide for derived classes. 
        dp::sg::ui::ViewStateWeakPtr  m_pLastViewState;
        dp::sg::core::NodeWeakPtr     m_pLastRoot;
        bool                          m_forcedTraversal;
        dp::sg::core::Transform*      m_currentTransform;
      };

      inline void AppTraverser::forceTraversal()
      {
        m_forcedTraversal = true;
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
