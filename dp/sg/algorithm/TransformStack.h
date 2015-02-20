// Copyright NVIDIA Corporation 2002-2012
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

#include <dp/math/Matmnt.h>
#include <stack>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      //! Utility class to hold a stack of concatenated transformation matrices.
      /** This class makes it easy to maintain all steps of the transformation pipeline from model space via world space and
        * view space to clip space (and vice versa).  \n
        * Individiual objects are defined in an local coordinate system called \e model space. There is often a hierarchy of
        * modeling coordinate systems that are maintained by this class by using \c popModelToWorld and \c pushModelToWorld.
        * \n
        * Objects are then transformed into the \e world space. \n
        * The \e view space is used to define a view volume. This system is used so that, with the eye or camera at the
        * origin looking toward -z, decreasing values of z are farther away from the eye, x is to the right and y is up. \n
        * From \e view space, we next go to the \e clip space, that is we're using the projection to get logical 2D device
        * coordinates (In contrast to physical device coordinates, which additionally need the viewport information).
        * \par
        * With these terms, the transformation pipeline looks like that:
        * \code
        *   model <-> world <-> view <-> clip
        * \endcode
        */
      class TransformStack
      {
        public:
          //! Default constructor.
          /** Initializes the model <-> world transformation stacks.  */
          TransformStack();

        public:
          //! Get the transformation from \e clip space to \e model space.
          const dp::math::Mat44f  & getClipToModel( void )  const;

          //! Get the transformation from \e clip space to \e view space.
          const dp::math::Mat44f  & getClipToView( void )   const;

          //! Get the transformation from \e model space to \e clip space.
          const dp::math::Mat44f  & getModelToClip( void )  const;

          //! Get the transformation from \e model space to \e view space.
          const dp::math::Mat44f  & getModelToView( void )  const;

          //! Get the transformation from \e model space to \e world space.
          const dp::math::Mat44f  & getModelToWorld( void ) const;

          //! Get the transformation from \e view space to \e clip space.
          const dp::math::Mat44f  & getViewToClip( void )   const;

          //! Get the transformation from \e view space to \e model space.
          const dp::math::Mat44f  & getViewToModel( void )  const;

          //! Get the transformation from \e view space to \e world space.
          const dp::math::Mat44f  & getViewToWorld( void )  const;

          //! Get the transformation from \e world space to \e model space.
          const dp::math::Mat44f  & getWorldToModel( void ) const;

          //! Get the transformation from \e world space to \e view space.
          const dp::math::Mat44f  & getWorldToView( void )  const;

          //! Get the depth of the model <-> world transformation stack.
          unsigned int getStackDepth( void )   const;

          //! Pop the top element of the model <-> world transformation stack.
          void      popModelToWorld( void );

          //! Push a new pair of matrices on the model <-> world transformation stack.
          /** \note \a modelWorld and \a worldModel should be inverse to each other to get correct results. */
          void      pushModelToWorld( const dp::math::Mat44f &modelWorld    //!<  additional model to world transformation 
                                             , const dp::math::Mat44f &worldModel    //!<  additional world to model transformation
                                             );

          //! Set the pair of matrices for the view <-> clip transformation.
          /** \note \a viewClip and \a clipView should be inverse to each other to get correct results. */
          void      setViewToClip( const dp::math::Mat44f &viewClip         //!<  new view to clip transformation (projection)
                                          , const dp::math::Mat44f &clipView         //!<  new clip to view transformation
                                          );

          //! Set the pair of matrices for the world <-> view transformation.
          /** \note \a worldView and \a viewWorld should be inverse to each other to get correct results. */
          void      setWorldToView( const dp::math::Mat44f &worldView       //!<  new world to view transformation
                                           , const dp::math::Mat44f &viewWorld       //!<  new view to world transformation
                                           );

        private :
          std::stack<dp::math::Mat44f>  m_modelWorld; //!<  stack of concatenated model to world matrices
          std::stack<dp::math::Mat44f>  m_worldModel; //!<  stack of concatenated world to model matrices
          dp::math::Mat44f              m_worldView;  //!<  world to view matrix
          dp::math::Mat44f              m_viewWorld;  //!<  view to world matrix
          dp::math::Mat44f              m_viewClip;   //!<  view to eye matrix (projection)
          dp::math::Mat44f              m_clipView;   //!<  eye to view matrix (inverse projection)

          mutable dp::math::Mat44f  m_clipModel;
          mutable dp::math::Mat44f  m_modelClip;
          mutable dp::math::Mat44f  m_modelView;
          mutable dp::math::Mat44f  m_viewModel;
          mutable bool              m_clipModelValid;
          mutable bool              m_modelClipValid;
          mutable bool              m_modelViewValid;
          mutable bool              m_viewModelValid;
      };


      inline  TransformStack::TransformStack()
        : m_clipModelValid(false)
        , m_modelClipValid(false)
        , m_modelViewValid(false)
        , m_viewModelValid(false)
      {
        m_modelWorld.push( dp::math::cIdentity44f );
        m_worldModel.push( dp::math::cIdentity44f );
      }

      inline  const dp::math::Mat44f  & TransformStack::getClipToModel( void )  const
      {
        if ( ! m_clipModelValid )
        {
          m_clipModel = m_clipView * getViewToModel();
          m_clipModelValid = true;
        }
        return( m_clipModel );
      }

      inline  const dp::math::Mat44f  & TransformStack::getClipToView( void )   const
      {
        return( m_clipView );
      }

      inline  const dp::math::Mat44f  & TransformStack::getModelToClip( void )  const
      {
        if ( ! m_modelClipValid )
        {
          m_modelClip = getModelToView() * m_viewClip;
          m_modelClipValid = true;
        }
        return( m_modelClip );
      }

      inline  const dp::math::Mat44f  & TransformStack::getModelToView( void )  const
      {
        if ( ! m_modelViewValid )
        {
          m_modelView = m_modelWorld.top() * m_worldView;
          m_modelViewValid = true;
        }
        return( m_modelView );
      }

      inline  const dp::math::Mat44f  & TransformStack::getModelToWorld( void ) const
      {
        return( m_modelWorld.top() );
      }
  
      inline  const dp::math::Mat44f  & TransformStack::getViewToClip( void )   const
      {
        return( m_viewClip );
      }
  
      inline  const dp::math::Mat44f  & TransformStack::getViewToModel( void )  const
      {
        if ( ! m_viewModelValid )
        {
          m_viewModel = m_viewWorld * m_worldModel.top();
          m_viewModelValid = true;
        }
        return( m_viewModel );
      }
  
      inline  const dp::math::Mat44f  & TransformStack::getViewToWorld( void )  const
      {
        return( m_viewWorld );
      }
  
      inline  const dp::math::Mat44f  & TransformStack::getWorldToModel( void ) const
      {
        return( m_worldModel.top() );
      }
  
      inline  const dp::math::Mat44f  & TransformStack::getWorldToView( void )  const
      {
        return( m_worldView );
      }

      inline  unsigned int  TransformStack::getStackDepth( void ) const
      {
        DP_ASSERT( m_modelWorld.size() == m_worldModel.size() );
        return( dp::checked_cast<unsigned int>(m_modelWorld.size()) );
      }

      inline  void  TransformStack::popModelToWorld( void )
      {
        m_modelWorld.pop();
        m_worldModel.pop();

        m_clipModelValid = false;
        m_modelClipValid = false;
        m_modelViewValid = false;
        m_viewModelValid = false;
      }

      inline  void  TransformStack::pushModelToWorld( const dp::math::Mat44f &modelWorld, const dp::math::Mat44f &worldModel )
      {
        m_modelWorld.push( modelWorld * m_modelWorld.top() );
        m_worldModel.push( m_worldModel.top() * worldModel );

        m_clipModelValid = false;
        m_modelClipValid = false;
        m_modelViewValid = false;
        m_viewModelValid = false;
      }

      inline  void  TransformStack::setViewToClip( const dp::math::Mat44f &viewClip, const dp::math::Mat44f &clipView )
      {
        m_viewClip = viewClip;
        m_clipView = clipView;

        m_clipModelValid = false;
        m_modelClipValid = false;
      }

      inline  void  TransformStack::setWorldToView( const dp::math::Mat44f &worldView, const dp::math::Mat44f &viewWorld )
      {
        m_worldView = worldView;
        m_viewWorld = viewWorld;

        m_clipModelValid = false;
        m_modelClipValid = false;
        m_modelViewValid = false;
        m_viewModelValid = false;
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
