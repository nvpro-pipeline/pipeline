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
/** @file */

#include <dp/sg/core/nvsgapi.h>
#include <dp/math/Vecnt.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/Camera.h>
#include <dp/sg/core/Group.h>
#include <dp/sg/core/Node.h>
#include <dp/sg/core/TextureHost.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Container class to hold all scene related information.
       *  \par Namespace: dp::sg::core
       *  \remarks The Scene represents the loaded/created scene. It contains the ambient and back
       *  color, predefined cameras (views), predefined camera animations, and the tree of objects
       *  (the scene graph).
       *  \note The Scene holds a number of predefined cameras, but the ViewState holds the active
       *  camera.
       *  \sa Camera, SceneLoader, SceneSaver, ViewState */
      class Scene : public Object
      {
        public:
          /*! \brief The container type of the cameras */
          typedef std::vector<CameraSharedPtr>                                            CameraContainer;

          /*! \brief The iterator over the CameraContainer */
          typedef ConstIterator<Scene,CameraContainer::iterator>                 CameraIterator;

          /*! \brief The const iterator over the CameraContainer */
          typedef ConstIterator<Scene,CameraContainer::const_iterator>           CameraConstIterator;

        public:
          DP_SG_CORE_API static SceneSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~Scene();

        public:
          /*! \brief Get the global ambient color.
           *  \return A reference to the constant ambient color.
           *  \remarks This color represents light that is not from any particular source. The default
           *  is light grey (0.2,0.2,0.2).
           *  \sa getBackColor, setAmbientColor */
          DP_SG_CORE_API const dp::math::Vec3f & getAmbientColor() const;

          /*! \brief Set the global ambient color.
           *  \param color A reference to the constant ambient color to set.
           *  \remarks This color represents light that is not from any particular source. The default
           *  is light grey (0.2,0.2,0.2).
           *  \sa getAmbientColor, setBackColor */
          DP_SG_CORE_API void setAmbientColor( const dp::math::Vec3f & color );

          /*! \brief Get the background color.
           *  \return A reference to the constant background color.
           *  \remarks This color is used to clear the viewport. The default is medium grey
           *  (0.4,0.4,0.4,1.0).
           *  \sa getAmbientColor, setBackColor */
          DP_SG_CORE_API const dp::math::Vec4f & getBackColor()  const;

          /*! \brief Set the background color.
           *  \param color A reference to the constant background color to set.
           *  \remarks This color is used to clear the viewport with. The default is medium grey
           *  (0.4,0.4,0.4,1.0).
           *  \sa getBackColor, setAmbientColor, */
          DP_SG_CORE_API void setBackColor( const dp::math::Vec4f & color );

           /*! \brief return the handle to background texture image.
          *  \return A TextureHost 
          *  \sa setBackImage */
          DP_SG_CORE_API const TextureHostSharedPtr & getBackImage() const;

          /*! brief Set the background texture image
           * \param image The TextureHost to be set as background.
           * \remarks The reference count for \a image is incremented. The texture is applied to the entire viewport. 
           * \sa getBackImage */
          DP_SG_CORE_API void setBackImage( const TextureHostSharedPtr & image );


          /*! \brief Get the number of cameras in this Scene.
           *  \return The number of cameras in this Scene.
           *  \sa beginCameras, endCameras, addCamera, removeCamera, clearCameras, findCamera */
          DP_SG_CORE_API unsigned int getNumberOfCameras() const;

          /*! \brief Get a const iterator to the first camera in this Scene.
           *  \return A const iterator to the first camera in this Scene.
           *  \sa getNumberOfCameras, endCameras, addCamera, removeCamera, clearCameras, findCamera */
          DP_SG_CORE_API CameraConstIterator beginCameras() const;

          /*! \brief Get an iterator to the first camera in this Scene.
           *  \return An iterator to the first camera in this Scene.
           *  \sa getNumberOfCameras, endCameras, addCamera, removeCamera, clearCameras, findCamera */
          DP_SG_CORE_API CameraIterator beginCameras();

          /*! \brief Get a const iterator that points just beyond the end of the camera in this Scene.
           *  \return A const iterator that points just beyond the end of the camera in this Scene.
           *  \sa getNumberOfCameras, beginCameras, addCamera, removeCamera, clearCameras, findCamera */
          DP_SG_CORE_API CameraConstIterator endCameras() const;

          /*! \brief Get an iterator that points just beyond the end of the camera in this Scene.
           *  \return An iterator that points just beyond the end of the camera in this Scene.
           *  \sa getNumberOfCameras, beginCameras, addCamera, removeCamera, clearCameras, findCamera */
          DP_SG_CORE_API CameraIterator endCameras();

          /*! \brief Adds a camera to this Scene.
           *  \param camera Specifies the camera to add
           *  \return An iterator that points to the position where \a camera was added.
           *  \sa getNumberOfCameras, beginCameras, endCameras, removeCamera, clearCameras, findCamera */
          DP_SG_CORE_API CameraIterator addCamera( const CameraSharedPtr & camera );

          /*! \brief Remove a camera from this Scene.
           *  \param camera The camera to remove from this Scene.
           *  \return \c true, if the camera has been removed from this Scene, otherwise \c false.
           *  \sa getNumberOfCameras, beginCameras, endCameras, addCamera, clearCameras, findCamera */
          DP_SG_CORE_API bool removeCamera( const CameraSharedPtr & camera );

          /*! \brief Remove a camera from this Scene.
           *  \param sci An iterator to the camera to remove from this Scene.
           *  \return An iterator pointing to the new location of the camera that followed the one removed by
           *  this function call, which is endCameras() if the operation removed the last camera.
           *  \sa getNumberOfCameras, beginCameras, endCameras, addCamera, clearCameras, findCamera */
          DP_SG_CORE_API CameraIterator removeCamera( const CameraIterator & sci );

          /*! \brief Remove all cameras from this Scene.
           *  \sa getNumberOfCameras, beginCameras, endCameras, addCamera, removeCamera, findCamera */
          DP_SG_CORE_API void clearCameras();

          /*  \brief Find a specified camera in this Scene.
           *  \param camera The camera to find.
           *  \return A const iterator to the found camera in this Scene.
           *  \sa getNumberOfCameras, beginCameras, endCameras, addCamera, removeCamera, clearCameras */
          DP_SG_CORE_API CameraConstIterator findCamera( const CameraSharedPtr & camera ) const;

          /*  \brief Find a specified camera in this Scene.
           *  \param camera The camera to find.
           *  \return An iterator to the found camera in this Scene.
           *  \sa getNumberOfCameras, beginCameras, endCameras, addCamera, removeCamera, clearCameras */
          DP_SG_CORE_API CameraIterator findCamera( const CameraSharedPtr & camera );

          /*! \brief Returns the root Node of the Scene.
           *  \return The root Node of the Scene.
           *  \sa setRootNode */
          const NodeSharedPtr & getRootNode() const;

          /*! \brief Sets the root Node of the Scene.
           *  \param root The Node to set as the root node.
           *  \remarks The reference count of \a root is incremented, and the reference count of any
           *  previous root node is decremented.
           *  \sa getRootNode */
          DP_SG_CORE_API void setRootNode( const NodeSharedPtr & root );

          /*! \brief Ask if this Scene contains any transparent StateAttribute.
           *  \return \c true, if the Scene contains a transparent StateAttribute, otherwise \c false. */
          DP_SG_CORE_API bool containsTransparency() const;

          /*! \brief Get the bounding box of the scene.
           *  \return The bounding box of the scene. If the root node is invalid, the bounding box
           *  is invalid. */
          DP_SG_CORE_API virtual dp::math::Box3f getBoundingBox() const;
      
          /*! \brief Get the bounding sphere of the scene.
           *  \return The bounding sphere of the scene. If the root node is invalid, the bounding sphere
           *  is invalid. */
          DP_SG_CORE_API virtual dp::math::Sphere3f getBoundingSphere() const;

        protected:
          /*! \brief Default-constructs a Scene.
           *  \remarks The Scene initially has an ambient color of light grey (0.2, 0.2, 0.2), and a
           *  background color of medium grey (0.4, 0.4, 0.4). By default there are no cameras, 
           *  no camera animations, and no tree object attached to the Scene. */
          DP_SG_CORE_API Scene();

        public:
        // reflected properties
          REFLECTION_INFO_API( DP_SG_CORE_API, Scene );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( AmbientColor );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( BackColor );
              DP_SG_CORE_API DECLARE_STATIC_PROPERTY( RootNode );
          END_DECLARE_STATIC_PROPERTIES

        private:
          dp::math::Vec3f             m_ambientColor;
          dp::math::Vec4f             m_backColor;
          TextureHostSharedPtr        m_backImage;
          CameraContainer             m_cameras;
          NodeSharedPtr               m_root;
      };

      inline unsigned int Scene::getNumberOfCameras() const
      {
        return( dp::util::checked_cast<unsigned int>(m_cameras.size()) );
      }

      inline Scene::CameraConstIterator Scene::beginCameras() const
      {
        return( CameraConstIterator( m_cameras.begin() ) );
      }

      inline Scene::CameraIterator Scene::beginCameras()
      {
        return( CameraIterator( m_cameras.begin() ) );
      }

      inline Scene::CameraConstIterator Scene::endCameras() const
      {
        return( CameraConstIterator( m_cameras.end() ) );
      }

      inline Scene::CameraIterator Scene::endCameras()
      {
        return( CameraIterator( m_cameras.end() ) );
      }

      inline Scene::CameraConstIterator Scene::findCamera( const CameraSharedPtr & camera ) const
      {
        return( CameraConstIterator( find( m_cameras.begin(), m_cameras.end(), camera ) ) );
      }

      inline Scene::CameraIterator Scene::findCamera( const CameraSharedPtr & camera )
      {
        return( CameraIterator( find( m_cameras.begin(), m_cameras.end(), camera ) ) );
      }

      inline void Scene::clearCameras()
      {
        m_cameras.clear();
      }

      inline const NodeSharedPtr & Scene::getRootNode() const
      {
        return( m_root );
      }

      inline bool Scene::containsTransparency() const
      {
        if(m_root)                                              // First check if root node is valid before requesting read access!
        {
          return( m_root->containsTransparency() );
        }
        return false;
      }

      inline const dp::math::Vec3f& Scene::getAmbientColor() const
      {
        return( m_ambientColor );
      }

      inline void Scene::setAmbientColor( const dp::math::Vec3f &color )
      {
        if ( color != m_ambientColor )
        {
          m_ambientColor = color;
          notify( PropertyEvent( this, PID_AmbientColor ) );
        }
      }

      inline const dp::math::Vec4f& Scene::getBackColor() const
      {
        return( m_backColor );
      }

      inline void Scene::setBackColor( const dp::math::Vec4f &color )
      {
        if ( color != m_backColor )
        {
          m_backColor = color;
          notify( PropertyEvent( this, PID_BackColor ) );
        }
      }

      inline const TextureHostSharedPtr & Scene::getBackImage() const
      {
        return(m_backImage);
      }

      inline dp::math::Box3f Scene::getBoundingBox() const
      {
        return( m_root ? m_root->getBoundingBox() : dp::math::Box3f() );
      }

      inline dp::math::Sphere3f Scene::getBoundingSphere() const
      {
        return( m_root ? m_root->getBoundingSphere() : dp::math::Sphere3f() );
      }

    } // namespace core
  } // namespace sg
} // namespace dp

