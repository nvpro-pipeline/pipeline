// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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

      /*! \brief Traverser to search for a tree object by class type or by name.
        * You want to apply a SearchTraverser on a scene or a certain node to search for tree objects by name or by class type.
        * If you search for objects by class type you can configure the SearchTraverser to search for an explicit class type or
        * for objects that have the specified class typ as base class.
        */
      class SearchTraverser : public SharedTraverser
      {
        public:
          /*! \brief Default-constructs a SearchTraverser
           */
          DP_SG_ALGORITHM_API SearchTraverser();

          //! Destructor
          DP_SG_ALGORITHM_API virtual ~SearchTraverser(void);

          /*! \brief Returns paths for all objects found.
           * \return The function returns for each object found during traversal the full path from the starting node
           * down to the found object. If you're only interested in the found objects itself, you should consider to
           * use getResults instead.
           * \sa getResults
           */
          DP_SG_ALGORITHM_API std::vector<dp::sg::core::PathSharedPtr> const& getPaths();

          /*! \brief Returns all objects found.
           * \return The function returns all objects found during traversal.
           */
          DP_SG_ALGORITHM_API const std::vector<dp::sg::core::ObjectSharedPtr> & getResults();

          /*! \brief Sets a class name as serach criterion.
           * \param name Specifies the class name to use as search criterion.
           * \remarks The class name needs to be fully qualified, like "class dp::sg::core::LightSource",
           * for a subsequent search to work as expected. Just using "LightSource" in this case would
           * not give the expected results. By default the class name is set to an empty string, which
           * would not yield any usable results if you intend to search objects by class name.
           * Also by default the search does only consider objects of the explicit class type specified
           * by the class name. With setBaseClassSearch you can configure the search to also consider
           * objects of types derived from the class specified by class name. If you, in addition to a
           * class name, also specified an object name as search criterion, only objects that match both,
           * the class name and the object name will be returned as search results.
           * \sa setObjectName, setBaseClassSearch */
          DP_SG_ALGORITHM_API void setClassName( const std::string& name );

          /*! \brief Returns the class name set as search criterion.
           * \return The function retuns the class name last set using setClassName or an empty string
           * if no class name was specified as search criterion before.
           * \sa setClassName
           */
          DP_SG_ALGORITHM_API const std::string& getClassName() const;

          /*! \brief Sets an object name as search criterion.
           * \param objectName Specifies the object name to use as search criterion.
           * \remarks If you specify an object name as search criterion, that object name is
           * compared to the corresponding search candidate's object name assigned to it by means of
           * the Object::setName member function. If you, in addition to an object name, also specified
           * a class name as search criterion, only objects that match both, the class name and the object name
           * will be returned as search results.
           * \sa Object::setName, setClassName
           */
          DP_SG_ALGORITHM_API void setObjectName(const std::string& objectName);

          /*! \brief Returns the object name set as search criterion.
           * \return The function retuns the object name last set using setObjectName or an empty string
           * if no object name was specified as search criterion before.
           * \sa setObjectName
           */
          DP_SG_ALGORITHM_API const std::string& getObjectName() const;

          /*! \brief Specifies a particular object to search for.
           * \param ptr Object to search for.
           * \remarks If a particular object is used as search criterion, other search criteria like
           * class name or object name will be ignored.
           */
          DP_SG_ALGORITHM_API void setObjectPointer( dp::sg::core::ObjectSharedPtr const& ptr );

          /*! \brief Returns the object to search for.
           * \return The function returns the object last set using setObjectPtr.
           */
          DP_SG_ALGORITHM_API dp::sg::core::ObjectWeakPtr getObjectPointer() const;

          /*! Configures the base class search criterion option.
           * \param enable If true, the search will also consider objects of types derived from the class
           * specified through setClassName. If false, only objects of the explicit type specified through
           * setClassName will be considered. By default, this option is set to false.
           * \sa setClassName
           */
          DP_SG_ALGORITHM_API void setBaseClassSearch( bool enable );

          /*! \brief Returns the enable state of the base class search option.
           * \return The function returns the enable state of the base class search option
           * last set using setBaseClassSearch.
           * \sa setBaseClassSearch
           */
          DP_SG_ALGORITHM_API bool getBaseClassSearch() const ;

          REFLECTION_INFO_API( DP_SG_ALGORITHM_API, SearchTraverser );
          BEGIN_DECLARE_STATIC_PROPERTIES
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( ClassName );
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( ObjectName );
              DP_SG_ALGORITHM_API DECLARE_STATIC_PROPERTY( BaseClassSearch );
          END_DECLARE_STATIC_PROPERTIES

        protected:
          //! Set up the search and traverse the scene.
          DP_SG_ALGORITHM_API virtual void  doApply( const dp::sg::core::NodeSharedPtr & root );

          /*! \brief Add an Object to the list of found items.
           *  \param obj A pointer to the read-locked object to add */
          DP_SG_ALGORITHM_API void addItem( const dp::sg::core::Object* obj );

          //! Routine to handle a \link dp::sg::core::Billboard Billboard \endlink node while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleBillboard(
            const dp::sg::core::Billboard * p //!< Points to the currently visited Billboard object.
            );

          //! Routine to handle a \link dp::sg::core::GeoNode GeoNode \endlink while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleGeoNode(
            const dp::sg::core::GeoNode * p //!< Points to the currently visited GeoNode object.
          );

          //! Routine to handle a \link dp::sg::core::Group Group \endlink node while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleGroup(
            const dp::sg::core::Group * p //!< Points to the currently visited Group object.
          );

          //! Routine to handle a \link dp::sg::core::LOD LOD \endlink (Level Of Detail) node while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleLOD(
            const dp::sg::core::LOD * p //!< Points to the currently visited LOD object.
          );

          DP_SG_ALGORITHM_API virtual void handleParameterGroupData( const dp::sg::core::ParameterGroupData * p );

          DP_SG_ALGORITHM_API void handlePipelineData( const dp::sg::core::PipelineData * p );

          DP_SG_ALGORITHM_API virtual void handleSampler( const dp::sg::core::Sampler * p );

          //! Routine to handle a \link dp::sg::core::Switch Switch \endlink node while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleSwitch(
            const dp::sg::core::Switch * p //!< Points to the currently visited Switch object.
          );

          //! Routine to handle a \link dp::sg::core::Transform Transform \endlink node while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleTransform(
            const dp::sg::core::Transform * p //!< Points to the currently visited Transform object.
          );

          DP_SG_ALGORITHM_API virtual void handleLightSource( const dp::sg::core::LightSource * p );

          //! Routine to handle a \link dp::sg::core::Primitives Primitives \endlink object while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handlePrimitive(
            const dp::sg::core::Primitive * p //!< Points to the currently visited Primitive object.
          );

          //! Routine to handle a \link dp::sg::core::ParallelCamera ParallelCamera \endlink object while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleParallelCamera(
            const dp::sg::core::ParallelCamera * p //!< Points to the currently visited ParallelCamera object.
          );

          //! Routine to handle a \link dp::sg::core::PerspectiveCamera PerspectiveCamera \endlink object while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handlePerspectiveCamera(
            const dp::sg::core::PerspectiveCamera * p //!< Points to the currently visited PerspectiveCamera object.
          );

          //! Routine to handle a \link dp::sg::core::MatrixCamera MatrixCamera \endlink object while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleMatrixCamera(
            const dp::sg::core::MatrixCamera * p //!< Points to the currently visited MatrixCamera object.
          );

          //! Routine to handle a \link dp::sg::core::VertexAttributeSet VertexAttributeSet \endlink object while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleVertexAttributeSet( const dp::sg::core::VertexAttributeSet *vas );

          //! Routine to handle a \link dp::sg::core::IndexSet IndexSet \endlink object while traversing the scene graph.
          DP_SG_ALGORITHM_API virtual void handleIndexSet( const dp::sg::core::IndexSet * iset );

          //! Search through a Camera object.
          /** This function is called when base class searching is enabled. */
          DP_SG_ALGORITHM_API virtual void search( const dp::sg::core::Camera *p             //!<  Camera to search through
                                      );

          //! Search through a Group object.
          /** This function is called when Group is searched through or base class searching is enabled. */
          DP_SG_ALGORITHM_API virtual void search( const dp::sg::core::Group *p                //!<  Group to search through
                                      );

          //! Search through a LightSource object.
          /** This function is called when base class searching is enabled. */
          DP_SG_ALGORITHM_API virtual void search( const dp::sg::core::LightSource *p        //!<  LightSource to search through
                                      );

          //! Search through a Node object.
          /** This function is called when base class searching is enabled. */
          DP_SG_ALGORITHM_API virtual void search( const dp::sg::core::Node *p               //!<  Node to search through
                                      );

          //! Search through an Object object.
          /** This function is called when base class searching is enabled. */
          DP_SG_ALGORITHM_API virtual void search( const dp::sg::core::Object *p             //!<  Object to search through
                                      );

          //! Search through a Transform object.
          /** This function is called when Transform is searched or a base class searching is enabled. */
          DP_SG_ALGORITHM_API virtual void search( const dp::sg::core::Transform *p            //!<  Transform to search through
                                      );

          //! Search through a VertexAttributeSet object.
          /** This function is called when a VertexAttributeSet is searched or a base class searching is enabled. */
          DP_SG_ALGORITHM_API virtual void search( const dp::sg::core::VertexAttributeSet * p );

        private:
          bool searchObject(const dp::sg::core::Object* p, const std::string &classNameToHandle);

          dp::sg::core::PathSharedPtr                 m_currentPath;
          std::string                                 m_className;
          std::set<dp::sg::core::ObjectSharedPtr>     m_foundObjects;
          std::string                                 m_objectName;
          dp::sg::core::ObjectSharedPtr               m_objectPointer;
          std::vector<dp::sg::core::PathSharedPtr>    m_paths;
          std::vector<dp::sg::core::ObjectSharedPtr>  m_results;
          bool                                        m_searchBaseClass;
      };

      inline void SearchTraverser::setClassName( const std::string& name )
      {
        if ( m_className != name )
        {
          m_className = name;
          notify( PropertyEvent( this, PID_ClassName ) );
        }
      }

      inline const std::string& SearchTraverser::getClassName( ) const
      {
        return m_className;
      }

      inline void SearchTraverser::setObjectName(const std::string &objectName)
      {
        if ( m_objectName != objectName )
        {
          m_objectName = objectName;
          notify( PropertyEvent( this, PID_ObjectName ) );
        }
      }

      inline const std::string& SearchTraverser::getObjectName() const
      {
        return m_objectName;
      }

      inline void SearchTraverser::setObjectPointer( dp::sg::core::ObjectSharedPtr const& ptr )
      {
        m_objectPointer = ptr;
      }

      inline dp::sg::core::ObjectWeakPtr SearchTraverser::getObjectPointer( ) const
      {
        return m_objectPointer;
      }

      inline void SearchTraverser::setBaseClassSearch( bool enable )
      {
        if ( m_searchBaseClass != enable )
        {
          m_searchBaseClass = enable;
          notify( PropertyEvent( this, PID_BaseClassSearch ) );
        }
      }

      inline bool SearchTraverser::getBaseClassSearch( ) const
      {
        return m_searchBaseClass;
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
