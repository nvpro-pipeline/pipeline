// Copyright NVIDIA Corporation 2010-2011
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

#include <dp/sg/ui/Config.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/HandledObject.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {
      SHARED_PTR_TYPES( RendererOptions );

      /*! \brief   This class is a container for RendererOptions.
          \remarks Use SceneRenderer*::addRendererOptions( const RendererOptionsSharedPtr &options ) to initialize the
                   options for a specific SceneRenderer. It is possible to store the options for multiple renderers at
                   the same time. All options of a SceneRenderer* should be prefix with the name of the SceneRenderer
                   to ensure that no collisions between options of different Renderers occur.
                   The properties and values should be stored by savers if possible. It is possible to add custom attributes
                   which contain information like last active renderer. Those attributes can be evaluated by the application
                   to initialize a Window.
      !*/
      class RendererOptions : public dp::sg::core::HandledObject
      {
      public:
        DP_SG_UI_API static dp::sg::ui::RendererOptionsSharedPtr create();

        DP_SG_UI_API virtual dp::sg::core::HandledObjectSharedPtr clone() const;

        virtual ~RendererOptions();

      public:
        /*! \brief Add a new property.
            \param name Name of the property.
            \param value Initial value of the property.
            \remarks Behaviour when adding two properties with the same name is undefined.
        !*/
        template <typename ValueType> void addProperty( const std::string &name, const std::string &annonation, const ValueType &value );

        /* \brief Get the value of a property.
           \param name Name of the property to get the value for.
           \param annotation An arbitrary annotation string.
           \returns Vale of the given property name.
           \remarks This is a shortcut for getValue(getProperty(name));
        !*/
        template <typename ValueType> ValueType getValue( const std::string &name ) const;

        /*! \brief Set the value of a property.
            \param name Name of the property to set the value for.
            \param value Value to set for the given property name.
            \remarks This is a shortcut for setValue(getProperty(name), value);.
        !*/
        template <typename ValueType> void setValue( const std::string &name, const ValueType &value );

        /*! \brief Get the annotation of this property. 
            \param name Name of the property to get the annotation for.
            \return A string with the annotation of this property.
        !*/
        DP_SG_UI_API std::string getAnnotation( const std::string &name) const;

        using Reflection::getValue;
        using Reflection::setValue;

        REFLECTION_INFO_API( DP_SG_UI_API, RendererOptions );
        BEGIN_DECLARE_STATIC_PROPERTIES
        END_DECLARE_STATIC_PROPERTIES

      protected:
        RendererOptions();
        RendererOptions( const RendererOptions &rhs );

      private:
        dp::util::PropertyListImpl *m_dynamicProperties;
      };

      /*! \brief Base class for classes that need RendererOptions.
       *  \remarks Use this class as base class for every class that needs to keep RendererOptions.
      !*/
      class RendererOptionsProvider
      {
      public:
        RendererOptionsProvider() : m_rendererOptions( RendererOptions::create() )  {}
        virtual ~RendererOptionsProvider()                                          {}

        //! Add all renderer options required to the given rendererOptions object.
        DP_SG_UI_API virtual void addRendererOptions( const RendererOptionsSharedPtr& rendererOptions );

        //! Set this renderOptions object as current render options. 
        DP_SG_UI_API virtual void setRendererOptions( const RendererOptionsSharedPtr& rendererOptions );

      protected:
        //! Overwrite this function if you need to add options
        DP_SG_UI_API virtual void doAddRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr& rendererOptions ) = 0;

        //! Overwrite this function if you need to re-validate property ids on rendererOptions change
        DP_SG_UI_API virtual void doSetRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr& rendererOptions ) = 0;

      protected:
        RendererOptionsSharedPtr m_rendererOptions;
      };

      template <typename ValueType> void RendererOptions::addProperty( const std::string &name, const std::string &annotation, const ValueType &value )
      {
        DP_ASSERT(!getProperty(name) && "Property has already been added");
        m_dynamicProperties->addProperty( name, new dp::util::TypedPropertyValue<ValueType>( dp::util::SEMANTIC_VALUE, annotation, true, value ) );
      }

      template <typename ValueType> ValueType RendererOptions::getValue( const std::string &name ) const
      {
        DP_ASSERT( getProperty(name) && "Property is not available" );
        return Reflection::getValue<ValueType>( getProperty( name ) );
      }

      template <typename ValueType> void RendererOptions::setValue( const std::string &name, const ValueType &value )
      {
        DP_ASSERT( getProperty(name) && "Property is not available" );
        return Reflection::setValue<ValueType>( getProperty( name ), value );
      }

      inline void RendererOptionsProvider::addRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr& rendererOptions )
      {
        if( rendererOptions )
        {
          doAddRendererOptions(rendererOptions);
        }
      }

      inline void RendererOptionsProvider::setRendererOptions( const dp::sg::ui::RendererOptionsSharedPtr& rendererOptions )
      {
        if( rendererOptions && rendererOptions != m_rendererOptions )
        {
          // new renderer options, add needed options
          addRendererOptions( rendererOptions );

          // give derived classes the opportunity to update themselves
          doSetRendererOptions( rendererOptions );

          // update renderer options object
          m_rendererOptions = rendererOptions;
        }
      }

    } // namespace ui
  } // namespace sg
} // namespace dp

