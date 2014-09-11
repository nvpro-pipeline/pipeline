// Copyright NVIDIA Corporation 2011-2012
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

#include <string>
#include <map>
#include <vector>
#include <iostream>

#include <dp/rix/gl/RiXGL.h>

#include <dp/util/Observer.h>

#include <GL/glew.h>


namespace dp
{
  namespace rix
  {
    namespace gl
    {
      class ContainerDescriptorGL : public dp::rix::core::ContainerDescriptor
      {
      public:

        struct ParameterInfo
        {
          std::string   m_name;
          dp::rix::core::ContainerParameterType m_type;
          unsigned int  m_size;           // max( 1, arraysize ) * m_elementsize
          unsigned int  m_componentSize;  // sizeof(float, int, double...
          unsigned int  m_arraySize;
          unsigned int  m_elementSize;
          unsigned int  m_offset;         // offset inside container m_data
        };

        ContainerDescriptorGL( RiXGL *renderer, size_t numParameters, dp::rix::core::ProgramParameter* parameters );

        dp::rix::core::ContainerEntry generateEntry( unsigned short index )
        {
          // generate a unique container entry, dependent on container descriptor and parameter index
          return (dp::rix::core::ContainerEntry)(m_id | index);
        }

        dp::rix::core::ContainerEntry getEntry( const char * name )
        {
          // TODO Sort array of ParameterInfos for binary search.
          for ( size_t i=0; i<m_parameterInfos.size(); ++i )
          {
            if ( m_parameterInfos[i].m_name == name )
            {
              return generateEntry( static_cast<unsigned int>(i) );
            }
          }
          std::cerr << "dp::rix::gl warn: parameter " << name << " does not exist." << std::endl;
          return generateEntry(~0);
        }

        unsigned short getIndex( dp::rix::core::ContainerEntry entry )
        {
          assert( m_id == (entry & (~0 << 16))  );
          return entry & 0xFFFF;
        }

        unsigned int               m_id;
        static unsigned int        m_freeId;
        std::vector<ParameterInfo> m_parameterInfos;
        unsigned int               m_size;

        RiXGL*                     m_renderer;
      };

      class ContainerGL : public dp::rix::core::Container, public dp::util::Subject, public dp::util::Observer
      {
      public:
        class Event : public dp::util::Event
        {
        public:
          Event( ContainerGLHandle handle )
            : m_handle( handle )
          {

          }

          ContainerGLHandle getHandle() const
          {
            return m_handle;
          }

        private:
          ContainerGLHandle m_handle;
        };

      public:
        ContainerGL( ContainerDescriptorGLSharedHandle const & desc );
        ~ContainerGL();

        void setData( dp::rix::core::ContainerEntry entry, const dp::rix::core::ContainerData& containerData );

        ContainerDescriptorGLSharedHandle m_descriptor;
        unsigned int m_size;
        void *       m_data;

        // only used inside the sort function to prevent lookups.
        // sort function expects m_bucket to be empty, m_count to be 0, also resets them after use
        std::vector<GeometryInstanceGLHandle> m_bucket;
        unsigned int m_count;

        struct ParameterDataBuffer
        {
          // specified by user
          BufferGLHandle m_bufferHandle; 
          size_t         m_offset;
          size_t         m_length;

        };

        struct ParameterDataSampler
        {
          SamplerHandle m_samplerHandle;
        };

        struct ParameterDataImage
        {
          TextureGLHandle m_textureHandle;
          int             m_level;
          bool            m_layered;
          int             m_layer;
          GLenum          m_access;
        };

      protected:
        virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload );
        virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

        void setData( const ContainerDescriptorGL::ParameterInfo& descriptor, const dp::rix::core::ContainerDataRaw&     containerData );
        void setData( const ContainerDescriptorGL::ParameterInfo& descriptor, const dp::rix::core::ContainerDataBuffer&  containerData );
        void setData( const ContainerDescriptorGL::ParameterInfo& descriptor, const dp::rix::core::ContainerDataSampler& containerData );
        void setData( const ContainerDescriptorGL::ParameterInfo& descriptor, const dp::rix::core::ContainerDataImage&   containerData );
      };
    } // namespace gl
  } // namespace rix
} // namespace dp

