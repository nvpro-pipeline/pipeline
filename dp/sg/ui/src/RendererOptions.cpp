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


#include <dp/sg/ui/RendererOptions.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {

      BEGIN_REFLECTION_INFO( RendererOptions )
        DERIVE_STATIC_PROPERTIES( RendererOptions, Reflection );
      END_REFLECTION_INFO

      RendererOptionsSharedPtr RendererOptions::create()
      {
        return( std::shared_ptr<RendererOptions>( new RendererOptions() ) );
      }

      dp::sg::core::HandledObjectSharedPtr RendererOptions::clone() const
      {
        return( std::shared_ptr<RendererOptions>( new RendererOptions( *this ) ) );
      }

      RendererOptions::RendererOptions()
      {
        // create a new PropertyListImpl and use pointer as id
        m_propertyLists = new dp::util::PropertyListChain( false );

        m_dynamicProperties = new dp::util::PropertyListImpl( false );
        m_dynamicProperties->setId( reinterpret_cast<size_t>(m_dynamicProperties) );
        m_propertyLists->addPropertyList( m_dynamicProperties );
      }

      RendererOptions::RendererOptions( const RendererOptions &rhs )
        : HandledObject( rhs )
        , m_dynamicProperties(0)
      {
        for ( dp::util::PropertyListChain::PropertyListConstIterator it = m_propertyLists->beginPropertyLists(); it != m_propertyLists->endPropertyLists(); ++it )
        {
          if ( (*it)->getId() == reinterpret_cast<size_t>(rhs.m_dynamicProperties) )
          {
            m_dynamicProperties = reinterpret_cast<dp::util::PropertyListImpl*>( *it );
            m_dynamicProperties->setId( reinterpret_cast<size_t>(m_dynamicProperties) );
          }
        }
      }

      RendererOptions::~RendererOptions()
      {
        m_propertyLists->removePropertyList( m_dynamicProperties );
        delete m_dynamicProperties;
      }

      std::string RendererOptions::getAnnotation( const std::string &property) const
      {
        return m_propertyLists->getProperty( property )->getAnnotation( );
      }

    } // namespace ui
  } // namespace sg
} // namespace dp
