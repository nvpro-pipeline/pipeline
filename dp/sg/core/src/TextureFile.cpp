// Copyright NVIDIA Corporation 2012
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


#include <dp/Types.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/util/Observer.h>
#include <boost/make_shared.hpp>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /************************************************************************/
      /* TextureFileCache                                                     */
      /************************************************************************/
      class TextureFileCache : public dp::util::Observer
      {
      public:
        static TextureFileSharedPtr getTextureFile( std::string const & filename, TextureTarget textureTarget );

        virtual void onNotify( dp::util::Event const & event, dp::util::Payload * payload );
        virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

      private:
        DEFINE_PTR_TYPES( Payload );
        class Payload : public dp::util::Payload
        {
        public:
          static PayloadSharedPtr create();
          virtual ~Payload();

        protected:
          Payload();

        public:
          std::string                      m_filename; // the filename is required for the onDestroyed event.
          dp::sg::core::TextureFileWeakPtr m_textureFile;
        };

        static TextureFileCache & instance();
        
        typedef std::map<std::string const, std::shared_ptr<Payload> > TextureFileMap;
        TextureFileMap m_cache;
      };


      TextureFileCache::PayloadSharedPtr TextureFileCache::Payload::create()
      {
        return( std::shared_ptr<Payload>( new Payload() ) );
      }

      TextureFileCache::Payload::Payload()
      {
      }

      TextureFileCache::Payload::~Payload()
      {
      }


      TextureFileCache & TextureFileCache::instance()
      { 
        static TextureFileCache cache;
        return cache;
      }

      TextureFileSharedPtr TextureFileCache::getTextureFile( std::string const & filename, TextureTarget textureTarget )
      {
        // check if the filename is already in the cache.
        TextureFileCache &self = instance();
        TextureFileMap::iterator it = self.m_cache.find(filename);

        if ( it == self.m_cache.end() )
        {
          // if not create a new TextureFile object
          PayloadSharedPtr payload = Payload::create();
          payload->m_filename = filename;
          TextureFileSharedPtr textureFile = std::shared_ptr<TextureFile>( new TextureFile( filename, textureTarget ) );
          payload->m_textureFile = textureFile.getWeakPtr();
          payload->m_textureFile->attach( &self, payload.getWeakPtr() );
          it = self.m_cache.insert( std::make_pair( filename, payload) ).first;
          return textureFile;
        }
        else
        {
          // else assert that the textureTarget hasn't changed.
          DP_ASSERT( it->second->m_textureFile->getTextureTarget() == textureTarget );
          return it->second->m_textureFile->getSharedPtr<TextureFile>();
        }
      }

      void TextureFileCache::onNotify( dp::util::Event const & event, dp::util::Payload * payload )
      {
        // nothing to do
      }

      void TextureFileCache::onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload )
      {
        // last instance of TextureFileCache has been released. Remove entry from cache to avoid infinite growing.
        instance().m_cache.erase( static_cast<Payload*>(payload)->m_filename );
      }

      /************************************************************************/
      /* TextureFile                                                          */
      /************************************************************************/
      TextureFileSharedPtr TextureFile::create( const std::string& filename, TextureTarget textureTarget )
      {
        return TextureFileCache::getTextureFile( filename, textureTarget );
      }

      HandledObjectSharedPtr TextureFile::clone() const
      {
        return( std::shared_ptr<TextureFile>( new TextureFile( *this ) ) );
      }

      TextureFile::TextureFile( const std::string& filename, TextureTarget textureTarget )
        : Texture( textureTarget )
        , m_filename( filename )
      {
      }

      bool TextureFile::isEquivalent( TextureSharedPtr const& texture, bool deepCompare ) const
      {
        if ( texture == this )
        {
          return( true );
        }

        bool equi = texture.isPtrTo<TextureFile>() && Texture::isEquivalent( texture, deepCompare );
        if ( equi )
        {
          TextureFileSharedPtr const& tf = texture.staticCast<TextureFile>();
          equi = ( m_filename == tf->m_filename );
        }
        return( equi );
      }

      void TextureFile::feedHashGenerator( dp::util::HashGenerator & hg ) const
      {
        hg.update( reinterpret_cast<const unsigned char *>(m_filename.c_str()), dp::checked_cast<unsigned int>(m_filename.length()) );
        Texture::feedHashGenerator( hg );
      }

    } // namespace core
  } // namespace sg
} // namespace dp
